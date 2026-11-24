# -*- coding: utf-8 -*-
"""
09. 발표 최종본 서빙 구성 작성 + test 평가

최종본 4모델 가중치(드라이브 'modeling 최종본.zip' 의 models/)를 saved_models/final/<모델>/ 에 둔 뒤 실행한다.
  1) 각 모델 폴더에 inference.json (학습 당시 입력 설정) 과 saved_models/final/ensemble.json 을 쓴다.
     점수 보정(0.50~0.78 → 0~1)과 등급 기준(🟡 0.4 / 🔴 0.5)은 최종본 server.py 의 값을 그대로 쓴다.
     이 값들은 시연 문장으로 정한 것이라 test를 보고 정한 값이 아니다.
  2) 04 서빙과 같은 처리(chunk_text → 청크 확률 평균 → 모델 평균 → 보정, fp32, max_length 패딩)로
     test 1,164명을 한 번 평가한다. 입력: 전체 텍스트 / 앞 800자(서비스 입력 분량).

모델은 test로 epoch과 조합을 골랐던 것이므로 AUROC는 낙관적인 값이다 (재현 결과는 08 참고).
등급 규칙의 test 결과는 test와 무관하게 정한 규칙을 한 번 적용한 값이다.

[실행] python src/09_final_serving.py --csv speakers_clean_kssds.csv
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import common as C

FINAL_DIR = os.path.join(C.ROOT, "saved_models", "final")
# 학습 당시 설정 (08_reproduce_final.py 와 같음)
FINAL_MODELS = {
    "KoBigBird_1234_f": dict(arch="KoBigBird", max_len=3072, chunk_chars=6000, max_chunks=5),
    "KLUE-RoBERTa_888": dict(arch="KLUE-RoBERTa", max_len=512, chunk_chars=800, max_chunks=8),
    "KoBigBird_2222_f": dict(arch="KoBigBird", max_len=3072, chunk_chars=6000, max_chunks=5),
    "KLUE-BERT_42": dict(arch="KLUE-BERT", max_len=512, chunk_chars=800, max_chunks=8),
}
ENSEMBLE = {
    "members": list(FINAL_MODELS),
    "calibration": {"min": 0.50, "max": 0.78},
    "threshold_watch": 0.4,
    "threshold_refer": 0.5,
    "threshold_source": "발표 최종본 server.py (보정 0.50~0.78, 등급 0.4/0.5 — 시연 문장으로 결정, test 미사용)",
}
INPUTS = {"full": None, "800": 800}


def calibrate(raw):
    c = ENSEMBLE["calibration"]
    return np.clip((np.asarray(raw) - c["min"]) / (c["max"] - c["min"]), 0.0, 1.0)


def write_configs():
    for name, inf in FINAL_MODELS.items():
        with open(os.path.join(FINAL_DIR, name, "inference.json"), "w", encoding="utf-8") as f:
            json.dump(inf, f, ensure_ascii=False)
    with open(os.path.join(FINAL_DIR, "ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(ENSEMBLE, f, ensure_ascii=False, indent=2)


def predict(name, inf, texts, device):
    """04 predict_one 과 같은 처리를 배치로: chunk_text → max_length 패딩 → sigmoid → 청크 평균."""
    path = os.path.join(FINAL_DIR, name)
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    rows = [(i, c) for i, t in enumerate(texts) for c in C.chunk_text(t, inf["chunk_chars"], inf["max_chunks"])]
    bs = 4 if inf["max_len"] > 512 else 32
    probs = []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            enc = tok([r[1] for r in rows[s:s + bs]], truncation=True, max_length=inf["max_len"],
                      padding="max_length", return_tensors="pt").to(device)
            probs.append(torch.sigmoid(mdl(**enc).logits.float()[:, 0]).cpu().numpy())
    del mdl; torch.cuda.empty_cache()
    p = pd.Series(np.concatenate(probs)).groupby([r[0] for r in rows]).agg(C.aggregate)
    return p.reindex(range(len(texts))).values


def grade_summary(y, risk):
    out = {"grade_rate": {"green": float((risk < ENSEMBLE["threshold_watch"]).mean()),
                          "yellow": float(((risk >= ENSEMBLE["threshold_watch"]) &
                                           (risk < ENSEMBLE["threshold_refer"])).mean()),
                          "red": float((risk >= ENSEMBLE["threshold_refer"]).mean())}}
    for rule in ("refer", "watch"):
        thr = ENSEMBLE[f"threshold_{rule}"]
        m = C.screening_metrics(y, risk, thr)
        m["ci"] = {k: v for k, v in C.bootstrap_ci(y, risk, thr).items() if k != "auroc"}
        out[rule] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="speakers_clean_kssds.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    write_configs()
    df = C.load_speakers(args.csv)
    split = pd.read_csv(C.SPLIT_CSV, dtype={"speaker_id": str})
    te = df[df["speaker_id"].isin(split.loc[split["split"] == "test", "speaker_id"])].reset_index(drop=True)
    print(f"장치 {device} | test {len(te)}명", flush=True)

    os.makedirs(C.PRED_DIR, exist_ok=True)
    result = {"ensemble": ENSEMBLE, "n_test": len(te), "inputs": {}}
    for key, n_chars in INPUTS.items():
        texts = [t if n_chars is None else str(t)[:n_chars] for t in te["text"]]
        per = {}
        for name, inf in FINAL_MODELS.items():
            per[name] = predict(name, inf, texts, device)
            print(f"  [{key}] {name} AUROC {roc_auc_score(te['label'], per[name]):.4f}", flush=True)
        raw = np.mean(list(per.values()), axis=0)
        risk = calibrate(raw)
        pd.DataFrame({"sid": te["speaker_id"], "raw": raw, "risk": risk}).to_csv(
            os.path.join(C.PRED_DIR, f"final_serving_{key}.csv"), index=False)

        r = {"members_auroc": {n: float(roc_auc_score(te["label"], p)) for n, p in per.items()},
             "raw_percentiles": dict(zip(["p5", "p50", "p95"], map(float, np.percentile(raw, [5, 50, 95]))))}
        for cut in C.REPORT_CUTS:
            y = (te["score"] >= cut).astype(int).values
            r[f"PHQ2>={cut}"] = {"auroc": float(roc_auc_score(y, raw)),
                                 "auroc_ci": C.bootstrap_ci(y, raw)["auroc"],
                                 **grade_summary(y, risk)}
        result["inputs"][key] = r

        g = r["PHQ2>=2"]
        print(f"[{key}] 앙상블 AUROC {g['auroc']:.4f} {np.round(g['auroc_ci'], 3)} | "
              f"PHQ2>=3 {r['PHQ2>=3']['auroc']:.4f} | 원점수 p5/p50/p95 "
              f"{np.round(list(r['raw_percentiles'].values()), 3)}", flush=True)
        print(f"  등급 비율 🟢 {g['grade_rate']['green']:.1%} 🟡 {g['grade_rate']['yellow']:.1%} "
              f"🔴 {g['grade_rate']['red']:.1%}", flush=True)
        for rule, label in (("refer", "🔴"), ("watch", "🟡 이상")):
            m = g[rule]
            print(f"  {label}: 재현율 {m['recall']:.3f} 정밀도 {m['precision']:.3f} "
                  f"특이도 {m['specificity']:.3f} 표시 {m['flagged_rate']:.1%}", flush=True)
            C.log_experiment({"run_id": f"final/serve/{key}/{rule}", "kind": "final_serving",
                              "config": "+".join(FINAL_MODELS), "label": "PHQ2>=2",
                              "n_test": len(te), "test_auroc": g["auroc"],
                              "test_auroc_ci_low": g["auroc_ci"][0], "test_auroc_ci_high": g["auroc_ci"][1],
                              "threshold": ENSEMBLE[f"threshold_{rule}"], "threshold_rule": f"final_{rule}_calibrated",
                              "test_recall": m["recall"], "test_precision": m["precision"],
                              "test_specificity": m["specificity"], "test_flagged_rate": m["flagged_rate"],
                              "test_auroc_phq2ge3": r["PHQ2>=3"]["auroc"]})

    with open(os.path.join(C.RESULTS, "final_serving.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
