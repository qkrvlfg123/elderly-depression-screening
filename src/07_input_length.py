# -*- coding: utf-8 -*-
"""
07. 입력 길이별 성능 · 서빙 임계값 결정

학습 입력은 화자 한 명의 답변 여러 개를 이은 텍스트(중앙값 약 1.2만 자)지만, 서비스 입력은
하루 답변 1개(데이터 기준 중앙값 약 780자)다. 입력이 짧아지면 점수 분포가 달라지므로
  1) 서빙 임계값을 서비스 입력 분량(SERVE_CHARS=800자)으로 자른 val에서 다시 정한다
     (06과 같은 규칙: 🟡 watch = val 재현율 0.80, 🔴 refer = val 특이도 0.80)
     → saved_models/ensemble.json 의 threshold_watch / threshold_refer 를 교체
       (06이 전체 길이로 정한 값은 threshold_*_full 로 보존)
  2) test 텍스트를 앞에서부터 N자로 잘라 AUROC와 두 임계값 세트에서의 지표를 낸다.
재학습 없음.

[실행] python src/07_input_length.py --csv speakers.csv   (06 다음에 실행)
[출력] results/input_length.csv, saved_models/ensemble.json
"""
import os
import json
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

LENGTHS = (400, 800, 1600, None)  # None = 자르지 않음
SERVE_CHARS = 800                 # 서비스 입력(답변 1개) 분량


def main():
    ap = argparse.ArgumentParser(description="입력 길이별 test 성능")
    ap.add_argument("--csv", default=os.path.join(C.ROOT, "speakers.csv"))
    ap.add_argument("--models-dir", default=os.path.join(C.ROOT, "saved_models"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts = C.make_split(C.load_speakers(args.csv))
    te = parts["test"]
    va = parts["val"].copy()
    va["text"] = va["text"].str.slice(0, SERVE_CHARS)
    ens_path = os.path.join(args.models_dir, "ensemble.json")
    with open(ens_path, encoding="utf-8") as f:
        ens = json.load(f)
    full = {k: ens.get(f"threshold_{k}_full", ens[f"threshold_{k}"]) for k in ("watch", "refer")}

    preds = {n: [] for n in LENGTHS}
    val_preds = []
    for name in ens["members"]:
        path = os.path.join(args.models_dir, name)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        for n in LENGTHS:
            t = te.copy()
            if n:
                t["text"] = t["text"].str.slice(0, n)
            preds[n].append(C.predict_speakers(model, tok, t, C.MODEL_CONFIGS[name]["arch"], device))
        val_preds.append(C.predict_speakers(model, tok, va, C.MODEL_CONFIGS[name]["arch"], device))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── 1) 서빙 임계값: 서비스 입력 분량 val에서 결정 ──
    pv = pd.concat(val_preds, axis=1).mean(axis=1)
    yv = (va.set_index("speaker_id")["score"].loc[pv.index] >= 2).astype(int).values
    serve = {"watch": C.threshold_for_recall(yv, pv.values, 0.80),
             "refer": C.threshold_for_specificity(yv, pv.values, 0.80)}
    ens.update({"threshold_watch": serve["watch"], "threshold_refer": serve["refer"],
                "threshold_watch_full": full["watch"], "threshold_refer_full": full["refer"],
                "threshold_source": f"val, 앞 {SERVE_CHARS}자 (서비스 입력 분량)"})
    with open(ens_path, "w", encoding="utf-8") as f:
        json.dump(ens, f, ensure_ascii=False, indent=2)
    print(f"서빙 임계값 ({SERVE_CHARS}자 val): watch {serve['watch']:.3f} / refer {serve['refer']:.3f} "
          f"(전체 길이 val: {full['watch']:.3f} / {full['refer']:.3f})")

    # ── 2) 입력 길이별 test 성능 ──
    rows = []
    s = te.set_index("speaker_id")["score"]
    for n in LENGTHS:
        p = pd.concat(preds[n], axis=1).mean(axis=1)
        y = (s.loc[p.index] >= 2).astype(int).values
        ci = C.bootstrap_ci(y, p.values)["auroc"]
        row = {"max_chars": n or "full", "auroc": roc_auc_score(y, p.values),
               "auroc_ci_low": ci[0], "auroc_ci_high": ci[1], "score_median": float(p.median())}
        for src, th in (("full", full), (f"serve{SERVE_CHARS}", serve)):
            for tname in ("watch", "refer"):
                m = C.screening_metrics(y, p.values, th[tname])
                row.update({f"{src}_{tname}_{k}": m[k]
                            for k in ("recall", "precision", "specificity", "flagged_rate")})
        rows.append(row)
        sv = f"serve{SERVE_CHARS}"
        print(f"{str(n or 'full'):>5}자: AUROC {row['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | 서빙 임계값 → "
              f"🔴 표시 {row[sv + '_refer_flagged_rate']:.1%} 재현율 {row[sv + '_refer_recall']:.3f} "
              f"정밀도 {row[sv + '_refer_precision']:.3f} · 🟡 이상 표시 {row[sv + '_watch_flagged_rate']:.1%} "
              f"재현율 {row[sv + '_watch_recall']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "input_length.csv"), index=False)


if __name__ == "__main__":
    main()
