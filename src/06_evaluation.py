# -*- coding: utf-8 -*-
"""
06. 앙상블 구성·임계값 선택(val) → test 평가

  1) 구성별 val AUROC로 순위 → 상위 k개 평균 앙상블, k는 val AUROC가 가장 높은 값으로 선택
  2) 임계값 (val에서 결정, PHQ2>=2 / >=3 각각)
     - watch  (🟡 관찰 이상)  : val 재현율 >= 0.80 을 만족하는 가장 높은 값 — 위험군을 놓치지 않기
     - refer  (🔴 상담 권유)  : val 특이도 >= 0.80 을 만족하는 가장 낮은 값 — 가족 알림이 가는 등급이라
                                 비위험군 오경보를 20% 이내로 제한 (기획서: 임계치 초과 시에만 알림, 알림 피로 방지)
     - fixed0.5               : 참고용
  3) 위에서 정한 앙상블·임계값으로 test 평가
     AUROC(부트스트랩 95% CI) · 재현율 · 정밀도 · 특이도 · 위험군 표시 비율
  출력: results/experiments.csv(앙상블·구성별 test 점수), results/test_pred_ensemble.csv,
        results/evaluation.json, saved_models/ensemble.json (서빙용 구성·임계값)

[실행] python src/06_evaluation.py --csv speakers.csv
"""
import os
import json
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

WATCH_RECALL = 0.80
REFER_SPECIFICITY = 0.80


def load_preds(split):
    out = {}
    for name in C.MODEL_CONFIGS:
        path = os.path.join(C.PRED_DIR, f"{name}_{split}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path, dtype={"sid": str})
            out[name] = d.set_index("sid")["prob"]
    return pd.DataFrame(out)


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    ap = argparse.ArgumentParser(description="val 기반 앙상블·임계값 선택 → test 평가")
    ap.add_argument("--csv", default=os.path.join(C.ROOT, "speakers.csv"))
    ap.add_argument("--models-dir", default=os.path.join(C.ROOT, "saved_models"))
    args = ap.parse_args()

    df = C.load_speakers(args.csv).set_index("speaker_id")
    pv, pt = load_preds("val"), load_preds("test")
    names = [n for n in pv.columns if n in pt.columns]
    yv2 = (df.loc[pv.index, "score"] >= 2).astype(int).values

    # ── 1) val로만 앙상블 구성 결정 ──
    val_auc = {n: roc_auc_score(yv2, pv[n].values) for n in names}
    order = sorted(names, key=lambda n: -val_auc[n])
    k_auc = {k: roc_auc_score(yv2, pv[order[:k]].mean(axis=1).values) for k in range(1, len(order) + 1)}
    best_k = max(k_auc, key=k_auc.get)
    members = order[:best_k]
    print("val AUROC (구성):", {n: round(val_auc[n], 4) for n in order})
    print("val AUROC (상위 k 평균):", {k: round(v, 4) for k, v in k_auc.items()})
    print(f"→ 선택: 상위 {best_k}개 {members}")

    ens_v = pv[members].mean(axis=1)
    ens_t = pt[members].mean(axis=1)

    # ── 2) val로 임계값 결정 ──
    thr = {}
    for cut in C.REPORT_CUTS:
        yv = (df.loc[ens_v.index, "score"] >= cut).astype(int).values
        thr[cut] = {"watch": C.threshold_for_recall(yv, ens_v.values, WATCH_RECALL),
                    "refer": C.threshold_for_specificity(yv, ens_v.values, REFER_SPECIFICITY),
                    "fixed0.5": 0.5}

    # ── 3) test 평가 ──
    report = {"members": members, "k": best_k, "val_auroc_members": val_auc,
              "val_auroc_topk": k_auc, "rules": {"watch": f"val recall >= {WATCH_RECALL}",
                                                 "refer": f"val specificity >= {REFER_SPECIFICITY}"},
              "test": {}}
    exp = pd.read_csv(C.EXPERIMENTS_CSV)
    exp = exp[~exp["run_id"].str.startswith("holdout/ensemble")]
    exp.to_csv(C.EXPERIMENTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n===== test ({len(ens_t)}명) =====")
    for cut in C.REPORT_CUTS:
        yv = (df.loc[ens_v.index, "score"] >= cut).astype(int).values
        yt = (df.loc[ens_t.index, "score"] >= cut).astype(int).values
        auc = roc_auc_score(yt, ens_t.values)
        auc_ci = C.bootstrap_ci(yt, ens_t.values)["auroc"]
        res = {"n": int(len(yt)), "n_pos": int(yt.sum()), "auroc": auc, "auroc_ci": auc_ci,
               "thresholds": {}}
        print(f"\n[PHQ2 >= {cut}] 위험군 {yt.sum()}명 / {len(yt)}명 · AUROC {auc:.4f} 95% CI {fmt_ci(auc_ci)}")
        for tname, t in thr[cut].items():
            m = C.screening_metrics(yt, ens_t.values, t)
            ci = C.bootstrap_ci(yt, ens_t.values, t)
            res["thresholds"][tname] = {"threshold": t, **m,
                                        **{f"{k}_ci": v for k, v in ci.items() if k != "auroc"}}
            print(f"  {tname:8s} 임계값 {t:.3f}: 재현율 {m['recall']:.3f} {fmt_ci(ci['recall'])} · "
                  f"정밀도 {m['precision']:.3f} {fmt_ci(ci['precision'])} · "
                  f"특이도 {m['specificity']:.3f} {fmt_ci(ci['specificity'])} · "
                  f"위험군 표시 {m['flagged_rate']:.1%} {fmt_ci(ci['flagged_rate'])}")
            C.log_experiment({
                "run_id": f"holdout/ensemble/PHQ2>={cut}/{tname}", "kind": "ensemble",
                "config": "+".join(members), "label": f"PHQ2>={cut}",
                "val_auroc": roc_auc_score(yv, ens_v.values),
                "test_auroc": auc, "test_auroc_ci_low": auc_ci[0], "test_auroc_ci_high": auc_ci[1],
                "threshold_rule": tname, "threshold": t,
                **{f"test_{k}": m[k] for k in ("recall", "precision", "specificity", "flagged_rate")},
                "n_val": len(ens_v), "n_test": len(ens_t),
            })
        report["test"][f"PHQ2>={cut}"] = res

    # 구성별 test 점수 (선택에는 쓰지 않음, 기록용)
    yt2 = (df.loc[pt.index, "score"] >= 2).astype(int).values
    yt3 = (df.loc[pt.index, "score"] >= 3).astype(int).values
    exp = pd.read_csv(C.EXPERIMENTS_CSV)
    for n in names:
        ci = C.bootstrap_ci(yt2, pt[n].values)["auroc"]
        mask = exp["run_id"] == f"holdout/{n}"
        exp.loc[mask, "test_auroc"] = roc_auc_score(yt2, pt[n].values)
        exp.loc[mask, "test_auroc_ci_low"], exp.loc[mask, "test_auroc_ci_high"] = ci
        exp.loc[mask, "test_auroc_phq2ge3"] = roc_auc_score(yt3, pt[n].values)
    exp.to_csv(C.EXPERIMENTS_CSV, index=False, encoding="utf-8-sig")

    ens_t.rename("prob").rename_axis("sid").reset_index().to_csv(
        os.path.join(C.RESULTS, "test_pred_ensemble.csv"), index=False)
    with open(os.path.join(C.RESULTS, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    # 서빙용: 앙상블 구성원과 PHQ2>=2 기준 val 임계값
    os.makedirs(args.models_dir, exist_ok=True)
    with open(os.path.join(args.models_dir, "ensemble.json"), "w", encoding="utf-8") as f:
        json.dump({"members": members,
                   "threshold_refer": thr[2]["refer"],
                   "threshold_watch": thr[2]["watch"]}, f, indent=2)
    print(f"\n저장: results/experiments.csv, results/test_pred_ensemble.csv, results/evaluation.json, "
          f"{os.path.join(args.models_dir, 'ensemble.json')}")


if __name__ == "__main__":
    main()
