# -*- coding: utf-8 -*-
"""
05. 화자 단위 5-fold 교차검증 — fold마다 모델을 새로 학습

  - 전체 화자를 라벨 계층화 5-fold로 나눈다 (시드 42).
  - fold마다: 나머지 4개 fold에서 12.5%를 떼어 epoch 선택용 val로 쓰고, 나머지로 학습.
    held-out fold는 학습·선택에 쓰지 않고 마지막에 한 번 평가.
  - 출력: fold별 AUROC(PHQ2>=2, >=3), 평균±표준편차, 전체 out-of-fold AUROC
    → results/experiments.csv (run_id=cv5/<구성>/...), results/preds/cv5_<구성>_oof.csv

[실행] python src/05_cross_validation.py --csv speakers.csv --config KoBigBird_1234_f
완료된 fold는 건너뛴다.
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import common as C

K = 5


def main():
    ap = argparse.ArgumentParser(description="화자 단위 5-fold 교차검증 (fold마다 재학습)")
    ap.add_argument("--csv", default=os.path.join(C.ROOT, "speakers.csv"))
    ap.add_argument("--config", required=True, choices=list(C.MODEL_CONFIGS))
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = C.load_speakers(args.csv)
    arch = C.MODEL_CONFIGS[args.config]["arch"]
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=C.SPLIT_SEED)
    os.makedirs(C.PRED_DIR, exist_ok=True)

    for f, (tr_i, te_i) in enumerate(skf.split(df, df["label"])):
        path = os.path.join(C.PRED_DIR, f"cv5_{args.config}_fold{f}.csv")
        if os.path.exists(path):
            print(f"[fold {f}] 완료됨 → 건너뜀")
            continue
        t0 = time.time()
        rest, te = df.iloc[tr_i], df.iloc[te_i].reset_index(drop=True)
        tr, va = train_test_split(rest, test_size=0.125, stratify=rest["label"],
                                  random_state=C.SPLIT_SEED)
        print(f"\n===== fold {f}: train {len(tr)} / val {len(va)} / held-out {len(te)} =====")
        model, tok, info = C.train_model(args.config, tr.reset_index(drop=True),
                                         va.reset_index(drop=True), device)
        p = C.predict_speakers(model, tok, te, arch, device)
        p.rename_axis("sid").reset_index().to_csv(path, index=False)
        s = te.set_index("speaker_id").loc[p.index, "score"]
        C.log_experiment({
            "run_id": f"cv5/{args.config}/fold{f}", "kind": "cv_fold", "config": args.config,
            "fold": f, "best_epoch": info["best_epoch"],
            "val_auroc_by_epoch": str(info["val_auroc_by_epoch"]),
            "val_auroc": max(info["val_auroc_by_epoch"]),
            "test_auroc": roc_auc_score((s >= 2).astype(int), p.values),
            "test_auroc_phq2ge3": roc_auc_score((s >= 3).astype(int), p.values),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "minutes": round((time.time() - t0) / 60, 1),
        })
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── 요약 ──
    oof = pd.concat([pd.read_csv(os.path.join(C.PRED_DIR, f"cv5_{args.config}_fold{f}.csv"),
                                 dtype={"sid": str}) for f in range(K)])
    oof.to_csv(os.path.join(C.PRED_DIR, f"cv5_{args.config}_oof.csv"), index=False)
    s = df.set_index("speaker_id").loc[oof["sid"], "score"].values
    exp = pd.read_csv(C.EXPERIMENTS_CSV)
    folds = exp[exp["run_id"].str.startswith(f"cv5/{args.config}/fold")]
    for cut, col in ((2, "test_auroc"), (3, "test_auroc_phq2ge3")):
        a = folds[col].values
        pooled = roc_auc_score((s >= cut).astype(int), oof["prob"].values)
        ci = C.bootstrap_ci((s >= cut).astype(int), oof["prob"].values)["auroc"]
        print(f"[PHQ2 >= {cut}] fold AUROC {np.round(a, 4).tolist()} → 평균 {a.mean():.4f} ± {a.std(ddof=1):.4f}"
              f" | out-of-fold {pooled:.4f} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
        C.log_experiment({
            "run_id": f"cv5/{args.config}/summary/PHQ2>={cut}", "kind": "cv_summary",
            "config": args.config, "label": f"PHQ2>={cut}",
            "test_auroc": pooled, "test_auroc_ci_low": ci[0], "test_auroc_ci_high": ci[1],
            "cv_fold_mean": a.mean(), "cv_fold_sd": a.std(ddof=1), "n_test": len(oof),
        })


if __name__ == "__main__":
    main()
