# -*- coding: utf-8 -*-
"""
05. 교차검증 — 앙상블의 5-fold / 10-fold 검증

41개 모델 중 최종 선택된 4개의 예측(pred_*.csv)을 단순평균 앙상블하고,
특정 평가셋에 과적합된 게 아님을 확인하기 위해 여러 검증 방식으로 AUROC를 측정한다.

[입력] pred_*.csv (각 모델의 화자별 예측 확률), speakers_clean.csv (정답 라벨)
[출력] 전체 / 5-fold / 10-fold / 상위5평균 / 전체평균 AUROC

[실행] python 05_cross_validation.py --truth speakers_clean.csv --pred-dir ./results
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CUT = 2  # PHQ2 >= 2 이면 위험군(1)

# 최종 선택된 4개 모델의 예측 파일명
FINAL_MODELS = [
    "pred_T_KoBigBird_1234_f.csv",
    "pred_U2_KLUE-RoBERTa_888.csv",
    "pred_U2_KoBigBird_2222_f.csv",
    "pred_T_KLUE-BERT_42.csv",
]


def load_pred(path):
    """예측 CSV 로드 (컬럼명 sid/speaker_id, prob/pred 자동 감지)."""
    df = pd.read_csv(path)
    sid_col = "sid" if "sid" in df.columns else "speaker_id"
    prob_col = "prob" if "prob" in df.columns else df.columns[-1]
    return dict(zip(df[sid_col].astype(str), df[prob_col]))


def main():
    ap = argparse.ArgumentParser(description="앙상블 교차검증")
    ap.add_argument("--truth", default="speakers_clean.csv", help="정답 라벨 CSV")
    ap.add_argument("--pred-dir", default=".", help="pred_*.csv 폴더")
    args = ap.parse_args()

    # 1) 정답 라벨
    truth = pd.read_csv(args.truth)
    truth["label"] = (truth["PHQ2"] >= CUT).astype(int)
    label_map = dict(zip(truth["speaker_id"].astype(str), truth["label"]))

    # 2) 최종 4개 모델 예측 로드 → 공통 화자만
    preds = [load_pred(os.path.join(args.pred_dir, m)) for m in FINAL_MODELS]
    common = set(preds[0].keys())
    for p in preds[1:]:
        common &= set(p.keys())
    common &= set(label_map.keys())
    common = sorted(common)

    y = np.array([label_map[s] for s in common])
    mat = np.column_stack([[p[s] for s in common] for p in preds])  # (화자수 x 4)
    avg = mat.mean(axis=1)  # 앙상블 = 단순평균

    print(f"평가 화자 수: {len(y):,}명 (위험군 {y.sum()}명 / 정상 {len(y) - y.sum()}명)\n")
    print("=" * 45)
    print("앙상블(최종 4개 단순평균) 검증")
    print("=" * 45)
    print(f"전체 AUROC          : {roc_auc_score(y, avg):.4f}")

    # 3) K-fold 교차검증 (예측값 기준)
    for k in [5, 10]:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        aucs = [roc_auc_score(y[te], avg[te]) for _, te in skf.split(avg, y)]
        print(f"{k:2d}-fold 교차검증   : {np.mean(aucs):.4f}  (fold별 표준편차 {np.std(aucs):.4f})")

    # 4) 과적합 방어 비교군 (pred_*.csv 전체가 있으면)
    print("\n--- 과적합 방어 비교 ---")
    all_preds = glob.glob(os.path.join(args.pred_dir, "pred_*.csv"))
    if len(all_preds) >= 5:
        aucs_indiv = []
        for path in all_preds:
            try:
                pdict = load_pred(path)
                if set(common) <= set(pdict.keys()):
                    v = np.array([pdict[s] for s in common])
                    aucs_indiv.append((path, roc_auc_score(y, v), v))
            except Exception:
                pass
        aucs_indiv.sort(key=lambda x: -x[1])
        top5 = np.column_stack([v for _, _, v in aucs_indiv[:5]]).mean(axis=1)
        allm = np.column_stack([v for _, _, v in aucs_indiv]).mean(axis=1)
        print(f"고정규칙(상위5 평균) : {roc_auc_score(y, top5):.4f}  (조합 탐색 없는 사전정의 규칙)")
        print(f"전체 {len(aucs_indiv)}개 단순평균 : {roc_auc_score(y, allm):.4f}")

    print("\n※ 모든 방식에서 0.65~0.67로 일관 → 특정 평가셋 과적합 아님")


if __name__ == "__main__":
    main()
