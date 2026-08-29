# -*- coding: utf-8 -*-
"""
06. 평가 — 최종 앙상블의 성능 지표 및 임계값 분석

불균형 데이터(정상 78%)에서 정확도(Accuracy)는 함정이므로 F2·Balanced Accuracy·recall
중심으로 평가한다. 임계값(절단점)별 recall/precision을 분석해 0.4/0.5 등급 경계의 근거를
확인하고, Youden's J 최적 임계값과 우리가 채택한 값(0.5)을 비교한다.

[입력] pred_*.csv (최종 4개 모델), speakers_clean.csv (정답)
[출력] AUROC / AUPRC / F2 / Balanced Accuracy / 임계값별 recall·precision / Youden's J

[실행] python 06_evaluation.py --truth speakers_clean.csv --pred-dir ./results
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, confusion_matrix, fbeta_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             average_precision_score, roc_curve)

CUT = 2
FINAL_MODELS = [
    "pred_T_KoBigBird_1234_f.csv",
    "pred_U2_KLUE-RoBERTa_888.csv",
    "pred_U2_KoBigBird_2222_f.csv",
    "pred_T_KLUE-BERT_42.csv",
]


def load_pred(path):
    df = pd.read_csv(path)
    sid_col = "sid" if "sid" in df.columns else "speaker_id"
    prob_col = "prob" if "prob" in df.columns else df.columns[-1]
    return dict(zip(df[sid_col].astype(str), df[prob_col]))


def main():
    ap = argparse.ArgumentParser(description="최종 앙상블 성능 평가")
    ap.add_argument("--truth", default="speakers_clean.csv", help="정답 라벨 CSV")
    ap.add_argument("--pred-dir", default=".", help="pred_*.csv 폴더")
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    truth["label"] = (truth["PHQ2"] >= CUT).astype(int)
    label_map = dict(zip(truth["speaker_id"].astype(str), truth["label"]))

    preds = [load_pred(os.path.join(args.pred_dir, m)) for m in FINAL_MODELS]
    common = set(preds[0].keys())
    for p in preds[1:]:
        common &= set(p.keys())
    common &= set(label_map.keys())
    common = sorted(common)

    y = np.array([label_map[s] for s in common])
    avg = np.column_stack([[p[s] for s in common] for p in preds]).mean(axis=1)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print("=" * 50)
    print("최종 앙상블 성능 평가")
    print(f"전체 {len(y):,}명 (위험군 {n_pos} / 정상 {n_neg}, 정상 쏠림 {n_neg / len(y) * 100:.0f}%)")
    print("=" * 50)

    # 임계값 무관 지표
    print("\n[임계값 무관]")
    print(f"  AUROC : {roc_auc_score(y, avg):.4f}")
    print(f"  AUPRC : {average_precision_score(y, avg):.4f}")

    # 임계값 0.5 기준 상세
    print("\n[임계값 0.5 기준]")
    pred = (avg >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    recall, precision, spec = tp / (tp + fn), tp / (tp + fp), tn / (tn + fp)
    print(f"  recall (재현율)    : {recall:.3f}  (위험군 {n_pos}명 중 {tp}명 포착)")
    print(f"  precision (정밀도) : {precision:.3f}  (위험판정 {tp + fp}명 중 {tp}명 실제)")
    print(f"  특이도             : {spec:.3f}")
    print(f"  F1 점수            : {fbeta_score(y, pred, beta=1):.3f}")
    print(f"  F2 점수            : {fbeta_score(y, pred, beta=2):.3f}  (recall 2배 중시)")
    print(f"  Balanced Accuracy  : {balanced_accuracy_score(y, pred):.3f}")
    print(f"  MCC                : {matthews_corrcoef(y, pred):.3f}")
    print(f"  혼동행렬: TP={tp} FP={fp} FN={fn} TN={tn}")

    # 정확도 함정 경고
    acc = (tp + tn) / len(y)
    all_neg = (y == 0).mean()
    print(f"\n  ⚠️ Accuracy {acc:.3f} — 전부 '정상'만 찍어도 {all_neg:.3f} "
          f"(불균형 함정, 대표지표로 쓰지 않음)")

    # 임계값별 분석
    print("\n[임계값별 recall / precision]")
    print(f"  {'임계값':>6} {'recall':>8} {'precision':>10} {'포착':>8}")
    for th in [0.3, 0.4, 0.5, 0.55, 0.6]:
        pr = (avg >= th).astype(int)
        if pr.sum() == 0:
            continue
        tn2, fp2, fn2, tp2 = confusion_matrix(y, pr).ravel()
        rec2 = tp2 / (tp2 + fn2)
        prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
        print(f"  {th:>6.2f} {rec2:>8.3f} {prec2:>10.3f} {tp2:>6}명")

    # Youden's J 최적 임계값
    fpr, tpr, thr = roc_curve(y, avg)
    J = tpr - fpr
    best_idx = np.argmax(J)
    print("\n[Youden's J 최적]")
    print(f"  최적 임계값 = {thr[best_idx]:.3f} (J={J[best_idx]:.3f})")
    print("  → 우리는 놓치지 않기 우선으로 이보다 낮은 0.5를 채택")


if __name__ == "__main__":
    main()
