# -*- coding: utf-8 -*-
"""
05. 기준 모델(baseline) 비교와 교차검증

트랜스포머 앙상블의 AUROC를 해석하기 위한 비교 기준으로, 가벼운 기준 모델 두 개를 화자 단위로 평가한다.

  (A) 인구학 변수만 : 나이·성별·교육 → 로지스틱 회귀
  (B) 텍스트 TF-IDF : 구술 텍스트의 글자 2~3-gram TF-IDF → 로지스틱 회귀

  1) 최종본과 같은 test 화자 1,164명(02와 같은 8:2 분할)에서 평가
     → 03이 저장한 최종본 예측(results/preds/final_serving_full.csv)이 있으면 같은 화자로 짝지어 비교
  2) 전체 화자 5-fold 교차검증 (fold마다 벡터화와 모델을 처음부터 다시 학습)

설정은 하나로 고정했고 test를 보고 고른 값이 없다 (하이퍼파라미터 탐색 없음).

[실행] python src/05_baseline.py --csv speakers_clean_kssds.csv
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

SEED = 42
MAX_CHARS = 20000     # 화자당 앞에서부터 사용할 글자 수 (평균 길이 수준, 계산 시간 제한용)
FINAL_PRED = os.path.join(C.PRED_DIR, "final_serving_full.csv")


def demographics(df):
    """나이·성별·교육을 숫자 행렬로. 값이 숫자가 아니면 범주 코드로 바꾼다."""
    cols = []
    for name in ("age", "sex", "edu"):
        raw = df[name].astype(str)
        try:
            cols.append(raw.astype(float).values)
        except ValueError:
            codes = {v: i for i, v in enumerate(sorted(set(raw)))}
            cols.append(raw.map(codes).astype(float).values)
    return np.column_stack(cols)


def text_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=5, max_features=200000,
                        sublinear_tf=True, dtype=np.float32),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )


def demo_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def auc_ci(y, p):
    return {"auroc": round(float(roc_auc_score(y, p)), 4),
            "ci95": [round(v, 4) for v in C.bootstrap_ci(y, p)["auroc"]]}


def paired_diff(y, a, b, n=2000, seed=0):
    """같은 화자를 재표집해 AUROC(a) − AUROC(b) 의 95% 구간."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {"diff": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
            "ci95": [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(C.ROOT, "speakers_clean_kssds.csv"))
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "baseline_cv.json"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t0 = time.time()
    df = C.load_speakers(args.csv)
    y = df["label"].values
    text = df["text"].astype(str).str[:MAX_CHARS].values
    X_demo = demographics(df)
    print(f"화자 {len(df):,}명, 위험군 비율 {y.mean():.3f}", flush=True)
    out = {"n_speakers": int(len(df)), "positive_rate": round(float(y.mean()), 4),
           "cutoff": C.CUT, "max_chars": MAX_CHARS, "seed": SEED}

    # ---- 1) 최종본과 같은 test 화자
    te = df["speaker_id"].isin(C.final_test_ids(args.csv)).values
    tr = ~te
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_positive": int(y[te].sum())}
    p_demo = demo_model().fit(X_demo[tr], y[tr]).predict_proba(X_demo[te])[:, 1]
    res["baseline_demographics"] = auc_ci(y[te], p_demo)
    p_text = text_model().fit(text[tr], y[tr]).predict_proba(text[te])[:, 1]
    res["baseline_tfidf"] = auc_ci(y[te], p_text)
    if os.path.exists(FINAL_PRED):
        fp = pd.read_csv(FINAL_PRED, dtype={"sid": str}).set_index("sid")["raw"]
        p_final = fp.loc[df.loc[te, "speaker_id"]].values
        res["final_ensemble"] = auc_ci(y[te], p_final)
        res["final_minus_tfidf"] = paired_diff(y[te], p_final, p_text)
    out["same_test_speakers"] = res
    print("같은 test 화자 비교", json.dumps(res, ensure_ascii=False), f"({round(time.time() - t0)}초)", flush=True)

    # ---- 2) 전체 화자 5-fold 교차검증 (fold마다 다시 학습)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    cv = {"demographics": [], "tfidf": []}
    oof_text = np.zeros(len(y))
    for i, (a, b) in enumerate(skf.split(text, y), 1):
        cv["demographics"].append(float(roc_auc_score(y[b], demo_model().fit(X_demo[a], y[a]).predict_proba(X_demo[b])[:, 1])))
        p = text_model().fit(text[a], y[a]).predict_proba(text[b])[:, 1]
        oof_text[b] = p
        cv["tfidf"].append(float(roc_auc_score(y[b], p)))
        print(f"fold {i}/{args.folds}: 인구학 {cv['demographics'][-1]:.4f}, TF-IDF {cv['tfidf'][-1]:.4f} "
              f"({round(time.time() - t0)}초)", flush=True)
    out["cross_validation"] = {
        "folds": args.folds,
        "demographics": {"mean": round(float(np.mean(cv["demographics"])), 4),
                         "std": round(float(np.std(cv["demographics"])), 4),
                         "per_fold": [round(v, 4) for v in cv["demographics"]]},
        "tfidf": {"mean": round(float(np.mean(cv["tfidf"])), 4), "std": round(float(np.std(cv["tfidf"])), 4),
                  "per_fold": [round(v, 4) for v in cv["tfidf"]],
                  "out_of_fold": auc_ci(y, oof_text)},
    }
    out["seconds"] = round(time.time() - t0)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("저장:", args.out, json.dumps(out["cross_validation"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
