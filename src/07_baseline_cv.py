"""
07. 기준 모델(baseline)과 실제 교차검증

트랜스포머 앙상블의 AUROC가 어느 정도의 의미를 갖는지 보려면 비교 기준이 필요하다.
이 스크립트는 가벼운 기준 모델 두 개를 화자 단위로 평가한다.

  (A) 인구학 변수만 : 나이·성별·교육 → 로지스틱 회귀
  (B) 텍스트 TF-IDF : 구술 전문의 글자 n-gram TF-IDF → 로지스틱 회귀

평가는 두 가지로 한다.

  1) 02~06 단계와 같은 평가 화자(예측 파일에 있는 1,164명)를 시험셋으로, 나머지를 학습셋으로 사용
     → 트랜스포머 예측과 같은 화자에서 직접 비교
  2) 전체 화자에 대한 5-fold 교차검증 (fold마다 벡터화와 모델을 처음부터 다시 학습)
     → 05 단계의 'fold 평균'과 달리 모델을 다시 학습하는 실제 교차검증

기준 모델은 설정을 하나로 고정했고 시험셋을 보고 고른 것이 없다(하이퍼파라미터 탐색 없음).

사용:
  python src/07_baseline_cv.py --csv speakers.csv --pred-dir ./results --out results/baseline_cv.json
"""
import argparse
import csv
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CUTOFF = 2            # PHQ-2 유사 2문항 합 >= 2 를 위험군으로 본다 (02 단계와 동일)
MAX_CHARS = 20000     # 화자당 앞에서부터 사용할 글자 수 (평균 길이 수준, 계산 시간 제한용)


def load_speakers(path):
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not (r.get("text") or "").strip() or r.get("PHQ2") in (None, ""):
                continue
            rows.append(r)
    return rows


def demographics(rows):
    """나이·성별·교육을 숫자 행렬로. 값이 숫자가 아니면 범주 코드로 바꾼다."""
    cols = []
    for name in ("age", "sex", "edu"):
        raw = [r.get(name, "") for r in rows]
        try:
            col = np.array([float(v) for v in raw])
        except ValueError:
            codes = {v: i for i, v in enumerate(sorted(set(raw)))}
            col = np.array([float(codes[v]) for v in raw])
        cols.append(col)
    return np.column_stack(cols)


def text_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=5, max_features=200000,
                        sublinear_tf=True, dtype=np.float32),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )


def demo_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() != y[b].max():
            vals.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pred-dir", default="./results")
    ap.add_argument("--out", default="results/baseline_cv.json")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    rows = load_speakers(args.csv)
    sid = np.array([str(r["speaker_id"]).strip() for r in rows])
    y = np.array([1 if int(float(r["PHQ2"])) >= CUTOFF else 0 for r in rows])
    text = np.array([r["text"][:MAX_CHARS] for r in rows], dtype=object)
    X_demo = demographics(rows)
    print(f"화자 {len(rows):,}명, 위험군 비율 {y.mean():.3f}", flush=True)

    out = {"n_speakers": int(len(rows)), "positive_rate": round(float(y.mean()), 4),
           "cutoff": CUTOFF, "max_chars": MAX_CHARS, "seed": SEED}

    # ---- 1) 02~06 단계와 같은 평가 화자에서 비교
    import glob
    import os
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "pred_*.csv")))
    preds = {}
    for pf in pred_files:
        with open(pf, encoding="utf-8-sig", newline="") as f:
            preds[os.path.basename(pf)] = {str(r["sid"]).strip(): float(r["pred"]) for r in csv.DictReader(f)}
    if preds:
        test_ids = set.intersection(*[set(p) for p in preds.values()])
        te = np.array([s in test_ids for s in sid])
        tr = ~te
        res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_positive": int(y[te].sum())}
        for name, p in preds.items():
            pv = np.array([p[s] for s in sid[te]])
            res[name] = {"auroc": round(float(roc_auc_score(y[te], pv)), 4), "ci95": boot_ci(y[te], pv)}
        ens = np.mean([[p[s] for s in sid[te]] for p in preds.values()], axis=0)
        res["transformer_mean_of_committed"] = {"auroc": round(float(roc_auc_score(y[te], ens)), 4),
                                                "ci95": boot_ci(y[te], ens), "n_models": len(preds)}
        m = demo_model().fit(X_demo[tr], y[tr])
        pd_ = m.predict_proba(X_demo[te])[:, 1]
        res["baseline_demographics"] = {"auroc": round(float(roc_auc_score(y[te], pd_)), 4), "ci95": boot_ci(y[te], pd_)}
        print("인구학 기준 모델 완료", round(time.time() - t0), "초", flush=True)
        m = text_model().fit(text[tr], y[tr])
        pt = m.predict_proba(text[te])[:, 1]
        res["baseline_tfidf"] = {"auroc": round(float(roc_auc_score(y[te], pt)), 4), "ci95": boot_ci(y[te], pt)}
        both = (ens - ens.mean()) / ens.std() + (pt - pt.mean()) / pt.std()
        res["transformer_plus_tfidf"] = {"auroc": round(float(roc_auc_score(y[te], both)), 4), "ci95": boot_ci(y[te], both),
                                         "note": "사후 결합이므로 참고용"}
        out["same_test_speakers"] = res
        print("같은 평가 화자 비교 완료", round(time.time() - t0), "초", json.dumps(res, ensure_ascii=False), flush=True)

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
        "demographics": {"mean": round(float(np.mean(cv["demographics"])), 4), "std": round(float(np.std(cv["demographics"])), 4),
                         "per_fold": [round(v, 4) for v in cv["demographics"]]},
        "tfidf": {"mean": round(float(np.mean(cv["tfidf"])), 4), "std": round(float(np.std(cv["tfidf"])), 4),
                  "per_fold": [round(v, 4) for v in cv["tfidf"]],
                  "out_of_fold_auroc": round(float(roc_auc_score(y, oof_text)), 4), "ci95": boot_ci(y, oof_text)},
    }
    out["seconds"] = round(time.time() - t0)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("저장:", args.out, flush=True)
    print(json.dumps(out["cross_validation"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
