# -*- coding: utf-8 -*-
"""
공통 모듈. 평가(03)·서빙(04)·기준 모델(05)이 같은 처리와 지표를 쓰도록 한 곳에 모은다.
학습(02)은 청킹에 자체 함수(sent_chunks, 같은 문장부호 분할 규칙)를 쓰고 여기서는 지표·기록 함수만 가져다 쓴다.

  - chunk_text / aggregate : 텍스트 청킹과 화자 단위 집계 (평가·서빙 공통)
  - load_speakers / final_test_ids : 데이터 로드와 화자 단위 8:2 분할의 test 화자
  - screening_metrics / bootstrap_ci / log_experiment : 평가 지표와 기록
"""
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 충돌 방지 (Windows)

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PRED_DIR = os.path.join(RESULTS, "preds")
EXPERIMENTS_CSV = os.path.join(RESULTS, "experiments.csv")

TARGET = "PHQ2"
CUT = 2                 # 학습 라벨: PHQ2 >= 2
REPORT_CUTS = (2, 3)    # 보고 기준: PHQ2 >= 2, PHQ2 >= 3

# 키워드마다 우울점수가 다르게 기록된 화자 (라벨 불일치 → 제외)
EXCLUDE_SPEAKERS = {"20002", "20046", "20816", "20898"}


def chunk_text(text, chunk_chars, max_chunks):
    """문장 경계(. ! ?)를 유지하며 앞에서부터 chunk_chars 이하 청크를 최대 max_chunks개 만든다.
    문장부호 없이 긴 구간은 chunk_chars 단위로 자른다."""
    # 앞에서부터 최대 chunk_chars*max_chunks 자만 쓰므로, 그 2배까지만 처리 (긴 입력 대비).
    # 학습 데이터 전체에서 자르기 전과 결과가 같음을 확인함.
    text = str(text)[: 2 * chunk_chars * max_chunks]
    sents = []
    for s in re.split(r"(?<=[.!?])\s*", text):
        s = s.strip()
        while len(s) > chunk_chars:
            sents.append(s[:chunk_chars])
            s = s[chunk_chars:].strip()
        if s:
            sents.append(s)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > chunk_chars:
            chunks.append(cur)
            if len(chunks) >= max_chunks:
                cur = ""
                break
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur and len(chunks) < max_chunks:
        chunks.append(cur)
    long_enough = [c for c in chunks if len(c) > 20]
    return long_enough or chunks or [str(text)]


def aggregate(chunk_probs):
    """청크 확률 → 화자(또는 요청) 단위 확률: 평균."""
    return float(np.mean(chunk_probs))


# ── 데이터 ─────────────────────────────────────────────────────────
def load_speakers(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["text", TARGET])
    df["speaker_id"] = df["speaker_id"].astype(str)
    df = df[~df["speaker_id"].isin(EXCLUDE_SPEAKERS)].reset_index(drop=True)
    df["score"] = df[TARGET].astype(float)
    df["label"] = (df["score"] >= CUT).astype(int)
    return df


def final_test_ids(csv_path):
    """02_train_final.py 와 같은 화자 8:2 분할(random_state=42)의 test 화자 ID."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(csv_path).dropna(subset=["text", TARGET]).reset_index(drop=True)
    _, te = train_test_split(df, test_size=0.2, random_state=42)
    return set(te["speaker_id"].astype(str))


# ── 평가 ─────────────────────────────────────────────────────────
def screening_metrics(y, p, thr):
    y, pr = np.asarray(y), np.asarray(p) >= thr
    tp, fp = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum())
    fn, tn = int((~pr & (y == 1)).sum()), int((~pr & (y == 0)).sum())
    return {
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "specificity": tn / max(1, tn + fp),
        "flagged_rate": float(pr.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(y, p, thr=None, n=2000, seed=0):
    """화자 재표집 부트스트랩 95% CI. AUROC(와 thr가 주어지면 임계값 지표)."""
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    keys = ["auroc"] + (["recall", "precision", "specificity", "flagged_rate"] if thr is not None else [])
    acc = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        acc["auroc"].append(roc_auc_score(y[i], p[i]))
        if thr is not None:
            m = screening_metrics(y[i], p[i], thr)
            for k in keys[1:]:
                acc[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}


def log_experiment(row):
    """results/experiments.csv 에 한 줄 추가 (같은 run_id가 있으면 교체)."""
    os.makedirs(RESULTS, exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(EXPERIMENTS_CSV):
        old = pd.read_csv(EXPERIMENTS_CSV)
        old = old[old["run_id"] != row["run_id"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8-sig")
