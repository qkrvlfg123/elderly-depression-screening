# -*- coding: utf-8 -*-
"""
공통 모듈 — 학습(02)·교차검증(05)·평가(06)·서빙(04)이 같은 방식을 쓰도록 한 곳에 모은다.

  - chunk_text / aggregate : 텍스트 청킹과 화자 단위 집계 (학습·평가·서빙 공통)
  - MODEL_CONFIGS          : 모델 구성 (아키텍처·입력 길이·청크·배치·학습률·시드·손실)
  - load_speakers / make_split : 데이터 로드와 화자 단위 train/val/test 분할
  - train_model / predict_speakers : 학습(val로 epoch 선택)과 화자별 예측
  - screening_metrics / bootstrap_ci : 평가 지표
"""
import os
import re
import json

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OpenMP 충돌 방지 (Windows)

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PRED_DIR = os.path.join(RESULTS, "preds")
SPLIT_CSV = os.path.join(RESULTS, "split.csv")
EXPERIMENTS_CSV = os.path.join(RESULTS, "experiments.csv")
OLD_TEST_CSV = os.path.join(RESULTS, "pred_FINAL_KLUE-BERT_42.csv")  # 기존 평가 화자 1,164명

TARGET = "PHQ2"
CUT = 2                 # 학습 라벨: PHQ2 >= 2
REPORT_CUTS = (2, 3)    # 보고 기준: PHQ2 >= 2, PHQ2 >= 3
SPLIT_SEED = 42

# 키워드마다 우울점수가 다르게 기록된 화자 (라벨 불일치 → 제외)
EXCLUDE_SPEAKERS = {"20002", "20046", "20816", "20898"}

POS_WEIGHT = 5.0
FOCAL_GAMMA = 2.0
EPOCHS = 3

# 한국어 기준 토큰당 약 2자 → 청크 글자 수는 max_len*2 이내로 둔다 (잘림 방지)
ARCH = {
    "KoBigBird":    dict(hf="monologg/kobigbird-bert-base", max_len=1024, chunk_chars=2000,
                         max_chunks=5, batch=2, accum=8, lr=1e-5),
    "KLUE-BERT":    dict(hf="klue/bert-base", max_len=512, chunk_chars=800,
                         max_chunks=8, batch=16, accum=1, lr=2e-5),
    "KLUE-RoBERTa": dict(hf="klue/roberta-base", max_len=512, chunk_chars=800,
                         max_chunks=8, batch=16, accum=1, lr=2e-5),
}

MODEL_CONFIGS = {
    "KoBigBird_42":      dict(arch="KoBigBird", seed=42, focal=False),
    "KoBigBird_1234_f":  dict(arch="KoBigBird", seed=1234, focal=True),
    "KoBigBird_2222_f":  dict(arch="KoBigBird", seed=2222, focal=True),
    "KLUE-BERT_42":      dict(arch="KLUE-BERT", seed=42, focal=False),
    "KLUE-BERT_2024":    dict(arch="KLUE-BERT", seed=2024, focal=False),
    "KLUE-RoBERTa_888":  dict(arch="KLUE-RoBERTa", seed=888, focal=False),
}


# ── 청킹 · 집계 (학습·평가·서빙 공통) ─────────────────────────────────
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


def make_split(df, val_frac=0.125):
    """test = 기존 평가 화자 1,164명. 나머지를 라벨 계층화로 train/val 분할 (전체 기준 약 70/10/20).
    결과를 results/split.csv 에 고정 저장하고, 이후에는 그 파일을 재사용한다."""
    from sklearn.model_selection import train_test_split
    if os.path.exists(SPLIT_CSV):
        sp = pd.read_csv(SPLIT_CSV, dtype={"speaker_id": str})
    else:
        test_ids = set(pd.read_csv(OLD_TEST_CSV)["sid"].astype(str))
        rest = df[~df["speaker_id"].isin(test_ids)]
        tr, va = train_test_split(rest, test_size=val_frac, stratify=rest["label"],
                                  random_state=SPLIT_SEED)
        sp = pd.concat([
            pd.DataFrame({"speaker_id": tr["speaker_id"], "split": "train"}),
            pd.DataFrame({"speaker_id": va["speaker_id"], "split": "val"}),
            pd.DataFrame({"speaker_id": sorted(test_ids & set(df["speaker_id"])), "split": "test"}),
        ]).sort_values(["split", "speaker_id"])
        os.makedirs(RESULTS, exist_ok=True)
        sp.to_csv(SPLIT_CSV, index=False)
    m = dict(zip(sp["speaker_id"], sp["split"]))
    return {k: df[df["speaker_id"].map(m) == k].reset_index(drop=True)
            for k in ("train", "val", "test")}


# ── 학습 · 예측 ─────────────────────────────────────────────────────
def _torch():
    import torch
    return torch


def _make_rows(dframe, a):
    rows = []
    for sid, text, label in zip(dframe["speaker_id"], dframe["text"], dframe["label"]):
        for c in chunk_text(text, a["chunk_chars"], a["max_chunks"]):
            rows.append((sid, c, float(label)))
    return rows


def _loader(rows, tok, max_len, batch, shuffle):
    torch = _torch()
    from torch.utils.data import DataLoader

    def collate(b):
        e = tok([r[1] for r in b], truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"], torch.tensor([r[2] for r in b])
    return DataLoader(rows, batch_size=batch, shuffle=shuffle, collate_fn=collate)


def _chunk_probs(model, rows, tok, max_len, batch, device):
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for ids, mask, _ in _loader(rows, tok, max_len, batch * 2, False):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            out.append(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def predict_speakers(model, tok, dframe, arch, device):
    """화자별 확률 Series (index=speaker_id). 청킹·집계는 chunk_text/aggregate 사용."""
    a = ARCH[arch]
    rows = _make_rows(dframe, a)
    p = _chunk_probs(model, rows, tok, a["max_len"], a["batch"], device)
    sids = pd.Series([r[0] for r in rows])
    return pd.Series(p).groupby(sids).agg(aggregate).rename("prob")


def train_model(config_name, tr_df, va_df, device, save_dir=None, log=print):
    """구성 하나 학습. 매 epoch val AUROC를 재고, val 최고 epoch의 가중치를 복원해 돌려준다.
    반환: model, tok, info(dict: best_epoch, val_auroc_by_epoch)"""
    torch = _torch()
    import torch.nn as nn
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from sklearn.metrics import roc_auc_score

    cfg = MODEL_CONFIGS[config_name]
    a = ARCH[cfg["arch"]]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(a["hf"])
    model = AutoModelForSequenceClassification.from_pretrained(a["hf"], num_labels=1).to(device)
    pw = torch.tensor([POS_WEIGHT], device=device)

    def loss_fn(logits, y):
        ce = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw,
                                                            reduction="none")
        if cfg["focal"]:
            p = torch.sigmoid(logits)
            pt = torch.where(y == 1, p, 1 - p)
            ce = (1 - pt) ** FOCAL_GAMMA * ce
        return ce.mean()

    tr_rows = _make_rows(tr_df, a)
    dl = _loader(tr_rows, tok, a["max_len"], a["batch"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=0.01)
    steps = max(1, len(dl) // a["accum"]) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    y_val = va_df.set_index("speaker_id")["label"]
    log(f"[{config_name}] train {len(tr_df)}명/{len(tr_rows)}청크, val {len(va_df)}명")

    best_auc, best_state, best_ep, hist = -1.0, None, -1, []
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        for i, (ids, mask, y) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float().squeeze(-1), y.to(device)) / a["accum"]
            loss.backward()
            if (i + 1) % a["accum"] == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (i + 1) % 500 == 0:
                log(f"  ep{ep + 1} step {i + 1}/{len(dl)} loss {loss.item() * a['accum']:.4f}")
        pv = predict_speakers(model, tok, va_df, cfg["arch"], device)
        auc = roc_auc_score(y_val.loc[pv.index].values, pv.values)
        hist.append(round(float(auc), 4))
        log(f"  ep{ep + 1} val AUROC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_ep = auc, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "inference.json"), "w", encoding="utf-8") as f:
            json.dump({"arch": cfg["arch"], **{k: a[k] for k in ("max_len", "chunk_chars",
                                                                 "max_chunks")}}, f)
    return model, tok, {"best_epoch": best_ep, "val_auroc_by_epoch": hist}


# ── 평가 ───────────────────────────────────────────────────────────
def threshold_for_recall(y, p, target=0.80):
    """재현율 >= target 을 만족하는 가장 높은 임계값."""
    cands = np.sort(np.unique(p))[::-1]
    for t in cands:
        if ((p >= t) & (y == 1)).sum() / max(1, y.sum()) >= target:
            return float(t)
    return float(cands[-1])


def threshold_for_specificity(y, p, target=0.80):
    """특이도 >= target 을 만족하는 가장 낮은 임계값 (비위험군 점수의 target 분위 바로 위)."""
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])
    return float(neg[int(np.ceil(target * len(neg))) - 1]) + 1e-9


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
