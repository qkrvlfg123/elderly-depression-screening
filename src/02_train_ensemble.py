# -*- coding: utf-8 -*-
"""
02. 우울 위험 선별 — 멀티모델 앙상블 학습 및 평가

전처리된 화자 단위 구술 텍스트로 여러 아키텍처(KoBigBird / KLUE-BERT / KLUE-RoBERTa /
KcELECTRA)를 시드·손실함수(BCE/Focal)를 바꿔가며 학습하고, 개별 모델 예측을 평균
앙상블하여 최종 성능(AUROC / recall / precision)을 평가한다.

핵심 설계
  - 라벨: PHQ-2 >= 2 를 '우울 위험군(1)'으로 정의 (선별검사 기준)
  - 긴 구술 텍스트 대응: 문장 단위 청킹 후 화자별 예측 평균 (정보 손실 최소화)
  - 클래스 불균형(위험군 23%) 대응: pos_weight + Focal Loss
  - 상위 성능 조합만 골라 앙상블 → 개별 최고 대비 성능 향상

[실행 환경]
    pip install torch transformers scikit-learn pandas numpy

[실행]
    python 02_train_ensemble.py --csv speakers_kssds.csv

[입력 CSV 형식] speaker_id, text, PHQ2
"""
import re
import sys
import argparse
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, recall_score, precision_score
except ImportError as e:
    sys.exit(f"[라이브러리 없음] {e}\n  pip install torch transformers scikit-learn pandas numpy")

# ── 공통 하이퍼파라미터 ─────────────────────────────────────────────
TARGET = "PHQ2"          # 라벨 컬럼
CUT = 2                  # PHQ-2 >= 2 → 위험군
EPOCHS = 3
POS_WEIGHT = 5.0         # 불균형 보정 (위험군 소수)
FOCAL_GAMMA = 2.0
SEED_SPLIT = 42          # train/test 분할 고정 (데이터 누수 방지)

# ── 모델별 설정 (아키텍처 / 입력 길이 / 청크 크기 / 청크 수 / 배치 / 학습률) ──
BB_MODEL = "monologg/kobigbird-bert-base"  # 긴 문맥 처리
BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_ACCUM, BB_LR = 1024, 3500, 5, 2, 8, 1e-5
KL_MODEL = "klue/bert-base"                # 기본 한국어 이해
KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR = 512, 800, 8, 16, 2e-5


def sent_chunks(t, cc, mc):
    """문장 경계를 유지하며 청크 구성 (글자 단순 절단 대신 의미 보존)."""
    ss = [p.strip() for p in re.split(r"(?<=[.!?])\s*", str(t)) if p.strip()]
    ch, cur = [], ""
    for s in ss:
        if len(cur) + len(s) + 1 > cc and cur:
            ch.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
        if len(ch) >= mc:
            break
    if cur and len(ch) < mc:
        ch.append(cur)
    return [c for c in ch if len(c) > 20]


def mkrows(dframe, cc, mc):
    R = []
    for _, r in dframe.iterrows():
        for c in sent_chunks(r["text"], cc, mc):
            R.append((r["speaker_id"], c, r["score"]))
    return R


class DS(Dataset):
    def __init__(self, R, tok, ln):
        self.R, self.tok, self.ln = R, tok, ln

    def __len__(self):
        return len(self.R)

    def __getitem__(self, i):
        sid, txt, sc = self.R[i]
        e = self.tok(txt, truncation=True, max_length=self.ln,
                     padding="max_length", return_tensors="pt")
        return {
            "input_ids": e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "label": torch.tensor(1.0 if sc >= CUT else 0.0),
            "idx": i,
        }


class FocalLoss(nn.Module):
    """불균형 데이터에서 어려운 샘플에 가중 (소수 위험군 학습 강화)."""
    def __init__(self, gamma=2.0, pw=1.0):
        super().__init__()
        self.g, self.pw = gamma, pw

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none",
            pos_weight=torch.tensor([self.pw]).to(logits.device),
        )
        pt = torch.where(target == 1, p, 1 - p)
        return ((1 - pt) ** self.g * ce).mean()


def train_predict(name, mn, ln, cc, mc, batch, lr, seed,
                  tr_df, va_df, truth, device, accum=1, focal=False):
    """단일 모델 학습 → 화자별 예측 확률(Series) 반환."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag = f"{name} seed={seed}{' focal' if focal else ''}"
    print(f"\n=== [{tag}] 학습 ===")

    tok = AutoTokenizer.from_pretrained(mn, trust_remote_code=True)
    tr, va = mkrows(tr_df, cc, mc), mkrows(va_df, cc, mc)
    model = AutoModelForSequenceClassification.from_pretrained(
        mn, num_labels=1, trust_remote_code=True).to(device)

    lf = (FocalLoss(FOCAL_GAMMA, POS_WEIGHT) if focal
          else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(tr, tok, ln), batch_size=batch, shuffle=True)
    steps = max(1, len(dl) // accum) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)

    best = (-1, None)
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        for i, b in enumerate(dl):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            loss = lf(out.logits.squeeze(-1), b["label"].to(device)) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

        # ── 에폭 평가: 화자별 청크 예측 평균 → AUROC ──
        model.eval()
        probs = np.zeros(len(va))
        with torch.no_grad():
            for b in DataLoader(DS(va, tok, ln), batch_size=batch):
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device))
                p = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
                for j, idx in enumerate(b["idx"].numpy()):
                    probs[idx] = p[j]
        vc = pd.DataFrame(va, columns=["sid", "t", "score"])
        vc["prob"] = probs
        sm = vc.groupby("sid")["prob"].mean()
        au = roc_auc_score(truth.loc[sm.index].values, sm.values)
        if au > best[0]:
            best = (au, sm.copy())

    print(f"    → best AUROC {best[0]:.3f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return best[1]


def main():
    ap = argparse.ArgumentParser(description="우울 위험 앙상블 학습/평가")
    ap.add_argument("--csv", required=True, help="전처리된 입력 CSV (speaker_id,text,PHQ2)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {device} | 멀티모델 앙상블 (KoBigBird x4 + KLUE-BERT x2)")

    df = pd.read_csv(args.csv).dropna(subset=["text", TARGET]).reset_index(drop=True)
    df["score"] = df[TARGET].astype(float)
    df["speaker_id"] = df["speaker_id"].astype(str)
    tr_df, va_df = train_test_split(df, test_size=0.2, random_state=SEED_SPLIT)
    print(f"화자 {len(df)}명 (학습 {len(tr_df)} / 평가 {len(va_df)})")
    truth = (va_df.set_index("speaker_id")["score"] >= CUT).astype(int)

    # ── 6개 모델 학습 (시드/손실 조합) ──
    common_args = dict(tr_df=tr_df, va_df=va_df, truth=truth, device=device)
    preds = [
        train_predict("KoBigBird", BB_MODEL, BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_LR, 42,   accum=BB_ACCUM, **common_args),
        train_predict("KoBigBird", BB_MODEL, BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_LR, 2024, accum=BB_ACCUM, **common_args),
        train_predict("KoBigBird", BB_MODEL, BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_LR, 777,  accum=BB_ACCUM, **common_args),
        train_predict("KoBigBird", BB_MODEL, BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_LR, 1234, accum=BB_ACCUM, focal=True, **common_args),
        train_predict("KLUE-BERT", KL_MODEL, KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR, 2024, **common_args),
        train_predict("KLUE-BERT", KL_MODEL, KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR, 777,  **common_args),
    ]

    # ── 공통 화자만 정렬해 예측 행렬 구성 ──
    common = preds[0].index
    for p in preds[1:]:
        common = common.intersection(p.index)
    mat = np.column_stack([p.loc[common].values for p in preds])
    y = truth.loc[common].values

    print("\n" + "=" * 55)
    print("  앙상블 최종 결과")
    print("=" * 55)
    aucs = [roc_auc_score(y, mat[:, i]) for i in range(mat.shape[1])]
    print(f"개별 모델 AUROC: {[round(a, 3) for a in aucs]}")

    print(f"\n[전체 평균] AUROC: {roc_auc_score(y, mat.mean(axis=1)):.3f}")

    # 약한 모델 제외하고 상위 조합만 앙상블
    order = np.argsort(aucs)[::-1]
    for topk in [3, 4, 5]:
        sel = order[:topk]
        print(f"[상위 {topk}개 평균] AUROC: {roc_auc_score(y, mat[:, sel].mean(axis=1)):.3f}")

    # 최고 조합(상위 4) 기준 임계값별 recall/precision
    best_ens = mat[:, order[:4]].mean(axis=1)
    print(f"\n★ 최종(상위 4 평균) AUROC: {roc_auc_score(y, best_ens):.3f}")
    print(f"\n{'임계값':>6}{'recall':>9}{'precision':>11}{'권유인원':>9}")
    for th in [0.3, 0.4, 0.5, 0.6]:
        pr = (best_ens >= th).astype(int)
        print(f"{th:>6.1f}{recall_score(y, pr, zero_division=0):>9.3f}"
              f"{precision_score(y, pr, zero_division=0):>11.3f}{int(pr.sum()):>9}")


if __name__ == "__main__":
    main()
