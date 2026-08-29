# -*- coding: utf-8 -*-
"""
03. 최종 배포 모델 학습 및 저장

앙상블 평가에서 성능이 검증된 조합을 다시 학습하되, best AUROC 시점의 가중치를
디스크에 저장한다. 저장된 모델은 API 서버(04_serve_api.py)가 로딩해 실시간 추론에 사용.

  - 기본: KoBigBird(seed=1234, focal) + KLUE-BERT(seed=42)  → 성능 강조 버전
  - --demo 옵션: KLUE-BERT 2개만 저장 → 빠른 시연 버전 (KoBigBird 제외, 예측 3~5초)

[실행]
    # 성능 버전 (KoBigBird 포함, 학습 오래 걸림)
    python 03_save_final_models.py --csv speakers_kssds.csv --out ./saved_models

    # 데모 버전 (가볍고 빠름)
    python 03_save_final_models.py --csv speakers_kssds.csv --out ./saved_models --demo

주의: 저장된 모델 가중치는 용량이 크므로 git에 커밋하지 않는다(.gitignore 처리).
"""
import os
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
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    sys.exit(f"[라이브러리 없음] {e}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지 (Windows 환경 이슈)

TARGET = "PHQ2"
CUT = 2
EPOCHS = 5
POS_WEIGHT = 5.0
FOCAL_GAMMA = 2.0
SEED_SPLIT = 42

BB_MODEL = "monologg/kobigbird-bert-base"
BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_ACCUM, BB_LR = 3072, 6000, 5, 1, 16, 1e-5
KL_MODEL = "klue/bert-base"
KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR = 512, 800, 8, 16, 2e-5


def sent_chunks(t, cc, mc):
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


def train_and_save(name, mn, ln, cc, mc, batch, lr, seed,
                   tr_df, va_df, truth, device, save_dir, accum=1, focal=False):
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

    best_auc, best_state = -1, None
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
        if au > best_auc:
            best_auc = au
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"    → best AUROC {best_auc:.3f}")

    # best 시점 가중치 복원 후 저장
    save_path = os.path.join(save_dir, f"model_{name}_{seed}{'_f' if focal else ''}")
    os.makedirs(save_path, exist_ok=True)
    model.load_state_dict(best_state)
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    print(f"    [모델 저장] {save_path}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="최종 배포 모델 학습 및 저장")
    ap.add_argument("--csv", required=True, help="전처리된 입력 CSV")
    ap.add_argument("--out", default="./saved_models", help="모델 저장 폴더")
    ap.add_argument("--demo", action="store_true",
                    help="데모 버전: KLUE-BERT 2개만 저장 (빠름)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.csv).dropna(subset=["text", TARGET]).reset_index(drop=True)
    df["score"] = df[TARGET].astype(float)
    df["speaker_id"] = df["speaker_id"].astype(str)
    tr_df, va_df = train_test_split(df, test_size=0.2, random_state=SEED_SPLIT)
    truth = (va_df.set_index("speaker_id")["score"] >= CUT).astype(int)
    print(f"장치: {device} | 화자 {len(df)}명 (학습 {len(tr_df)} / 평가 {len(va_df)})")

    ca = dict(tr_df=tr_df, va_df=va_df, truth=truth, device=device, save_dir=args.out)

    if args.demo:
        print("\n===== 데모 버전: KLUE-BERT 2개 저장 =====")
        train_and_save("KLUE-BERT", KL_MODEL, KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR, 42,   **ca)
        train_and_save("KLUE-BERT", KL_MODEL, KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR, 2024, **ca)
    else:
        print("\n===== 성능 버전: KoBigBird(focal) + KLUE-BERT 저장 =====")
        train_and_save("KoBigBird", BB_MODEL, BB_LEN, BB_CHUNK, BB_MAXC, BB_BATCH, BB_LR, 1234, accum=BB_ACCUM, focal=True, **ca)
        train_and_save("KLUE-BERT", KL_MODEL, KL_LEN, KL_CHUNK, KL_MAXC, KL_BATCH, KL_LR, 42, **ca)

    print(f"\n[완료] 모델 저장 → {args.out}")


if __name__ == "__main__":
    main()
