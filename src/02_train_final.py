# -*- coding: utf-8 -*-
"""
02. 발표 최종본 4모델 학습 (재현)

이전 최종 앙상블(AUROC 0.674 보고)을 같은 데이터·분할·설정으로 다시 학습한다.
학습 코드는 당시 스크립트(run.T.py)의 청킹·데이터셋·손실·학습 루프를 그대로 옮겼다 (fp32).

  - 데이터: speakers_clean_kssds.csv (KSSDS 문장 분리, 라벨 불일치 4명 제외, 5,819명)
  - 분할  : 화자 8:2 (train_test_split, random_state=42) → train 4,655 / test 1,164
  - 모델  : KoBigBird_1234_f, KoBigBird_2222_f, KLUE-RoBERTa_888, KLUE-BERT_42
  - 설정  : 5 epoch, KoBigBird 3072토큰·6000자×5청크·배치1×누적16·lr 1e-5,
            KLUE 계열 512토큰·800자×8청크·배치16·lr 2e-5, pos_weight 5, focal γ=2 (_f만)

당시에는 모델마다 test AUROC가 가장 높은 epoch을 골랐다. 여기서는 epoch마다 test 예측을
모두 저장하고 두 값을 함께 보고한다.
  - fixed_last : 미리 정한 규칙(마지막 epoch 5) — 대표값
  - best_test  : 당시 방식(test로 epoch 선택) — 재현 확인용, 낙관적

[실행] python src/02_train_final.py --csv speakers_clean_kssds.csv
완료된 구성은 건너뛴다. 가중치(마지막 epoch)는 saved_models/repro/<구성>/ 에 저장.
"""
import os
import re
import json
import time
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import common as C

TARGET = "PHQ2"; CUT = 2; EPOCHS = 5; POS_WEIGHT = 5.0; FOCAL_GAMMA = 2.0
BB = dict(mn="monologg/kobigbird-bert-base", ln=3072, cc=6000, mc=5, batch=1, lr=1e-5, accum=16)
KL = dict(mn="klue/bert-base", ln=512, cc=800, mc=8, batch=16, lr=2e-5, accum=1)
RB = dict(mn="klue/roberta-base", ln=512, cc=800, mc=8, batch=16, lr=2e-5, accum=1)
FINAL4 = [  # (이름, 설정, seed, focal) — 발표 PPT 최종 4개
    ("KoBigBird_1234_f", BB, 1234, True),
    ("KLUE-RoBERTa_888", RB, 888, False),
    ("KoBigBird_2222_f", BB, 2222, True),
    ("KLUE-BERT_42", KL, 42, False),
]
REPRO_DIR = os.path.join(C.PRED_DIR, "repro")


# ── 당시 코드 그대로 (run.T.py) ──────────────────────────────────────
def sent_chunks(t, cc, mc):
    ss = [p.strip() for p in re.split(r"(?<=[.!?])\s*", str(t)) if p.strip()]
    ch, cur = [], ""
    for s in ss:
        if len(cur) + len(s) + 1 > cc and cur: ch.append(cur); cur = s
        else: cur = (cur + " " + s).strip()
        if len(ch) >= mc: break
    if cur and len(ch) < mc: ch.append(cur)
    return [c for c in ch if len(c) > 20]


def mkrows(dframe, cc, mc):
    R = []
    for _, r in dframe.iterrows():
        for c in sent_chunks(r["text"], cc, mc): R.append((r["speaker_id"], c, r["score"]))
    return R


class DS(Dataset):
    def __init__(self, R, tok, ln): self.R, self.tok, self.ln = R, tok, ln
    def __len__(self): return len(self.R)
    def __getitem__(self, i):
        sid, txt, sc = self.R[i]
        e = self.tok(txt, truncation=True, max_length=self.ln, padding="max_length", return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0),
                "label": torch.tensor(1.0 if sc >= CUT else 0.0), "idx": i}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pw=1.0): super().__init__(); self.g = gamma; self.pw = pw
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                            pos_weight=torch.tensor([self.pw]).to(logits.device))
        pt = torch.where(target == 1, p, 1 - p)
        return ((1 - pt) ** self.g * ce).mean()


def train_epochs(name, cfg, seed, focal, tr_df, te_df, truth, device, save_dir):
    """당시 train_predict와 같은 학습. epoch마다 test 화자별 예측을 저장한다."""
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n=== [{name}] 학습 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["mn"])
    tr, te = mkrows(tr_df, cfg["cc"], cfg["mc"]), mkrows(te_df, cfg["cc"], cfg["mc"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["mn"], num_labels=1).to(device)
    lf = (FocalLoss(FOCAL_GAMMA, POS_WEIGHT) if focal
          else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    dl = DataLoader(DS(tr, tok, cfg["ln"]), batch_size=cfg["batch"], shuffle=True)
    accum = cfg["accum"]
    steps = max(1, len(dl) // accum) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    print(f"  train {len(tr_df)}명/{len(tr)}청크, test {len(te_df)}명/{len(te)}청크", flush=True)
    aucs = []
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train(); opt.zero_grad()
        for i, b in enumerate(dl):
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = lf(out.logits.squeeze(-1), b["label"].to(device)) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
            if (i + 1) % 2000 == 0:
                print(f"    ep{ep + 1} step {i + 1}/{len(dl)}", flush=True)
        model.eval(); probs = np.zeros(len(te))
        with torch.no_grad():
            for b in DataLoader(DS(te, tok, cfg["ln"]), batch_size=cfg["batch"]):
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
                p = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
                for j, idx in enumerate(b["idx"].numpy()): probs[idx] = p[j]
        vc = pd.DataFrame(te, columns=["sid", "t", "score"]); vc["prob"] = probs
        sm = vc.groupby("sid")["prob"].mean().rename("prob")
        sm.rename_axis("sid").reset_index().to_csv(os.path.join(REPRO_DIR, f"{name}_ep{ep + 1}.csv"), index=False)
        au = roc_auc_score(truth.loc[sm.index].values, sm.values)
        aucs.append(round(float(au), 4))
        print(f"  ep{ep + 1} test AUROC {au:.4f} ({(time.time() - t0) / 60:.0f}분)", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "inference.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": name.split("_")[0], "max_len": cfg["ln"], "chunk_chars": cfg["cc"],
                   "max_chunks": cfg["mc"]}, f)
    del model; torch.cuda.empty_cache()
    return aucs


# ── 요약 ────────────────────────────────────────────────────────────
def summarize(truth):
    def load(name, ep):
        d = pd.read_csv(os.path.join(REPRO_DIR, f"{name}_ep{ep}.csv"), dtype={"sid": str})
        return d.set_index("sid")["prob"]
    y = truth
    per = {n: [roc_auc_score(y.loc[load(n, e).index], load(n, e)) for e in range(1, EPOCHS + 1)]
           for n, *_ in FINAL4}
    out = {}
    for rule in ("fixed_last", "best_test"):
        eps = {n: (EPOCHS if rule == "fixed_last" else int(np.argmax(per[n])) + 1) for n in per}
        P = pd.concat([load(n, eps[n]) for n in per], axis=1)
        ens = P.mean(axis=1)
        yy = y.loc[ens.index].values
        auc = roc_auc_score(yy, ens.values)
        ci = C.bootstrap_ci(yy, ens.values)["auroc"]
        y3 = (truth_score.loc[ens.index] >= 3).astype(int).values
        m05 = C.screening_metrics(yy, ens.values, 0.5)
        out[rule] = {"epochs": eps, "auroc": auc, "auroc_ci": ci, "auroc_phq2ge3": roc_auc_score(y3, ens.values),
                     "at_0.5": m05, "members": {n: per[n][eps[n] - 1] for n in per}}
        print(f"\n[{rule}] epoch {eps}")
        print(f"  개별 AUROC {({n: round(per[n][eps[n] - 1], 4) for n in per})}")
        print(f"  4개 평균 AUROC {auc:.4f} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] | PHQ2>=3 {out[rule]['auroc_phq2ge3']:.4f}")
        print(f"  임계값 0.5: 재현율 {m05['recall']:.3f} 정밀도 {m05['precision']:.3f} "
              f"특이도 {m05['specificity']:.3f} 위험군 표시 {m05['flagged_rate']:.1%}")
        C.log_experiment({"run_id": f"repro/final4/{rule}", "kind": "repro_ensemble",
                          "config": "+".join(per), "threshold_rule": "0.5", "threshold": 0.5,
                          "test_auroc": auc, "test_auroc_ci_low": ci[0], "test_auroc_ci_high": ci[1],
                          "test_auroc_phq2ge3": out[rule]["auroc_phq2ge3"],
                          **{f"test_{k}": m05[k] for k in ("recall", "precision", "specificity", "flagged_rate")},
                          "best_epoch": str(eps), "n_train": 4655, "n_test": len(ens)})
    with open(os.path.join(C.RESULTS, "repro_final4.json"), "w", encoding="utf-8") as f:
        json.dump({"per_epoch_test_auroc": per, **out}, f, ensure_ascii=False, indent=2, default=float)


def main():
    global truth_score
    ap = argparse.ArgumentParser(description="최종본 4모델 앙상블 재현")
    ap.add_argument("--csv", default=os.path.join(C.ROOT, "speakers_clean_kssds.csv"))
    ap.add_argument("--models-dir", default=os.path.join(C.ROOT, "saved_models", "repro"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv).dropna(subset=["text", TARGET]).reset_index(drop=True)
    df["score"] = df[TARGET].astype(float); df["speaker_id"] = df["speaker_id"].astype(str)
    tr_df, te_df = train_test_split(df, test_size=0.2, random_state=42)
    truth = (te_df.set_index("speaker_id")["score"] >= CUT).astype(int)
    truth_score = te_df.set_index("speaker_id")["score"]
    print(f"장치 {device} | 화자 {len(df)}명 (train {len(tr_df)} / test {len(te_df)})", flush=True)
    os.makedirs(REPRO_DIR, exist_ok=True)

    for name, cfg, seed, focal in FINAL4:
        if os.path.exists(os.path.join(REPRO_DIR, f"{name}_ep{EPOCHS}.csv")):
            print(f"[{name}] 완료됨 → 건너뜀", flush=True)
            continue
        t0 = time.time()
        aucs = train_epochs(name, cfg, seed, focal, tr_df, te_df, truth, device,
                            os.path.join(args.models_dir, name))
        C.log_experiment({"run_id": f"repro/{name}", "kind": "repro_single", "config": name,
                          "arch": name.split("_")[0], "seed": seed, "loss": "focal" if focal else "bce",
                          "max_len": cfg["ln"], "chunk_chars": cfg["cc"], "max_chunks": cfg["mc"], "lr": cfg["lr"],
                          "epochs": EPOCHS, "test_auroc_by_epoch": str(aucs), "test_auroc": aucs[-1],
                          "n_train": len(tr_df), "n_test": len(te_df),
                          "minutes": round((time.time() - t0) / 60, 1)})
    summarize(truth)


if __name__ == "__main__":
    main()
