# -*- coding: utf-8 -*-
"""
02. 모델 구성별 학습 — 화자 단위 train/val/test 분할

  - 분할: test = 기존 평가 화자 1,164명(results/pred_FINAL_*.csv), 나머지를 라벨 계층화로
    train/val 분할 (전체 기준 약 70/10/20, 시드 42). 분할은 results/split.csv 에 고정.
  - 라벨: PHQ2 >= 2
  - 구성마다 매 epoch val AUROC로 최적 epoch 선택 → 그 가중치를 saved_models/<구성>/ 에 저장
  - val·test 화자별 예측을 results/preds/<구성>_{val,test}.csv 로 저장
  - test 지표는 여기서 계산하지 않는다. 모델·앙상블·임계값 선택(val)과 test 평가는 06에서 한 번에.

[실행]
    python src/02_train_ensemble.py --csv speakers.csv                 # 전체 구성
    python src/02_train_ensemble.py --csv speakers.csv --configs KLUE-BERT_42
이미 예측 파일이 있는 구성은 건너뛴다 (--force 로 재학습).
"""
import os
import time
import argparse

import common as C


def main():
    ap = argparse.ArgumentParser(description="구성별 학습 (val로 epoch 선택)")
    ap.add_argument("--csv", default=os.path.join(C.ROOT, "speakers.csv"))
    ap.add_argument("--configs", nargs="*", default=list(C.MODEL_CONFIGS))
    ap.add_argument("--models-dir", default=os.path.join(C.ROOT, "saved_models"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = C.load_speakers(args.csv)
    parts = C.make_split(df)
    print(f"장치 {device} | 화자 {len(df)}명 → " +
          ", ".join(f"{k} {len(v)}명(위험군 {v['label'].mean():.1%})" for k, v in parts.items()))
    os.makedirs(C.PRED_DIR, exist_ok=True)

    for name in args.configs:
        val_path = os.path.join(C.PRED_DIR, f"{name}_val.csv")
        test_path = os.path.join(C.PRED_DIR, f"{name}_test.csv")
        if os.path.exists(test_path) and not args.force:
            print(f"[{name}] 예측 파일 있음 → 건너뜀")
            continue
        t0 = time.time()
        model, tok, info = C.train_model(name, parts["train"], parts["val"], device,
                                         save_dir=os.path.join(args.models_dir, name))
        arch = C.MODEL_CONFIGS[name]["arch"]
        pv = C.predict_speakers(model, tok, parts["val"], arch, device)
        pt = C.predict_speakers(model, tok, parts["test"], arch, device)
        pv.rename_axis("sid").reset_index().to_csv(val_path, index=False)
        pt.rename_axis("sid").reset_index().to_csv(test_path, index=False)
        cfg = C.MODEL_CONFIGS[name]
        C.log_experiment({
            "run_id": f"holdout/{name}", "kind": "single", "config": name,
            "arch": arch, "seed": cfg["seed"], "loss": "focal" if cfg["focal"] else "bce",
            **{k: C.ARCH[arch][k] for k in ("max_len", "chunk_chars", "max_chunks", "lr")},
            "epochs": C.EPOCHS, "best_epoch": info["best_epoch"],
            "val_auroc_by_epoch": str(info["val_auroc_by_epoch"]),
            "val_auroc": max(info["val_auroc_by_epoch"]),
            "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(parts["test"]),
            "minutes": round((time.time() - t0) / 60, 1),
        })
        print(f"[{name}] 완료 {(time.time() - t0) / 60:.1f}분, best epoch {info['best_epoch']}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
