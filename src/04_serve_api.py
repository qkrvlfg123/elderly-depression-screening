# -*- coding: utf-8 -*-
"""
04. 우울 위험 선별 API 서버 (FastAPI)

기본은 발표 최종본 4모델(saved_models/final/, 09가 ensemble.json·inference.json 작성)을 로딩한다.
02→06→07 파이프라인 모델은 --models ./saved_models 로 띄운다.
청킹·집계는 학습·평가와 같은 함수(common.chunk_text / common.aggregate)를 쓰고,
모델별 입력 길이·청크 설정은 각 모델 폴더의 inference.json을 그대로 따른다.

점수: 모델 평균(raw) → ensemble.json 에 calibration 이 있으면 [min, max] 를 0~1로 선형 확장
      (최종본: 0.50 / 0.78) → risk_score
등급: 상담 권유 🔴 risk_score >= threshold_refer / 관찰 🟡 >= threshold_watch / 안정 🟢 그 외
      (최종본: 0.5 / 0.4, 시연 문장으로 정한 값 / 02 파이프라인: 07이 val에서 정한 값)

[실행]
    pip install fastapi uvicorn torch transformers numpy pandas
    python src/04_serve_api.py                            # 최종본, http://localhost:8000

[모델 폴더 구조]
    saved_models/
      ├─ ensemble.json            # {"members": [...], "threshold_refer": ..., "threshold_watch": ..., "calibration": {...}}
      ├─ KoBigBird_1234_f/        # config.json, 가중치, 토크나이저, inference.json
      └─ ...

[엔드포인트]
    POST /predict   {"text": "..."}  → risk_score / happiness_index / grade / ...
    GET  /health                      → 로딩된 모델 목록 / 임계값 / device

주의: 이 도구는 '진단'이 아니라 '선별·권유'용입니다. 상담 권유(🔴) 등급도 부드러운
      권유 메시지로만 전달합니다.
"""
import os
import json
import argparse

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import common as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="우울 위험 선별 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

loaded = []
THRESHOLDS = {}
CALIBRATION = {}


def calibrate(raw):
    """calibration 이 있으면 [min, max] 를 0~1로 선형 확장 (최종본 server.py 와 같은 식)."""
    if not CALIBRATION:
        return raw
    x = (raw - CALIBRATION["min"]) / (CALIBRATION["max"] - CALIBRATION["min"])
    return max(0.0, min(1.0, x))


def to_grade(score):
    """위험도(0~1) → 행복지수, 등급, 이모지, 메시지."""
    happiness = round((1 - score) * 100)
    if score >= THRESHOLDS["refer"]:
        return happiness, "상담 권유", "🔴", "가까운 분과 이야기 나눠보는 건 어떨까요."
    if score >= THRESHOLDS["watch"]:
        return happiness, "관찰", "🟡", "조금 쉬어가도 좋아요."
    return happiness, "안정", "🟢", "좋은 하루를 보내고 계세요!"


def predict_one(text, m):
    """학습·평가와 같은 청킹(chunk_text)과 집계(aggregate)."""
    probs = []
    for c in C.chunk_text(text, m["chunk_chars"], m["max_chunks"]):
        # 학습·평가와 같이 max_len까지 패딩 (KoBigBird는 길이에 따라 어텐션 방식이 바뀜)
        enc = m["tok"](c, truncation=True, max_length=m["max_len"], padding="max_length",
                       return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs.append(torch.sigmoid(m["mdl"](**enc).logits.float())[0][0].item())
    return C.aggregate(probs)


class Req(BaseModel):
    text: str


@app.post("/predict")
def predict(req: Req):
    if not loaded:
        return {"error": "모델이 로딩되지 않았습니다 (models 폴더 확인)"}
    per_model = {m["name"]: round(predict_one(req.text, m), 4) for m in loaded}
    raw = float(np.mean(list(per_model.values())))     # 앙상블 = 모델 평균
    score = calibrate(raw)
    happiness, grade, emoji, message = to_grade(score)
    return {
        "risk_score": round(score, 4),
        "happiness_index": happiness,
        "grade": grade, "emoji": emoji, "message": message,
        "raw_score": round(raw, 4),         # 보정 전 모델 평균 (calibration 이 없으면 risk_score 와 같음)
        "scores_by_model": per_model,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if loaded else "no_models",
        "models_loaded": len(loaded),
        "models": [m["name"] for m in loaded],
        "thresholds": THRESHOLDS,
        "calibration": CALIBRATION or None,
        "device": DEVICE,
    }


def load_models(models_dir):
    ens_path = os.path.join(models_dir, "ensemble.json")
    if not os.path.isfile(ens_path):
        print(f"  [경고] {ens_path} 가 없습니다. 최종본은 09, 02 파이프라인은 06 → 07 을 먼저 실행하세요.")
        return
    with open(ens_path, encoding="utf-8") as f:
        ens = json.load(f)
    if "threshold_source" not in ens:
        print("  [중단] ensemble.json 에 threshold_source 가 없습니다 (최종본: 09 실행 / 02 파이프라인: 06 다음 07 실행).")
        return
    if not 0.0 <= ens["threshold_watch"] <= ens["threshold_refer"] <= 1.0:
        print(f"  [중단] 임계값 순서가 잘못됐습니다: watch {ens['threshold_watch']} / refer {ens['threshold_refer']}")
        return
    THRESHOLDS.update(refer=ens["threshold_refer"], watch=ens["threshold_watch"])
    CALIBRATION.update(ens.get("calibration") or {})
    print(f"[{DEVICE}] 앙상블 {ens['members']} 로딩 중...")
    for name in ens["members"]:
        path = os.path.join(models_dir, name)
        with open(os.path.join(path, "inference.json"), encoding="utf-8") as f:
            inf = json.load(f)
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
        loaded.append({"name": name, "tok": tok, "mdl": mdl, **inf})
        print(f"  [OK] {name}  (max_len={inf['max_len']}, chunk={inf['chunk_chars']}×{inf['max_chunks']})")
    print(f"총 {len(loaded)}개 로딩 완료. 임계값 {THRESHOLDS} 보정 {CALIBRATION or '없음'}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="우울 위험 선별 API 서버")
    ap.add_argument("--models", default=os.path.join(C.ROOT, "saved_models", "final"), help="저장된 모델 폴더 경로")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load_models(args.models)
    import uvicorn
    print(f"서버 시작: http://localhost:{args.port}")
    print("  예측: POST /predict  {\"text\": \"...\"}")
    print("  상태: GET  /health")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
