# -*- coding: utf-8 -*-
"""
04. 우울 위험 선별 API 서버 (FastAPI) — 최종 4모델 앙상블 + 점수 보정

saved_models/ 폴더 안의 학습된 모델들을 자동 스캔·로딩해 앙상블한다. 아키텍처가 서로
달라도(KLUE-BERT / KLUE-RoBERTa / KoBigBird 등) config.json이 있으면 자동 인식한다.
STT로 변환된 어르신 텍스트를 받아 우울 위험도를 계산하고, '행복지수(0~100)'와
3단계 상태(안정/관찰/상담 권유)로 변환해 응답한다. (서비스의 /mood 엔드포인트)

핵심 설계
  - 서로 다른 아키텍처 자동 로딩 + 모델별 안전한 max_len 자동 결정
  - 점수 보정(calibration): 모델이 원점수를 0.50~0.78 좁은 구간에만 주는 성향을
    발견 → 이 구간을 0~1로 선형 확장해 등급 경계(0.4/0.5)가 제대로 작동하게 함
  - recall 우선 낮은 절단점 (위험군을 놓치지 않는 게 목적)

[실행]
    pip install fastapi uvicorn torch transformers numpy
    python 04_serve_api.py --models ./saved_models   # http://localhost:8000

[모델 폴더 구조] models 루트 아래에 모델별 폴더 (각 폴더에 config.json 필요)
    saved_models/
      ├─ model_KLUE-BERT_42/
      ├─ model_KLUE-RoBERTa_888/
      ├─ model_KoBigBird_1234_f/
      └─ model_KoBigBird_2222_f/

[엔드포인트]
    POST /predict   {"text": "..."}  → risk_score / happiness_index / grade / ...
    GET  /health                      → 로딩된 모델 목록 / device

주의: 이 도구는 '진단'이 아니라 '선별·권유'용입니다. 상담 권유(🔴) 등급도 부드러운
      권유 메시지로만 전달합니다.
"""
import os
import argparse
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 긴 입력 모델(BigBird 등)의 토큰 상한. 데모 속도를 위해 2048로 제한 (최대 4096까지 상향 가능).
LONG_MODEL_MAXLEN = 2048

# ── 점수 보정(calibration) ──
# 이 모델들은 원점수를 대략 0.50~0.78 좁은 구간에만 준다. 그대로 쓰면 밝은 문장도
# 0.5 밑으로 안 내려가 전부 🔴로 뜬다. 아래 두 값 사이를 0~1로 "펴서" 등급 경계
# (0.4/0.5)가 제대로 갈리게 만든다. (PPT의 0.4/0.5 기준은 보정된 점수에 그대로 적용)
# 등급이 한쪽으로 쏠리면 이 두 값만 조정 (밝은 게 🟢 안 뜨면 CAL_MIN↑, 어두운 게 🔴 안 뜨면 CAL_MAX↓)
CAL_MIN = 0.50   # 이 값 이하 → 위험도 0 (가장 안정)
CAL_MAX = 0.78   # 이 값 이상 → 위험도 1 (가장 위험)

app = FastAPI(title="우울 위험 선별 API (4-ensemble)")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

loaded = []


def derive_max_len(cfg):
    """모델 config를 보고 안전한 최대 토큰 길이를 결정."""
    mp = getattr(cfg, "max_position_embeddings", 512)
    if mp <= 514:            # BERT(512)/RoBERTa(514) 계열은 특수토큰 오프셋 감안해 512로 통일
        return 512
    return min(mp, LONG_MODEL_MAXLEN)  # BigBird 등 장문 모델


def calibrate(raw):
    """모델 원점수(좁은 구간)를 0~1로 선형 확장."""
    x = (raw - CAL_MIN) / (CAL_MAX - CAL_MIN)
    return max(0.0, min(1.0, x))


def to_grade(score):
    """위험도 점수(0~1) → 행복지수, 등급, 이모지, 메시지."""
    happiness = round((1 - score) * 100)
    if score < 0.4:
        return happiness, "안정", "🟢", "좋은 하루를 보내고 계세요!"
    elif score < 0.5:
        return happiness, "관찰", "🟡", "조금 쉬어가도 좋아요."
    else:
        return happiness, "상담 권유", "🔴", "가까운 분과 이야기 나눠보는 건 어떨까요."


def make_chunks(text, max_len):
    """모델 토큰 용량에 맞춰 텍스트를 글자 단위로 분할."""
    chunk_chars = max_len * 2           # 대략적인 글자 예산 (한국어 기준 여유있게)
    maxc = 5 if max_len <= 512 else 2   # 장문 모델은 청크 수를 줄여 속도 관리
    total = min(len(text), chunk_chars * maxc)
    chunks = [text[i:i + chunk_chars] for i in range(0, total, chunk_chars)][:maxc]
    return chunks if chunks else [text]


def predict_one(text, m):
    probs = []
    for c in make_chunks(text, m["len"]):
        enc = m["tok"](c, truncation=True, max_length=m["len"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logit = m["mdl"](**enc).logits
            if logit.shape[-1] == 1:                    # 단일 로짓 → sigmoid
                prob = torch.sigmoid(logit)[0][0].item()
            else:                                       # 2로짓 → softmax 양성 확률
                prob = torch.softmax(logit, dim=-1)[0][1].item()
        probs.append(prob)
    return max(probs)


class Req(BaseModel):
    text: str


@app.post("/predict")
def predict(req: Req):
    if not loaded:
        return {"error": "모델이 로딩되지 않았습니다 (models 폴더 확인)"}
    per_model = {m["name"]: round(predict_one(req.text, m), 4) for m in loaded}
    raw = float(np.mean(list(per_model.values())))   # 모델 평균 (보정 전)
    score = calibrate(raw)                            # 0~1로 펴준 최종 위험도
    happiness, grade, emoji, message = to_grade(score)
    return {
        "risk_score": round(score, 4),      # 위험도 0~1 (보정 후, 등급 기준)
        "happiness_index": happiness,       # 행복지수 0~100
        "grade": grade, "emoji": emoji, "message": message,
        "raw_score": round(raw, 4),         # 참고: 보정 전 원점수
        "scores_by_model": per_model,       # 참고: 모델별 개별 점수
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if loaded else "no_models",
        "models_loaded": len(loaded),
        "models": [m["name"] for m in loaded],
        "device": DEVICE,
    }


def load_models(models_dir):
    print(f"[{DEVICE}] 모델 로딩 중... (개수에 따라 20~40초)")
    if not os.path.isdir(models_dir):
        print(f"  [경고] 모델 폴더가 없습니다: {models_dir}")
        return
    # 루트 바로 아래에서 config.json 을 가진 폴더만 모델로 인식
    model_dirs = sorted(
        os.path.join(models_dir, d) for d in os.listdir(models_dir)
        if os.path.isfile(os.path.join(models_dir, d, "config.json"))
    )
    for path in model_dirs:
        name = os.path.basename(path)
        try:
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
            max_len = derive_max_len(mdl.config)
            loaded.append({"name": name, "tok": tok, "mdl": mdl, "len": max_len})
            print(f"  [OK] {name}  (arch={mdl.config.model_type}, max_len={max_len})")
        except Exception as e:
            print(f"  [실패] {name}: {e}")
    print(f"총 {len(loaded)}개 로딩 완료. 서버 준비됨.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="우울 위험 선별 API 서버 (4-ensemble)")
    ap.add_argument("--models", default="./saved_models", help="저장된 모델 폴더 경로")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load_models(args.models)
    import uvicorn
    print(f"서버 시작: http://localhost:{args.port}")
    print("  예측: POST /predict  {\"text\": \"...\"}")
    print("  상태: GET  /health")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
