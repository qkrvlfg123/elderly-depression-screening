# 우울 위험 선별 API 명세

FastAPI 기반 추론 서버입니다. STT로 변환된 어르신 텍스트를 받아 우울 위험도를 계산하고,
행복지수와 3단계 상태로 변환해 반환합니다. 모델은 발표 최종본 4모델 앙상블입니다.

> ⚠️ 이 도구는 **진단이 아니라 선별·권유**용입니다. 상담 권유(🔴) 등급도 부드러운 권유 메시지로만 전달합니다.

<br>

## 서버 실행

```bash
pip install fastapi uvicorn torch transformers numpy pandas
# 최종본 가중치 4개를 saved_models/final/<모델명>/ 에 두고, 03_evaluate_final.py로 ensemble.json·inference.json 작성 후
python src/04_serve_api.py
# → 서버 시작: http://localhost:8000  (saved_models/final/ensemble.json 의 구성원 4개 로딩)
```

<br>

## `POST /predict`

어르신 답변 텍스트로 우울 위험도를 계산합니다.

**요청**
```json
{ "text": "STT로 변환된 어르신 답변 텍스트" }
```

**응답**
```json
{
  "risk_score": 0.32,          // 위험도 0~1 (보정 후, 등급 기준값)
  "happiness_index": 68,       // 행복지수 0~100
  "grade": "안정",             // 안정 / 관찰 / 상담 권유
  "emoji": "🟢",
  "message": "좋은 하루를 보내고 계세요!",
  "raw_score": 0.5896,         // 참고: 보정 전 모델 평균
  "scores_by_model": {         // 참고: 모델별 개별 점수
    "KoBigBird_1234_f": 0.58,
    "KLUE-RoBERTa_888": 0.60,
    "KoBigBird_2222_f": 0.57,
    "KLUE-BERT_42": 0.61
  }
}
```

**점수 보정**: risk_score = clip((raw_score − 0.50) / (0.78 − 0.50), 0, 1). 모델 원점수가 대부분 0.4–0.7의 좁은 구간에 있어서, 발표 최종본에서 이 구간을 0~1로 펴서 등급을 나눴습니다.

**등급 기준** (발표 최종본과 같음, `saved_models/final/ensemble.json`)

| risk_score | 원점수 환산 | 등급 | 메시지 |
|-----------|------|------|--------|
| < 0.4 | < 0.612 | 🟢 안정 | 좋은 하루를 보내고 계세요! |
| 0.4 \~ 0.5 | 0.612 \~ 0.64 | 🟡 관찰 | 조금 쉬어가도 좋아요. |
| ≥ 0.5 | ≥ 0.64 | 🔴 상담 권유 | 가까운 분과 이야기 나눠보는 건 어떨까요. |

test 1,164명, 답변 1개 분량(앞 800자) 입력 기준으로 🔴는 전체의 23.9%를 표시하고 위험군의 34%를 포함합니다(정밀도 0.32). 🟡 이상은 34.4%, 위험군의 45%를 포함합니다. 자세한 수치는 README에 있습니다.

행복지수 = round((1 − risk_score) × 100).

<br>

## `GET /health`

로딩 상태를 확인합니다.

```json
{ "status": "ok", "models_loaded": 4,
  "models": ["KoBigBird_1234_f", "KLUE-RoBERTa_888", "KoBigBird_2222_f", "KLUE-BERT_42"],
  "thresholds": {"refer": 0.5, "watch": 0.4},
  "calibration": {"min": 0.5, "max": 0.78},
  "device": "cuda" }
```

<br>

## 호출 예시 (Node.js / 프론트엔드)

```javascript
const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: 어르신답변텍스트 }),
});
const result = await res.json();
// result.happiness_index, result.grade, result.message 사용
```

> 호출하는 동안 "분석 중..." 로딩 표시를 권장합니다.

<br>

## 전처리 노트

- 텍스트는 학습·평가와 같은 함수(`src/common.py`의 `chunk_text`, `aggregate`)로 나누고 청크 확률을 평균합니다. 모델별 입력 길이·청크 크기는 각 모델 폴더의 `inference.json`을 따릅니다(KoBigBird 3,072토큰 · 6,000자 × 5청크, KLUE 계열 512토큰 · 800자 × 8청크).
- 발표 당시 서버는 4,096자 × 2청크 중 최댓값을 썼습니다. 학습과 맞추기 위해 위 방식으로 바꾸고, 학습처럼 청크를 max_length까지 패딩합니다(KoBigBird는 입력 길이에 따라 어텐션 방식이 달라져, 패딩하지 않으면 학습 때와 다른 점수가 나옵니다).
