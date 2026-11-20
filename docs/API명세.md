# 우울 위험 선별 API 명세

FastAPI 기반 추론 서버. STT로 변환된 어르신 텍스트를 받아 우울 위험도를 계산하고,
행복지수와 3단계 상태로 변환해 반환한다.

> ⚠️ 이 도구는 **진단이 아니라 선별·권유**용입니다. 상담 권유(🔴) 등급도 부드러운 권유 메시지로만 전달합니다.

<br>

## 서버 실행

```bash
pip install fastapi uvicorn torch transformers numpy
python src/04_serve_api.py --models ./saved_models
# → 서버 시작: http://localhost:8000  (모델 4개 로딩에 20~40초)
```

<br>

## `POST /predict`

어르신 답변 텍스트로 우울 위험도를 계산한다.

**요청**
```json
{ "text": "STT로 변환된 어르신 답변 텍스트" }
```

**응답**
```json
{
  "risk_score": 0.39,          // 위험도 0~1 (재조정 후, 등급 기준값) = (0.61 - 0.50) / (0.78 - 0.50)
  "happiness_index": 61,       // 행복지수 0~100 = 100 × (1 - 위험도)
  "grade": "안정",             // 안정 / 관찰 / 상담 권유
  "emoji": "🟢",
  "message": "좋은 하루를 보내고 계세요!",
  "raw_score": 0.61,           // 참고: 보정 전 원점수
  "scores_by_model": {         // 참고: 모델별 개별 점수
    "model_KLUE-BERT_42": 0.58,
    "model_KLUE-RoBERTa_888": 0.62,
    "model_KoBigBird_1234_f": 0.60,
    "model_KoBigBird_2222_f": 0.63
  }
}
```

**등급 기준** (보정된 `risk_score` 기준)

| risk_score | 등급 | 행복지수 | 메시지 |
|-----------|------|---------|--------|
| < 0.4 | 🟢 안정 | 60\~100 | 좋은 하루를 보내고 계세요! |
| 0.4 \~ 0.5 | 🟡 관찰 | 50\~60 | 조금 쉬어가도 좋아요. |
| ≥ 0.5 | 🔴 상담 권유 | 0\~50 | 가까운 분과 이야기 나눠보는 건 어떨까요. |

<br>

## `GET /health`

로딩 상태 확인.

```json
{ "status": "ok", "models_loaded": 4,
  "models": ["model_KLUE-BERT_42", "model_KLUE-RoBERTa_888",
             "model_KoBigBird_1234_f", "model_KoBigBird_2222_f"],
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

> 예측은 1건당 약 5\~10초(4모델). 호출 동안 "분석 중..." 로딩 표시를 권장.

<br>

## 점수 재조정 노트

이 모델들은 원점수를 대략 **0.50\~0.78 좁은 구간**에만 주는 성향이 있어, 보정 없이 쓰면 밝은 문장도 0.5 밑으로 내려가지 않아 전부 상담 권유로 뜬다. 서버 상단의 `CAL_MIN(0.50)` / `CAL_MAX(0.78)` 사이를 0\~1로 선형 확장해 등급 경계가 제대로 작동하게 맞춰두었다. 등급이 한쪽으로 쏠리면 이 두 값만 조정하면 되며, 등급 기준(0.4/0.5)은 그대로 유지된다.
