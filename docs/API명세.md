# 우울 위험 선별 API 명세

FastAPI 기반 추론 서버. STT로 변환된 어르신 텍스트를 받아 우울 위험도를 계산하고,
행복지수와 3단계 상태로 변환해 반환한다.

> ⚠️ 이 도구는 **진단이 아니라 선별·권유**용입니다. 상담 권유(🔴) 등급도 부드러운 권유 메시지로만 전달합니다.

<br>

## 서버 실행

```bash
pip install fastapi uvicorn torch transformers numpy pandas
python src/04_serve_api.py --models ./saved_models
# → 서버 시작: http://localhost:8000  (saved_models/ensemble.json 의 구성원 4개 로딩)
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
  "risk_score": 0.32,          // 위험도 0~1 (모델별 확률의 평균, 등급 기준값)
  "happiness_index": 68,       // 행복지수 0~100
  "grade": "안정",             // 안정 / 관찰 / 상담 권유
  "emoji": "🟢",
  "message": "좋은 하루를 보내고 계세요!",
  "raw_score": 0.32,           // 참고: 이전 명세 호환용 (보정을 없애 risk_score와 같은 값)
  "scores_by_model": {         // 참고: 모델별 개별 점수
    "KoBigBird_1234_f": 0.30,
    "KLUE-BERT_42": 0.33,
    "KLUE-BERT_2024": 0.31,
    "KoBigBird_2222_f": 0.34
  }
}
```

**등급 기준** (`saved_models/ensemble.json` 의 임계값. 서비스 입력 분량(답변 1개, 앞 800자)으로 자른 val에서 결정)

| risk_score | 등급 | 메시지 | 결정 규칙 |
|-----------|------|--------|-----------|
| < 0.388 | 🟢 안정 | 좋은 하루를 보내고 계세요! | |
| 0.388 \~ 0.611 | 🟡 관찰 | 조금 쉬어가도 좋아요. | `threshold_watch`: val 재현율 0.80 |
| ≥ 0.611 | 🔴 상담 권유 | 가까운 분과 이야기 나눠보는 건 어떨까요. | `threshold_refer`: val 특이도 0.80 |

🔴는 가족 알림으로 이어지는 등급이라(예시 코드의 `notifyFamily`) 비위험군 오경보를 20% 이내로 제한하는 기준을 쓴다. 🟡는 알림이 없는 등급이라 위험군을 놓치지 않는 쪽(재현율)으로 정했다. 800자 test 기준으로 🔴는 전체의 약 25%, 🟡 이상은 위험군의 약 86%를 포함한다.

행복지수 = round((1 − risk_score) × 100).

<br>

## `GET /health`

로딩 상태 확인.

```json
{ "status": "ok", "models_loaded": 4,
  "models": ["KoBigBird_1234_f", "KLUE-BERT_42", "KLUE-BERT_2024", "KoBigBird_2222_f"],
  "thresholds": {"refer": 0.611, "watch": 0.388},
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

> 호출 동안 "분석 중..." 로딩 표시를 권장.

<br>

## 전처리 · 임계값 노트

- 텍스트는 학습·평가와 같은 함수(`src/common.py`의 `chunk_text`, `aggregate`)로 나누고 집계한다. 모델별 입력 길이·청크 크기는 각 모델 폴더의 `inference.json`(학습 시 저장)을 따른다.
- 이전 버전의 수동 점수 보정(CAL_MIN 0.50 / CAL_MAX 0.78)은 없앴다. 등급 임계값은 07_input_length.py가 서비스 입력 분량(앞 800자)으로 자른 val에서 정해 `ensemble.json`에 저장한다. 같은 임계값을 test에 적용한 결과는 README의 결과 표 참고.
