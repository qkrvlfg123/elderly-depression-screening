# 우울 위험 선별 API 명세

FastAPI 기반 추론 서버. STT로 변환된 고령자 발화 텍스트를 받아 우울 위험도를 계산하고, 행복지수와 3단계 등급으로 바꿔 돌려줌. 모델은 4모델 앙상블(KoBigBird 2개, KLUE-RoBERTa, KLUE-BERT)임.

진단 도구가 아니라 선별과 상담 권유용임. 상담 권유 등급도 권유 문구로만 전달함.

## 서버 실행

```bash
pip install fastapi uvicorn torch transformers numpy pandas
# 배포 가중치 4개를 saved_models/final/<모델명>/ 에 두고, 03_evaluate_final.py 로 ensemble.json, inference.json 을 만든 뒤 실행
python src/04_serve_api.py
# 서버 시작: http://localhost:8000  (saved_models/final/ensemble.json 의 구성원 4개 로딩)
```

- 옵션: `--models` 모델 폴더(기본 `saved_models/final`), `--port` 포트(기본 8000).
- `ensemble.json` 이 없거나 `threshold_source` 항목이 없으면 모델을 로딩하지 않음. 이때 `/health` 는 `"status": "no_models"` 를 돌려줌.

## `POST /predict`

발화 텍스트로 우울 위험도를 계산함.

요청

```json
{ "text": "STT로 변환된 어르신 답변 텍스트" }
```

응답

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

- 위 값은 형식을 보여 주는 예시임.
- 모델이 로딩되지 않은 상태에서는 `{"error": "..."}` 만 돌려줌.

### 점수 보정

`risk_score = clip((raw_score - 0.50) / (0.78 - 0.50), 0, 1)`

- 모델 평균 점수(원점수)가 좁은 구간에 몰려 있어 0.50~0.78 구간을 0~1로 펴서 등급을 나눔.
- 평가 화자 1,164명의 원점수 5~95 백분위는 전체 텍스트 입력에서 0.41~0.68, 앞 800자 입력에서 0.39~0.71임(`results/final_serving.json`).
- `ensemble.json` 에 `calibration` 이 없으면 보정하지 않고 `risk_score` 가 `raw_score` 와 같음.

### 등급 기준

값은 `saved_models/final/ensemble.json` 에 있음. 보정 구간과 등급 기준은 시연 문장으로 정한 초기값이며 데이터로 조정한 값이 아님.

| risk_score | 원점수 환산 | 등급 | 메시지 |
|---|---|---|---|
| 0.4 미만 | 0.612 미만 | 안정 | 좋은 하루를 보내고 계세요! |
| 0.4 이상 0.5 미만 | 0.612 이상 0.64 미만 | 관찰 | 조금 쉬어가도 좋아요. |
| 0.5 이상 | 0.64 이상 | 상담 권유 | 가까운 분과 이야기 나눠보는 건 어떨까요. |

- `emoji` 항목은 등급별 색 원 문자임(안정 초록, 관찰 노랑, 상담 권유 빨강).
- 평가 화자 1,164명, 답변 1개 분량(앞 800자) 입력 기준으로 상담 권유는 전체의 23.9%에 표시되고 위험군의 34%를 포함함(정밀도 0.32). 관찰 이상은 34.4%에 표시되고 위험군의 45%를 포함함. 상세 수치는 README와 `results/final_serving.json`에 있음.

### 행복지수

`happiness_index = round((1 - risk_score) * 100)`

## `GET /health`

로딩 상태 확인용.

```json
{ "status": "ok", "models_loaded": 4,
  "models": ["KoBigBird_1234_f", "KLUE-RoBERTa_888", "KoBigBird_2222_f", "KLUE-BERT_42"],
  "thresholds": {"refer": 0.5, "watch": 0.4},
  "calibration": {"min": 0.5, "max": 0.78},
  "device": "cuda" }
```

## 호출 예시 (Node.js, 프론트엔드)

```javascript
const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: 어르신답변텍스트 }),
});
const result = await res.json();
// result.happiness_index, result.grade, result.message 사용
```

- 4개 모델을 차례로 추론하므로 응답까지 시간이 걸림. 호출하는 동안 로딩 표시를 두는 것을 권장함.
- CORS는 모든 출처를 허용하도록 설정되어 있음.

## 전처리

- 텍스트는 평가와 같은 함수(`src/common.py`의 `chunk_text`, `aggregate`)로 나누고 청크 확률을 평균함.
- 모델별 입력 길이와 청크 크기는 각 모델 폴더의 `inference.json`을 따름. KoBigBird는 3,072토큰, 6,000자 × 최대 5청크. KLUE 계열은 512토큰, 800자 × 최대 8청크.
- 청크는 학습 때처럼 max_length까지 패딩함. KoBigBird는 입력 길이에 따라 어텐션 방식이 달라져, 패딩하지 않으면 학습 때와 다른 점수가 나옴.
- 발표 당시 서버는 4,096자 × 2청크 중 최댓값을 썼음(기록). 학습과 맞추려고 위 방식으로 바꿈.
