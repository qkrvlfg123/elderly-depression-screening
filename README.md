# 사랑방 자서전 — 우울 위험 선별 AI

> 매일 질문 하나에 말로 답하는 녹화 한 번으로 — 자서전이 쌓이고 · 건강 신호가 감지되고 · 또래와 연결됩니다.

**사랑방 자서전**은 고령층이 오늘의 질문에 구술로 답하면 그 영상이 자동으로 숏폼·자서전으로 제작되고, 발화 텍스트에서 **우울 신호를 비의료적으로 관찰**해 가족에게 전달하는 시니어 헬스케어 서비스입니다.

이 레포지토리는 그중 **Data Analyst(DA) 파트 — 우울 위험 선별 AI**의 코드를 담고 있습니다. 데이터 전처리부터 멀티모델 앙상블 학습, 모델 저장, 추론 API 서버까지 end-to-end 파이프라인입니다.

- **팀** : 그랜:토리 · **기간** : 2026.08.03 \~ 2026.08.12
- **데모** : https://sarangbang-autobiography.vercel.app

<br>

## 👥 팀 구성 · 내가 맡은 파트

팀 그랜:토리에서 **DA(데이터 분석) 파트를 본인이 맡았고**, 이 레포는 그 DA 파트의 코드입니다.
서비스 전체(구술 녹화 · 숏폼/자서전 제작 · 가족 전달)는 팀 전체가 함께 만들었습니다.

## 🎯 내가 맡은 파트 — DA (전처리 · 앙상블 모델링 · 추론)

우울 측정 AI의 **데이터 수집 → 전처리 → 모델링 → 평가 → 서비스화 로직**을 전담했습니다.

| 단계 | 내용 |
|------|------|
| **데이터셋 구축** | AI-Hub 고령자 구술 데이터(No. 71703) 기반, 화자 단위 통합·정제·라벨 검증 → **최종 5,819명** (총 1.2억+ 자, 평균 2만 자) |
| **라벨 설계** | 데이터셋의 우울 2문항 합(0\~8점)에서 **2점 이상을 위험군**으로 정의(전체의 23%), 라벨 일관성 검증. PHQ-2와 비슷한 문항이지만 표준 PHQ-2(0\~6점, 3점 이상)와는 점수 범위와 기준이 다른 자체 정의 |
| **긴 텍스트 최적화** | 평균 2만 자(최대 28만 자) 구술 → 문장 단위 청킹, 모델별 청크 크기 조정. 계산량 때문에 구술의 앞부분만 사용(KLUE-BERT 최대 6,400자, KoBigBird 최대 17,500자) |
| **모델링** | KoBigBird · KLUE-BERT · KLUE-RoBERTa · KcELECTRA **4종 × 시드 × Focal Loss = 총 41개 모델** 학습·비교 (이 레포의 스크립트에는 그중 대표 구성만 들어 있음: `02`는 6개 학습, `03`은 2개 저장) |
| **앙상블** | 상위 4개 모델 평균 앙상블 → 개별 최고(0.651) 대비 **AUROC 0.674**. 모델 순위와 앙상블 구성을 평가 화자로 정했기 때문에 낙관적으로 치우친 값(아래 재평가 참고) |
| **평가** | 불균형(위험군 23%) 대응 지표: **F2 0.59 · Balanced Acc 0.61 · Recall 0.82** (임계값 0.5, 위험군 258명 중 211명 포착) |
| **서비스화** | 최종 4모델 앙상블 추론 서버(FastAPI) 구현 — 위험도(0\~1) → **행복지수(0\~100)** 변환, 3단계 상태(안정/관찰/상담) 분류, **점수 재조정**(경험적으로 정한 구간을 0\~1로 펴는 선형 변환) 로직 설계 |

> **당시 보고한 수치** (화자 1,164명 · 위험군 258 / 정상 906, 4개 모델 앙상블)
> AUROC 0.6741 · recall 0.818 · precision 0.280 · F2 0.591 · Balanced Acc 0.609 · Youden's J 최적 0.546 (놓치지 않기 우선으로 0.5 채택)

### 재평가 — 같은 평가 화자에서 다시 계산한 결과

레포에 들어 있는 예측 파일 2개와 원본 라벨로 지표를 다시 계산하고, 비교 기준이 되는 가벼운 모델을 추가로 학습했습니다
([`src/07_baseline_cv.py`](./src/07_baseline_cv.py), 결과 [`results/baseline_cv.json`](./results/baseline_cv.json)).

| 모델 | AUROC (평가 화자 1,164명) | 95% 신뢰구간 |
|---|---|---|
| 인구학 변수만 (나이·성별·교육, 로지스틱 회귀) | 0.564 | 0.522–0.605 |
| **TF-IDF + 로지스틱 회귀** (글자 2·3-gram) | **0.645** | 0.608–0.682 |
| KLUE-BERT (레포의 예측 파일) | 0.625 | 0.589–0.661 |
| KoBigBird (레포의 예측 파일) | 0.647 | 0.610–0.684 |
| 위 트랜스포머 2개 평균 | 0.652 | 0.615–0.688 |

- **트랜스포머와 TF-IDF 기준 모델의 차이가 신뢰구간 안에 있습니다.** 수 시간 학습한 트랜스포머 2개 평균(0.652)과
  CPU에서 몇 분 만에 학습한 TF-IDF 모델(0.645)이 사실상 같은 수준입니다. 구술 텍스트에 신호가 있기는 하지만(인구학 변수만 쓴 0.564보다 높음),
  이 과제에서 모델을 키우는 것으로는 성능이 거의 오르지 않았습니다.
- **당시 보고한 0.674는 이 레포의 파일로 재현되지 않습니다.** 4개 모델 앙상블의 값인데 예측 파일은 2개만 들어 있고, 그 2개의 평균은 0.652입니다.
  또 모델 순위·앙상블 구성·최적 epoch를 모두 이 평가 화자로 정했기 때문에 0.674 자체도 낙관적으로 치우친 값입니다.
- **기준 모델의 5-fold 교차검증**(fold마다 벡터화와 모델을 다시 학습, 전체 5,823명): TF-IDF 0.634 ± 0.011, 인구학 변수 0.542 ± 0.014.
- **임계값 0.5에서의 실제 동작**(트랜스포머 2개 평균): 재현율 0.79, 정밀도 0.28, **특이도 0.41**. 전체 화자의 63%가 위험군으로 표시되고,
  위험군으로 표시된 사람 중 72%는 위험군이 아닙니다.

**이전 README의 서술 중 정정한 것**

- "5-fold·10-fold 교차검증으로 일반화 성능 확인" — `05_cross_validation.py`는 모델을 다시 학습하지 않고 이미 나온 예측을 나눠 AUROC를 평균합니다.
  그래서 전체 AUROC와 거의 같은 값이 나올 뿐(재계산 시 0.651 · 0.649 vs 전체 0.652) 일반화 성능을 보여 주지 않습니다. 교차검증이 아니라 평가셋 안에서의 안정성 확인입니다.
- "보라매병원 연구(0.632)를 상회" — 데이터·라벨·대상 집단이 달라 직접 비교할 수 없고, 신뢰구간을 고려하면 차이를 말할 수 없어 삭제했습니다.
- "점수 보정(calibration)" — 통계적 보정(Platt·isotonic)이 아니라 눈으로 정한 두 값 사이를 펴는 선형 변환입니다.
  전제로 삼은 "원점수가 0.50\~0.78에만 분포한다"도 레포의 예측 파일과 맞지 않습니다(두 모델 평균의 실제 범위 0.16\~0.83, 1,164명 중 427명이 0.5 미만).
- 서빙 코드(`04`)는 평가(`02`)와 청킹·집계 방식이 다릅니다(평가: 문장 단위 청크의 평균 / 서빙: 글자 단위 청크의 최댓값). 위 지표는 평가 방식 기준이며 배포된 API의 동작을 그대로 나타내지 않습니다.

<br>

## 🧩 파이프라인

```
JSON 원본 (AI-Hub 71703)
      │
      ▼  00_make_speakers_csv.py     화자별 통합 + PHQ-2/GAD-2·인구통계 정리
speakers.csv
      │
      ▼  01_preprocess_kssds.py      KSSDS 문장 분리 (문장 경계 보존)
speakers_kssds.csv
      │
      ▼  02_train_ensemble.py        6모델 학습 → 상위 조합 앙상블 평가 (AUROC/recall/precision)
      │
      ▼  03_save_final_models.py     최고 조합 재학습 + best 가중치 저장
saved_models/  (KoBigBird_1234_f · KoBigBird_2222_f · KLUE-RoBERTa_888 · KLUE-BERT_42)
      │
      ▼  04_serve_api.py             FastAPI 4모델 앙상블 서버 (위험도 → 보정 → 행복지수 → 3단계)
POST /predict → {"risk_score": 0.39, "happiness_index": 61, "grade": "안정", ...}

  └─ 검증 ─┬─ 05_cross_validation.py   평가셋을 fold로 나눈 AUROC 평균 (재학습 없음)
           ├─ 06_evaluation.py         F2 · Balanced Acc · recall · Youden's J · 임계값 분석
           └─ 07_baseline_cv.py        기준 모델(TF-IDF·인구학) + 재학습하는 5-fold 교차검증
```

<br>

## 📂 구조

```
.
├── src/
│   ├── 00_make_speakers_csv.py    # JSON 원본 → 화자별 통합 CSV
│   ├── 01_preprocess_kssds.py     # KSSDS 문장 분리 전처리
│   ├── 02_train_ensemble.py       # 멀티모델 앙상블 학습·평가
│   ├── 03_save_final_models.py    # 최종 모델 학습·저장 (--demo 옵션)
│   ├── 04_serve_api.py            # 4모델 앙상블 추론 API (FastAPI + 점수 보정)
│   ├── 05_cross_validation.py     # 평가셋을 fold로 나눈 AUROC 평균 (재학습 없음)
│   ├── 06_evaluation.py           # F2·Balanced Acc·recall·Youden's J 평가
│   └── 07_baseline_cv.py          # 기준 모델과 재학습하는 5-fold 교차검증
├── results/
│   ├── pred_FINAL_KLUE-BERT_42.csv      # 화자별 예측 확률 (4개 중 2개만 포함)
│   ├── pred_FINAL_KoBigBird_1234_f.csv
│   └── baseline_cv.json                 # 기준 모델·재평가 결과
├── docs/
│   └── API명세.md                       # 추론 API 입출력 명세
├── requirements.txt
└── .gitignore
```

<br>

## 🛠 기술 스택

- **Language** : Python 3.12
- **Modeling** : PyTorch, Transformers (KoBigBird · KLUE-BERT · KLUE-RoBERTa · KcELECTRA)
- **전처리** : KSSDS (한국어 문장 분리)
- **Serving** : FastAPI · Uvicorn · pydantic
- **Data** : AI-Hub 고령자 근현대 경험 기반 스토리 구술 데이터 (No. 71703)

<br>

## ▶️ 실행

```bash
pip install -r requirements.txt

# 0) JSON 원본 → 화자별 통합 CSV (AI-Hub 71703 데이터 필요)
python src/00_make_speakers_csv.py --json-dir ./json --out speakers.csv

# 1) 전처리 (KSSDS는 버전 충돌 방지를 위해 별도 conda 환경 권장 — 파일 상단 주석 참고)
python src/01_preprocess_kssds.py --in speakers.csv --out speakers_kssds.csv

# 2) 앙상블 학습·평가
python src/02_train_ensemble.py --csv speakers_kssds.csv

# 3) 최종 모델 저장 (데모: 가볍고 빠름 / 기본: 성능 버전)
python src/03_save_final_models.py --csv speakers_kssds.csv --out ./saved_models --demo

# 4) API 서버 실행 (saved_models/ 안의 모델을 자동 스캔·앙상블)
python src/04_serve_api.py --models ./saved_models
# → POST http://localhost:8000/predict  {"text": "..."}

# 5~6) 검증·평가
#    두 스크립트는 4개 모델의 예측 파일 이름이 코드에 적혀 있는데, 레포에는 그중 2개만 다른 이름(pred_FINAL_*)으로 들어 있어
#    그대로는 실행되지 않습니다. 레포의 파일로 다시 계산하려면 아래 7)을 사용하세요.
python src/05_cross_validation.py --truth speakers_clean.csv --pred-dir ./results
python src/06_evaluation.py --truth speakers_clean.csv --pred-dir ./results

# 7) 기준 모델 + 재학습하는 5-fold 교차검증 (CPU 약 20분, scikit-learn만 필요)
python src/07_baseline_cv.py --csv speakers.csv --pred-dir ./results --out results/baseline_cv.json
```

추론 API 입출력 형식은 [`docs/API명세.md`](./docs/API명세.md) 참고.

<br>

## 📌 참고 · 한계

- 우울 라벨은 **PHQ-2와 비슷한 2문항의 합에 자체 기준(2점 이상)을 적용한 것**이며, 의학적 진단이 아닌 **관찰·연결**이 목적입니다. (자서전/회고가 주 기능, 정서 관찰은 보조 기능 — 진단을 하지 않는 비의료 범위로 설계)
- 원본·전처리 데이터(`speakers*.csv`)와 학습된 모델 가중치는 **AI-Hub 라이선스 및 용량 문제로 포함하지 않습니다.** AI-Hub에서 데이터셋 71703을 받아 화자 단위로 통합한 뒤 실행하세요.
- **성능이 낮습니다.** AUROC 0.65 수준이고 임계값 0.5에서 위험군으로 표시된 사람의 72%가 실제로는 위험군이 아닙니다. 가족에게 "상담 권유"를 전달하는 용도로 쓰기에는 오경보가 많습니다.
- 평가 화자를 모델 선택에도 사용했기 때문에, 모델 선택에 쓰지 않은 별도 시험셋에서의 성능은 확인하지 못했습니다.
- 구술 데이터는 과거 회상 중심이라 현재 우울 예측에 구조적 한계가 있으며, 향후 **음성·표정 멀티모달**로 확장 예정입니다.
