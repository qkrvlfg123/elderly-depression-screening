
# 사랑방 자서전 — 우울 위험 선별 AI

> 매일 질문 하나에 말로 답하는 녹화 한 번으로 — 자서전이 쌓이고 · 건강 신호가 감지되고 · 또래와 연결됩니다.

**사랑방 자서전**은 고령층이 오늘의 질문에 구술로 답하면 그 영상이 자동으로 숏폼·자서전으로 제작되고, 발화 텍스트에서 **우울 신호를 비의료적으로 관찰**해 가족에게 전달하는 시니어 헬스케어 서비스입니다.

이 레포지토리는 그중 **Data Analyst(DA) 파트 — 우울 위험 선별 AI**의 코드를 담고 있습니다. 데이터 정리부터 학습, 평가, 추론 API 서버까지의 파이프라인입니다.

- **팀** : 그랜:토리 (의료 6기 2팀) · **기간** : 2026.08.03 \~ 2026.08.12
- **데모** : https://sarangbang-autobiography.vercel.app

<br>

## 👥 팀 구성

팀 그랜:토리에서 DA(데이터 분석) 파트를 맡았고, 이 레포는 그 DA 파트의 코드입니다.
서비스 전체(구술 녹화 · 숏폼/자서전 제작 · 가족 전달)는 팀 전체가 함께 만들었습니다.

<br>

## 🎯 개요

| 단계 | 내용 |
|------|------|
| **데이터** | AI-Hub 고령자 구술 데이터(No. 71703), 화자 단위 통합 5,823명 → 키워드마다 우울점수가 다르게 기록된 4명 제외 → **5,819명** (화자당 평균 약 2.1만 자) |
| **전처리** | KSSDS 문장 분리 |
| **라벨** | PHQ-2 유사 우울점수 **2점 이상 = 위험군** (23.3%). 3점 이상(9.9%) 기준 결과도 함께 보고 |
| **분할** | 화자 단위 8:2 (`train_test_split`, random_state=42) → train 4,655 / test 1,164명 (위험군 258명) |
| **모델** | KoBigBird 2 · KLUE-RoBERTa 1 · KLUE-BERT 1 = **4모델 앙상블** (확률 평균) |
| **결과** | test AUROC **0.661** (95% CI 0.625–0.697), 답변 1개 분량(800자) 입력 시 **0.604** |
| **서빙** | FastAPI. 모델 평균 → 점수 보정 → 행복지수 · 3단계 등급(🟢 안정 · 🟡 관찰 · 🔴 상담 권유) |

<br>

## 🧠 모델

| 모델 | 손실 | 입력 | 학습 |
|------|------|------|------|
| KoBigBird_1234_f | Focal (γ=2) | 3,072토큰 · 6,000자 × 최대 5청크 | lr 1e-5, 배치 1 × 누적 16, 5 epoch |
| KoBigBird_2222_f | Focal (γ=2) | 3,072토큰 · 6,000자 × 최대 5청크 | lr 1e-5, 배치 1 × 누적 16, 5 epoch |
| KLUE-RoBERTa_888 | BCE | 512토큰 · 800자 × 최대 8청크 | lr 2e-5, 배치 16, 5 epoch |
| KLUE-BERT_42 | BCE | 512토큰 · 800자 × 최대 8청크 | lr 2e-5, 배치 16, 5 epoch |

- 모델 이름의 숫자는 시드, `_f`는 Focal 손실입니다. 위험군 가중치(pos_weight)는 5입니다.
- 화자 텍스트를 문장부호(. ! ?) 경계에서 청크로 나누고, 화자 확률은 청크 확률의 평균입니다. 한국어는 토큰당 약 2자라서 청크가 토큰 상한 안에 들어가도록 맞췄습니다.

<br>

## 📈 결과 (test 1,164명)

서빙(`04_serve_api.py`)과 같은 처리로 평가했습니다(`src/03_evaluate_final.py`).

| 모델 | AUROC, 전체 텍스트 | AUROC, 앞 800자 |
|------|:---:|:---:|
| KoBigBird_1234_f | 0.647 | 0.599 |
| KoBigBird_2222_f | 0.647 | 0.598 |
| KLUE-BERT_42 | 0.625 | 0.567 |
| KLUE-RoBERTa_888 | 0.587 | 0.521 |
| **4모델 앙상블** | **0.661 (0.625–0.697)** | **0.604 (0.564–0.644)** |
| 4모델 앙상블, PHQ2 ≥ 3 | 0.668 (0.617–0.717) | 0.599 (0.541–0.653) |

**등급별 결과** (PHQ2 ≥ 2)

| 입력 | 🟢 · 🟡 · 🔴 비율 | 🔴 재현율 · 정밀도 · 특이도 | 🟡 이상 재현율 · 정밀도 · 표시 비율 |
|------|:---:|:---:|:---:|
| 전체 텍스트 | 72.1% · 12.3% · 15.6% | 0.260 · 0.368 · 0.873 | 0.415 · 0.329 · 27.9% |
| **앞 800자 (답변 1개)** | **65.6% · 10.5% · 23.9%** | **0.341 · 0.317 · 0.790** | **0.450 · 0.290 · 34.4%** |

- 서비스 입력은 하루 답변 1개이고, 데이터에서 답변 1개는 중앙값 약 780자입니다. 서비스에서 기대할 성능은 800자 행에 가깝습니다.
- 800자 입력에서 🔴는 전체의 약 24%를 표시하고, 그중 실제 위험군은 32%입니다(무작위로 고를 때 22%). 위험군의 34%를 잡습니다.
- 괄호는 test 화자를 2,000회 재표집한 부트스트랩 95% 신뢰구간입니다. 상세 수치는 `results/final_serving.json`에 있습니다.

### 발표 수치(0.674)와의 관계

발표 때는 별도 val 없이 test 점수로 모델마다 epoch을 고르고, 학습한 여러 모델 중 test 점수가 가장 높은 4개 조합을 골랐습니다. 그래서 위 test 결과는 낙관적인 값입니다. 이를 확인하려고 같은 데이터·분할·설정으로 4모델을 다시 학습하고, epoch마다 test 예측을 저장했습니다(`src/02_train_final.py`).

| 모델 | ep1 | ep2 | ep3 | ep4 | ep5 |
|------|:---:|:---:|:---:|:---:|:---:|
| KoBigBird_1234_f | 0.584 | 0.606 | 0.632 | 0.643 | 0.645 |
| KoBigBird_2222_f | 0.582 | 0.593 | 0.624 | 0.636 | 0.642 |
| KLUE-RoBERTa_888 | 0.556 | 0.587 | 0.568 | 0.592 | 0.586 |
| KLUE-BERT_42 | 0.612 | 0.629 | 0.608 | 0.593 | 0.574 |

| 4모델 앙상블 AUROC | 값 | test로 고른 것 |
|------|:---:|------|
| 발표 수치 (실험 때 저장한 예측) | 0.674 | epoch, 4개 조합 |
| 배포 모델 (위 결과) | 0.661 | epoch, 4개 조합 |
| 재학습, test로 epoch 선택 | 0.659 (0.622–0.695) | epoch |
| **재학습, epoch 5 고정** | **0.641 (0.603–0.677)** | 없음 (조합은 최종본을 따름) |

- epoch을 미리 정한 값(0.641)이 test로 고르지 않은 성능에 가장 가깝습니다.
- 발표 수치는 실험 때 저장한 예측 파일로 계산했고, 배포 가중치는 따로 학습한 것이라 0.013 차이가 납니다(KLUE-RoBERTa_888이 0.646 → 0.587로 가장 크게 다릅니다).
- 이전 버전 README의 5-fold/10-fold 수치(0.675/0.672)는 재학습 없이 같은 test 예측을 나눠 평균한 값이라 교차검증이 아니어서 뺐습니다.

<br>

## 🧩 파이프라인

```
JSON 원본 (AI-Hub 71703)
      │
      ▼  00_make_speakers_csv.py     화자별 통합 + PHQ-2/GAD-2·인구통계 정리
speakers.csv
      │
      ▼  01_preprocess_kssds.py      KSSDS 문장 분리
speakers_clean_kssds.csv
      │
      ▼  02_train_final.py           4모델 학습 (5 epoch, epoch별 test 예측 저장) → results/repro_final4.json
      ▼  03_evaluate_final.py        서빙 구성(ensemble.json) 작성 + test 평가 → results/final_serving.json
      ▼  04_serve_api.py             FastAPI 앙상블 서버 (위험도 → 보정 → 행복지수 → 3단계)
POST /predict → {"risk_score": 0.32, "happiness_index": 68, "grade": "안정", ...}
```

<br>

## 📂 구조

```
.
├── src/
│   ├── common.py                  # 청킹·집계, 데이터 로드, 지표 (공통)
│   ├── 00_make_speakers_csv.py    # JSON 원본 → 화자별 통합 CSV
│   ├── 01_preprocess_kssds.py     # KSSDS 문장 분리
│   ├── 02_train_final.py          # 4모델 학습
│   ├── 03_evaluate_final.py       # 서빙 구성 작성 + test 평가
│   └── 04_serve_api.py            # 앙상블 추론 API (FastAPI)
├── results/
│   ├── final_serving.json         # 배포 모델 test 지표 (등급별, 신뢰구간 포함)
│   ├── repro_final4.json          # 재학습 epoch별 test AUROC
│   └── experiments.csv            # 학습·평가 기록
├── docs/
│   └── API명세.md                 # 추론 API 입출력 명세
├── requirements.txt
└── .gitignore
```

화자별 예측 파일(`results/preds/`)은 AI-Hub 화자 ID와 연결된 모델 출력이라 레포에 올리지 않습니다.

<br>

## 🛠 기술 스택

- **Language** : Python 3.12+
- **Modeling** : PyTorch, Transformers (KoBigBird · KLUE-BERT · KLUE-RoBERTa)
- **Serving** : FastAPI · Uvicorn · pydantic
- **Data** : AI-Hub 고령자 근현대 경험 기반 스토리 구술 데이터 (No. 71703)

<br>

## ▶️ 실행

```bash
pip install -r requirements.txt

# 0) JSON 원본 → 화자별 통합 CSV (AI-Hub 71703 데이터 필요)
python src/00_make_speakers_csv.py --json-dir ./json --out speakers.csv

# 1) KSSDS 문장 분리
python src/01_preprocess_kssds.py --in speakers.csv --out speakers_clean_kssds.csv

# 2) 4모델 학습 (RTX 4070 기준 KoBigBird 모델당 약 7시간, KLUE 계열 약 1시간). 끝난 모델은 건너뜀
python src/02_train_final.py --csv speakers_clean_kssds.csv     # → saved_models/repro/<모델>/

# 3) 서빙 구성 작성 + test 평가 (약 30분). 배포 가중치 4개를 saved_models/final/<모델>/ 에 둔다
python src/03_evaluate_final.py --csv speakers_clean_kssds.csv

# 4) API 서버
python src/04_serve_api.py
# → POST http://localhost:8000/predict  {"text": "..."}
```

서빙 등급은 모델 평균 점수를 0.50–0.78 구간에서 0–1로 편 위험도에 적용합니다. 위험도 0.5 이상이면 상담 권유, 0.4 이상이면 관찰, 그 아래는 안정입니다. 추론 API 입출력 형식은 [`docs/API명세.md`](./docs/API명세.md)에 있습니다.

<br>

## 📌 참고 · 한계

- 우울 라벨은 **PHQ-2 유사 선별 문항**이며, 의학적 진단이 아닌 **관찰·연결**이 목적입니다. (자서전/회고가 주 기능, 정서 관찰은 보조 기능)
- 원본 데이터(`speakers*.csv`)와 학습된 모델 가중치는 **AI-Hub 라이선스 및 용량 문제로 포함하지 않습니다.** AI-Hub에서 데이터셋 71703을 받아 화자 단위로 통합한 뒤 실행하세요.
- epoch과 앙상블 조합을 test로 골랐기 때문에 test AUROC 0.661은 낙관적인 값이고, epoch을 미리 정해 재학습하면 0.641입니다. 새로운 화자에서의 성능은 별도 데이터로 확인해야 합니다.
- 답변 1개 분량(800자) 입력에서는 AUROC 0.604이고, 상담 권유 등급은 약 24%를 표시하는 대신 위험군의 약 34%만 잡습니다.
- 점수 보정 구간(0.50–0.78)과 등급 기준(0.4 / 0.5)은 시연 문장으로 정한 값입니다. 보정 후 0.4–0.5 구간이 원점수 0.612–0.64로 좁아 🟡 등급은 약 10%에 그칩니다.
- 구술 데이터는 과거 회상 중심이라 현재 우울 예측에 구조적 한계가 있으며, 향후 **음성·표정 멀티모달**로 확장 예정입니다.
