# 사랑방 자서전 — 우울 위험 선별 AI

> 매일 질문 하나에 말로 답하는 녹화 한 번으로 — 자서전이 쌓이고 · 건강 신호가 감지되고 · 또래와 연결됩니다.

**사랑방 자서전**은 고령층이 오늘의 질문에 구술로 답하면 그 영상이 자동으로 숏폼·자서전으로 제작되고, 발화 텍스트에서 **우울 신호를 비의료적으로 관찰**해 가족에게 전달하는 시니어 헬스케어 서비스입니다.

이 레포지토리는 그중 **Data Analyst(DA) 파트 — 우울 위험 선별 AI**의 코드를 담고 있습니다. 데이터 전처리부터 멀티모델 앙상블 학습, 모델 저장, 추론 API 서버까지 end-to-end 파이프라인입니다.

- **팀** : 그랜:토리 (의료 6기 2팀) · **기간** : 2026.08.03 \~ 2026.08.12
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
| **라벨 설계** | PHQ-2 우울 선별검사 기준(**2점 이상 = 위험군**)으로 학습 정답지 정의, 라벨 일관성 검증 |
| **긴 텍스트 최적화** | 평균 2만 자(최대 28만 자) 구술 → 문장 단위 청킹으로 정보 손실 최소화, 모델별 청크 크기 최적화 |
| **모델링** | KoBigBird · KLUE-BERT · KLUE-RoBERTa · KcELECTRA **4종 × 시드 × Focal Loss = 총 41개 모델** 학습·비교 |
| **앙상블** | 최고 성능 조합을 평균 앙상블 → 개별 최고(0.651) 대비 향상된 **AUROC 0.674** |
| **평가** | 불균형(위험군 23%) 대응 지표: **F2 0.59 · Balanced Acc 0.61 · Recall 0.82** (임계값 0.5, 위험군 258명 중 211명 포착) |
| **서비스화** | 최종 4모델 앙상블 추론 서버(FastAPI) 구현 — 위험도(0\~1) → **행복지수(0\~100)** 변환, 3단계 상태(안정/관찰/상담) 분류, **점수 보정(calibration)** 로직 설계 |

**핵심 성과** — 텍스트 단독 **AUROC 0.674**로, 서울대 보라매병원 연구의 텍스트 단독 우울·자살위험 평가(0.632)를 상회. 5-fold(0.675)·10-fold(0.672) 교차검증으로 일반화 성능 확인.

> **검증 결과 요약** (화자 1,164명 · 위험군 258 / 정상 906)
> AUROC 0.6741 · 5-fold 0.6753 · 10-fold 0.6722 · recall 0.818 · precision 0.280 · F2 0.591 · Balanced Acc 0.609 · Youden's J 최적 0.546 (놓치지 않기 우선으로 0.5 채택)

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
POST /predict → {"risk_score": 0.32, "happiness_index": 68, "grade": "안정", ...}

  └─ 검증 ─┬─ 05_cross_validation.py   5-fold / 10-fold + 과적합 방어 비교
           └─ 06_evaluation.py         F2 · Balanced Acc · recall · Youden's J · 임계값 분석
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
│   ├── 05_cross_validation.py     # 5-fold / 10-fold 교차검증
│   └── 06_evaluation.py           # F2·Balanced Acc·recall·Youden's J 평가
├── results/
│   ├── pred_FINAL_KLUE-BERT_42.csv      # 화자별 예측 확률 (샘플)
│   └── pred_FINAL_KoBigBird_1234_f.csv
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

# 5~6) 검증·평가 (예측 결과 pred_*.csv 필요)
python src/05_cross_validation.py --truth speakers_clean.csv --pred-dir ./results
python src/06_evaluation.py --truth speakers_clean.csv --pred-dir ./results
```

추론 API 입출력 형식은 [`docs/API명세.md`](./docs/API명세.md) 참고.

<br>

## 📌 참고 · 한계

- 우울 라벨은 **PHQ-2 선별검사 기준**이며, 의학적 진단이 아닌 **관찰·연결**이 목적입니다. (자서전/회고가 주 기능, 정서 관찰은 보조 기능 — 의료기기 규제 회피 설계)
- 원본·전처리 데이터(`speakers*.csv`)와 학습된 모델 가중치는 **AI-Hub 라이선스 및 용량 문제로 포함하지 않습니다.** AI-Hub에서 데이터셋 71703을 받아 화자 단위로 통합한 뒤 실행하세요.
- 구술 데이터는 과거 회상 중심이라 현재 우울 예측에 구조적 한계가 있으며, 향후 **음성·표정 멀티모달**로 확장 예정입니다.
