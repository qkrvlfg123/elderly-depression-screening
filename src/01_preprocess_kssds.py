# -*- coding: utf-8 -*-
"""
01. KSSDS 문장 분리 전처리

화자별로 통합된 구술 텍스트를 KSSDS(한국어 문장 분리기)로 문장 단위로 나눈 뒤,
마침표+공백으로 다시 이어붙여 저장한다.
학습·평가의 청킹이 문장부호(. ! ?) 경계에서 텍스트를 나누므로, 문장 끝에 마침표를 넣어 두기 위함.

[실행 환경] KSSDS 전용 환경 권장 (transformers 버전 충돌 방지)
    conda create -n kssds_env python=3.12 -y
    conda activate kssds_env
    pip install transformers==4.42.4 KSSDS pandas

[실행]
    python src/01_preprocess_kssds.py --in speakers.csv --out speakers_clean_kssds.csv

[입력 CSV 형식] speaker_id, text, PHQ2 (최소 text 컬럼 필요)
[출력] text 컬럼이 KSSDS 문장 분리 버전으로 교체된 CSV

화자 수가 약 5,800명이라 시간이 오래 걸린다. 진행 상황은 200명마다 출력된다.
"""
import sys
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="KSSDS 문장 분리 전처리")
    ap.add_argument("--in", dest="csv_in", required=True, help="입력 CSV 경로")
    ap.add_argument("--out", dest="csv_out", required=True, help="출력 CSV 경로")
    args = ap.parse_args()

    try:
        from KSSDS import KSSDS
    except ImportError as e:
        print("[설치 필요] conda activate kssds_env 후: "
              "pip install transformers==4.42.4 KSSDS pandas")
        print(f"  상세: {e}")
        sys.exit(1)

    print("KSSDS 로딩 중... (처음이면 모델 다운로드, 수백 MB)")
    kssds = KSSDS()
    print("로딩 완료.\n")

    df = pd.read_csv(args.csv_in).dropna(subset=["text"]).reset_index(drop=True)
    print(f"총 {len(df)}명. KSSDS 문장 분리 시작 (시간이 걸립니다)...\n")

    new_texts = []
    for i, row in df.iterrows():
        text = str(row["text"])
        try:
            sents = kssds.split_sentences(text)
            # 문장을 '마침표+공백'으로 다시 이어붙임 (청킹이 문장부호 경계에서 나누도록)
            joined = ". ".join(s.strip() for s in sents if s.strip())
            new_texts.append(joined)
        except Exception:
            new_texts.append(text)  # 실패 시 원문 유지
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)} 완료")

    df["text"] = new_texts
    df.to_csv(args.csv_out, index=False, encoding="utf-8-sig")
    print(f"\n완료! 저장됨: {args.csv_out}")
    print("다음 단계: 02_train_final.py 로 학습을 실행하세요.")


if __name__ == "__main__":
    main()
