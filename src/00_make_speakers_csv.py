# -*- coding: utf-8 -*-
"""
00. JSON 원본 → 화자별 통합 speakers.csv 생성

AI-Hub 고령자 구술 데이터(dataSetSn=71703)의 JSON 파일들을 읽어 같은 화자(speaker_id)의
여러 답변을 하나로 합치고, 우울점수(PHQ-2 유사)·불안점수(GAD-2 유사)·인구통계 정보를 정리한다.
파이프라인의 첫 단계이며, 출력(speakers.csv)은 01_preprocess_kssds.py의 입력이 된다.

[JSON 구조]
  qa: [{question, answer}, ...]     # 질문·답변 쌍
  teller: [{나이, 성별, 교육년, 우울점수(...), 불안점수(...)}]
  keyword: 감정 키워드
  audioFile: "{speaker_id}_{keyword}.wav"

[출력] speakers.csv
  컬럼: speaker_id, n_keywords, age, sex, edu, PHQ2, GAD2, n_chars, text

[실행] python src/00_make_speakers_csv.py --json-dir ./json --out speakers.csv
"""
import os
import json
import glob
import argparse
from collections import defaultdict
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="JSON 원본 → 화자별 통합 CSV")
    ap.add_argument("--json-dir", default=".", help="JSON 파일들이 있는 폴더")
    ap.add_argument("--out", default="speakers.csv", help="출력 CSV 경로")
    args = ap.parse_args()

    # 화자별로 데이터 누적
    speakers = defaultdict(lambda: {
        "texts": [], "keywords": [],
        "age": None, "sex": None, "edu": None,
        "dep": None, "anx": None,
    })

    files = glob.glob(os.path.join(args.json_dir, "*.json"))
    print(f"JSON 파일 {len(files)}개 발견. 처리 시작...")

    for i, fp in enumerate(files):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # speaker_id = 파일명 앞부분 (예: 10001_고맙다.json → 10001)
        base = os.path.basename(fp)
        speaker_id = base.split("_")[0]

        # 답변 텍스트 합치기
        answers = [qa.get("answer", "").strip()
                   for qa in data.get("qa", []) if qa.get("answer", "").strip()]
        if answers:
            speakers[speaker_id]["texts"].append(" ".join(answers))

        # 키워드
        kw = data.get("keyword", "")
        if kw:
            speakers[speaker_id]["keywords"].append(kw)

        # 화자 정보(teller): 처음 한 번만
        teller = data.get("teller", [])
        if teller and speakers[speaker_id]["age"] is None:
            t = teller[0]
            speakers[speaker_id]["age"] = t.get("나이")
            speakers[speaker_id]["sex"] = t.get("성별")
            speakers[speaker_id]["edu"] = t.get("교육년")
            # 우울점수 2문항 합 (PHQ-2 유사)
            dep_keys = [k for k in t if k.startswith("우울점수")]
            speakers[speaker_id]["dep"] = sum(int(t.get(k, 0)) for k in dep_keys)
            # 불안점수 2문항 합 (GAD-2 유사)
            anx_keys = [k for k in t if k.startswith("불안점수")]
            speakers[speaker_id]["anx"] = sum(int(t.get(k, 0)) for k in anx_keys)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)} 처리")

    # DataFrame 변환
    rows = []
    for sid, d in speakers.items():
        full_text = " ".join(d["texts"])
        rows.append({
            "speaker_id": sid,
            "n_keywords": len(d["keywords"]),
            "age": d["age"],
            "sex": d["sex"],
            "edu": d["edu"],
            "PHQ2": d["dep"],       # 우울점수 합 (0~8)
            "GAD2": d["anx"],       # 불안점수 합 (0~8)
            "n_chars": len(full_text),
            "text": full_text,
        })

    df = pd.DataFrame(rows).dropna(subset=["PHQ2"]).reset_index(drop=True)  # 라벨 없는 화자 제외
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\n완료! {args.out} 저장")
    print(f"  총 화자: {len(df):,}명")
    print(f"  총 글자: {df['n_chars'].sum():,}자")
    print(f"  화자당 평균: {df['n_chars'].mean():,.0f}자")
    print(f"  위험군(PHQ2>=2): {(df['PHQ2'] >= 2).sum():,}명 "
          f"({(df['PHQ2'] >= 2).mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
