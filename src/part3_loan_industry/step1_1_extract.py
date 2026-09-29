# -*- coding: utf-8 -*-
"""
1단계-1: 원본에서 필요한 컬럼만 뽑고, 대구·경북 + 금융·보험업 제외로 걸러낸다.
출력: step1_1_filtered.parquet  (은행 원본 파생 파일 — GitHub에 올리지 않는다)
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()

HERE = Path(__file__).resolve().parent                  # .../여신업종분석/단계별 분석
RAW = HERE.parents[2] / "은행_원본데이터" / "(iM뱅크) 2026 교육용 법인 익명데이터.xlsx"
OUT = HERE / "step1_1_filtered.parquet"

# 명세서 2절 — 가져올 컬럼 24개
COLS = [
    # 키·속성 (6)
    "기준년월", "법인ID", "업종_대분류", "업종_중분류", "사업장_시도", "법인_고객등급",
    # 여신 합계 (3)
    "여신_운전자금대출잔액", "여신_시설자금대출잔액", "여신한도금액",
    # 운전자금 하위 7개 (주택자금 제외)
    "운전_할인어음잔액", "운전_당좌대출잔액", "운전_일반자금대출잔액", "운전_무역금융잔액",
    "운전_기업구매자금대출잔액", "운전_외상매출채권담보대출잔액", "운전_기타운전자금대출잔액",
    # 시설자금 하위 3개 (주택자금 제외)
    "시설_일반자금대출잔액", "시설_에너지절약시설대출잔액", "시설_기타시설자금대출잔액",
    # 수신 (3) — 규모분위 계산용
    "요구불예금잔액", "거치식예금잔액", "적립식예금잔액",
    # 외환 (2) — 처치 정의용
    "외환_수출실적금액", "외환_수입실적금액",
]

print(f"원본 읽는 중... (6~7분 걸림) {RAW.name}")
df = pd.read_excel(RAW, usecols=COLS, engine="openpyxl")
print(f"원본 로드: {len(df):,}행 × {df.shape[1]}열, 법인 {df['법인ID'].nunique():,}개 ({time.time()-t0:.0f}초)")

# 명세서 3절 — 필터
n0 = len(df)
df = df[df["사업장_시도"].isin(["대구", "경북"])]
n1 = len(df)
df = df[df["업종_대분류"] != "금융 및 보험업"]
n2 = len(df)

print(f"대구·경북 필터: {n0:,} → {n1:,}행 (제외 {n0-n1:,})")
print(f"금융·보험업 제외: {n1:,} → {n2:,}행 (제외 {n1-n2:,})")
print(f"남은 법인: {df['법인ID'].nunique():,}개")
print(f"기간: {df['기준년월'].min()} ~ {df['기준년월'].max()}, 월 수 {df['기준년월'].nunique()}")
print(f"지역별 행 수: {df['사업장_시도'].value_counts().to_dict()}")

na = df.isna().sum()
na = na[na > 0]
print(f"결측이 있는 컬럼: {na.to_dict() if len(na) else '없음'}")

df.to_parquet(OUT, index=False)
print(f"저장: {OUT.name} ({OUT.stat().st_size/1e6:.1f}MB), 총 {time.time()-t0:.0f}초")
