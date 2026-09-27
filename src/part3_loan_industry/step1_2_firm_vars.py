# -*- coding: utf-8 -*-
"""
1단계-2: 회사 단위로 한 번만 정해지는 값 4개를 만든다.
  외환노출, 수출노출, 거래기간, 규모분위
입력: step1_1_filtered.parquet
출력: step1_2_firm_vars.parquet  (은행 원본 파생 파일 — GitHub에 올리지 않는다)
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_1_filtered.parquet")
print(f"입력: {len(df):,}행, 법인 {df['법인ID'].nunique():,}개")

# 월별 행에 "이 달에 수출/수입 실적이 있었나" 표시
df["_수출있음"] = df["외환_수출실적금액"] > 0
df["_수입있음"] = df["외환_수입실적금액"] > 0

# 규모 = 예금 3개 + 대출 2개 합계 (방향성 문서의 "수신+여신 합계")
df["_규모"] = df[["요구불예금잔액", "거치식예금잔액", "적립식예금잔액",
                 "여신_운전자금대출잔액", "여신_시설자금대출잔액"]].sum(axis=1)

# 회사 단위로 묶어서 계산 — 회사 하나당 한 줄
firm = df.groupby("법인ID").agg(
    수출횟수=("_수출있음", "sum"),
    수입횟수=("_수입있음", "sum"),
    거래기간=("기준년월", "nunique"),
    평균규모=("_규모", "mean"),
).reset_index()

firm["외환노출"] = ((firm["수출횟수"] + firm["수입횟수"]) > 0).astype(int)
firm["수출노출"] = (firm["수출횟수"] > 0).astype(int)

# 규모분위: 회사 하나를 한 번씩만 세어서 5등분 (1=작음, 5=큼)
firm["규모분위"] = pd.qcut(firm["평균규모"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

print("\n=== 회사 단위 결과 ===")
print(f"외환노출 법인: {firm['외환노출'].sum():,}개 ({firm['외환노출'].mean()*100:.1f}%)  [기대값 1,032]")
print(f"수출노출 법인: {firm['수출노출'].sum():,}개 ({firm['수출노출'].mean()*100:.1f}%)  [기대값 523]")
print(f"거래기간: 최소 {firm['거래기간'].min()}개월, 최대 {firm['거래기간'].max()}개월, 평균 {firm['거래기간'].mean():.1f}개월")
print(f"36개월 전부 관측(균형패널): {(firm['거래기간']==36).sum():,}개")
print(f"규모분위별 법인 수: {firm['규모분위'].value_counts().sort_index().to_dict()}")

# 참고 점검: 오래 관측된 회사일수록 외환노출로 분류되기 쉬운가 (CLAUDE.md 열린 이슈 4)
bins = pd.cut(firm["거래기간"], [0, 12, 24, 35, 36], labels=["1~12개월", "13~24개월", "25~35개월", "36개월"])
chk = firm.groupby(bins, observed=True).agg(법인수=("법인ID", "count"), 외환노출비율=("외환노출", "mean"))
chk["외환노출비율"] = (chk["외환노출비율"] * 100).round(1)
print("\n=== 참고: 관측기간별 외환노출 비율 ===")
print(chk.to_string())

# 월별 행에 회사 값을 다시 붙이기
keep = ["법인ID", "외환노출", "수출노출", "거래기간", "평균규모", "규모분위"]
out = df.drop(columns=["_수출있음", "_수입있음", "_규모"]).merge(firm[keep], on="법인ID", how="left")
out.to_parquet(HERE / "step1_2_firm_vars.parquet", index=False)
print(f"\n저장: step1_2_firm_vars.parquet ({len(out):,}행 × {out.shape[1]}열)")
