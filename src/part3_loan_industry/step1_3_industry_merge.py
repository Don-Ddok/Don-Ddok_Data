# -*- coding: utf-8 -*-
"""
1단계-3: 은행 업종 ↔ KOSIS 업종 연결, 생산지수 붙이기
  - 업종 이름의 띄어쓰기·쉼표·세미콜론 차이를 없애고 맞춘다
  - 대구·경북 지역 지수가 있으면 그걸, 없으면 전국 지수를 쓴다
입력: step1_2_firm_vars.parquet, ../production_index_industry.csv
출력: step1_3_industry.parquet  (은행 원본 파생 파일 — GitHub에 올리지 않는다)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_2_firm_vars.parquet")
kosis = pd.read_csv(HERE.parent / "production_index_industry.csv")
kosis = kosis[kosis["level"] == "중분류"].copy()          # 제조업 중분류 25개만
print(f"입력: {len(df):,}행, 법인 {df['법인ID'].nunique():,}개 / KOSIS 중분류 {kosis['industry'].nunique()}개")

# KOSIS가 공표하지 않은 계열은 0으로 채워져 있다(예: 대구 목재 60개월 전부 0).
# 실제 생산 0이 아니므로 "값 없음"으로 바꿔서, 전국 지수로 넘어가게 한다.
zero_series = kosis[kosis["value"] <= 0].groupby(["region", "industry"]).size()
print(f"값이 0 이하인 계열(미공표로 판단, 결측 처리): {zero_series.to_dict() if len(zero_series) else '없음'}")
kosis.loc[kosis["value"] <= 0, "value"] = pd.NA
kosis = kosis.dropna(subset=["value"])


# ① 업종 이름 맞추기 — 공백·쉼표·세미콜론을 지운 뒤 비교
def norm(s):
    return re.sub(r"[\s,;]", "", str(s))


kosis_names = sorted(kosis["industry"].unique())
lookup = {norm(k): k for k in kosis_names}
df["KOSIS업종"] = df["업종_중분류"].map(lambda s: lookup.get(norm(s)))

bank_names = sorted(df["업종_중분류"].unique())
matched = {b: lookup[norm(b)] for b in bank_names if norm(b) in lookup}
print(f"\n은행 업종 {len(bank_names)}개 중 KOSIS와 연결된 업종: {len(matched)}개")
unmatched_kosis = [k for k in kosis_names if k not in matched.values()]
print(f"KOSIS 쪽에서 짝이 없는 업종: {unmatched_kosis}")

diff_spelling = {b: k for b, k in matched.items() if b != k}
print(f"표기가 달라서 정규화로 맞춘 업종 {len(diff_spelling)}개:")
for b, k in diff_spelling.items():
    print(f"  은행 '{b}'  →  KOSIS '{k}'")

n_firm_matched = df.loc[df["KOSIS업종"].notna(), "법인ID"].nunique()
print(f"\n연결된 법인: {n_firm_matched:,}개  [기대값 3,678]")

# ② 생산지수 붙이기 — 지역 우선, 없으면 전국
df["ym"] = df["기준년월"].astype(int)
df["kosis_region"] = df["사업장_시도"].map({"대구": "대구광역시", "경북": "경상북도"})

regional = kosis[kosis["region"].isin(["대구광역시", "경상북도"])][["region", "industry", "ym", "value"]]
regional = regional.rename(columns={"region": "kosis_region", "industry": "KOSIS업종", "value": "생산지수_지역"})
national = kosis[kosis["region"] == "전국"][["industry", "ym", "value"]]
national = national.rename(columns={"industry": "KOSIS업종", "value": "생산지수_전국"})

df = df.merge(regional, on=["kosis_region", "KOSIS업종", "ym"], how="left")
df = df.merge(national, on=["KOSIS업종", "ym"], how="left")
df["생산지수"] = df["생산지수_지역"].fillna(df["생산지수_전국"])
df["생산지수_출처"] = pd.NA
df.loc[df["생산지수_지역"].notna(), "생산지수_출처"] = "지역"
df.loc[df["생산지수_지역"].isna() & df["생산지수_전국"].notna(), "생산지수_출처"] = "전국대체"

m = df["KOSIS업종"].notna()
print(f"\n연결된 행: {m.sum():,}행 / 연결 안 된 행(비제조업): {(~m).sum():,}행")
print(f"연결된 행 중 생산지수가 비어 있는 행: {df.loc[m, '생산지수'].isna().sum():,}  [기대값 0]")
print(f"생산지수 출처: {df.loc[m, '생산지수_출처'].value_counts().to_dict()}")

fb = (df[m & (df["생산지수_출처"] == "전국대체")]
      .groupby(["사업장_시도", "KOSIS업종"])["법인ID"].nunique().reset_index(name="법인수"))
print("\n전국 지수로 대체된 업종 (지역 지수 없음):")
print(fb.to_string(index=False))

df.to_parquet(HERE / "step1_3_industry.parquet", index=False)
print(f"\n저장: step1_3_industry.parquet ({len(df):,}행 × {df.shape[1]}열)")
