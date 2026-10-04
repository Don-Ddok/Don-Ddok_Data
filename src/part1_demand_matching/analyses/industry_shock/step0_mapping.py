# -*- coding: utf-8 -*-
"""0단계 (회귀 없음): 파트3 업종 매핑으로 연결되는 법인 수, 업종 생산지수 YoY 변동 — 파트3 미러의 step1_3_industry.parquet 사용"""
import os, re, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
P3 = r"C:\test\external\part3_work\통계 프로젝트\여신업종분석\단계별 분석"
KOS = r"C:\test\external\Don-Ddok_Data\data\external\production_index_industry.csv"
DATA = r"C:\test\data\processed\df_ready.csv"
print("미러 파일:", [f for f in os.listdir(P3) if f.endswith(".parquet")])
k = pd.read_csv(KOS)
k = k[k["level"] == "중분류"].copy(); k.loc[k["value"] <= 0, "value"] = np.nan
print(f"KOSIS 생산지수: 지역 {sorted(k['region'].unique())}, 중분류 {k['industry'].nunique()}개, 기간 {k['ym'].min()}~{k['ym'].max()}")
norm = lambda s: re.sub(r"[\s,;]", "", str(s))
lookup = {norm(x): x for x in k["industry"].unique()}
d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "업종_중분류", "exposed", "외환_수출실적금액"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)]
last = d.dropna(subset=["업종_중분류"]).sort_values("기준년월").groupby("법인ID")["업종_중분류"].last()
f = pd.DataFrame({"업종": last}); f["KOSIS"] = f["업종"].map(lambda s: lookup.get(norm(s)))
f["노출"] = d.groupby("법인ID")["exposed"].first(); f["수출"] = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
m = f[f["KOSIS"].notna()]
print(f"\n법인 {len(f):,} 중 KOSIS 중분류 연결 {len(m):,} (노출 {int(m['노출'].sum()):,} / 비노출 {int((m['노출'] == 0).sum()):,}); "
      f"수출 실적 있는 525곳 중 연결 {int(m['수출'].sum())}곳 / 연결된 업종 {m['KOSIS'].nunique()}개")
print("업종별 연결 법인 (노출/전체):")
print(m.groupby("KOSIS").agg(전체=("노출", "size"), 노출=("노출", "sum"), 수출=("수출", "sum")).sort_values("전체", ascending=False).to_string())
# 생산지수 YoY (지역 우선, 없으면 전국) — 업종×월 변동
k["yoy"] = k.groupby(["region", "industry"])["value"].pct_change(12) * 100
y = k[k["ym"].between(202301, 202512)]
for reg in ["대구광역시", "경상북도", "전국"]:
    w = y[y["region"] == reg].pivot(index="ym", columns="industry", values="yoy")
    w = w[[c for c in w.columns if c in set(m["KOSIS"])]].dropna(axis=1, how="all")
    c = w.corr().values; iu = np.triu_indices_from(c, 1)
    print(f"\n[{reg}] 업종 {w.shape[1]}개 × {w.shape[0]}개월, YoY 표준편차 중앙값 {w.std().median():.1f}%p, 업종 간 상관 중앙값 {np.nanmedian(c[iu]):.2f} (p25 {np.nanpercentile(c[iu], 25):.2f}, p75 {np.nanpercentile(c[iu], 75):.2f}), 미공표 계열 {int(w.isna().all().sum())}")
