# -*- coding: utf-8 -*-
"""
6단계-3: 무역금융 기술통계 (검정 없음 — 5-1에서 검정 제외한 항목)
  ① 보유 구성: 누가 무역금융을 쓰나
  ② 수출 실적과의 연결: 같은 회사 안에서 무역금융 잔액과 외환 수출실적이 같이 움직이나
     ("수출이 줄면 기계적으로 줄어든다"는 5-1 판단의 근거 확인)
  ③ 지역 수출이 감소한 달 vs 증가한 달의 6개월 무역금융 변화(t−1 → t+6), 보유 회사 평균
입력: step1_loan_industry_panel.parquet, 수출 충격(common.py)
"""
import sys

import numpy as np
import pandas as pd

from common import HERE, load_shock

sys.stdout.reconfigure(encoding="utf-8")

C = "운전_무역금융잔액"
df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
holders = df.groupby("법인ID")[C].max() > 0
d = df[df["법인ID"].isin(holders[holders].index)].copy()
firm = d.groupby("법인ID").agg(노출=("외환노출", "first"), 수출노출=("수출노출", "first"),
                              업종=("업종_대분류", "first"))

print("=== ① 보유 구성 ===")
print(f"무역금융 보유 회사 {len(firm)}곳: 외환노출 {int(firm['노출'].sum())}곳({firm['노출'].mean()*100:.1f}%), "
      f"수출노출 {int(firm['수출노출'].sum())}곳 / 비노출 {int((firm['노출'] == 0).sum())}곳")
print(f"업종: 제조업 {(firm['업종'] == '제조업').mean()*100:.1f}%, 도매 및 소매업 "
      f"{(firm['업종'] == '도매 및 소매업').mean()*100:.1f}%")
pos = d[d[C] > 0]
print(f"무역금융 잔액이 있는 달 중 같은 달 수출실적>0: {(pos['외환_수출실적금액'] > 0).mean()*100:.1f}%, "
      f"수입실적>0: {(pos['외환_수입실적금액'] > 0).mean()*100:.1f}%")

print("\n=== ② 같은 회사 안에서 무역금융 vs 수출실적 (로그, 회사 내 상관) ===")
e = d[d["외환노출"] == 1].copy()
e["l_tf"], e["l_ex"] = np.log1p(e[C]), np.log1p(e["외환_수출실적금액"])
dm = e[["l_tf", "l_ex"]] - e.groupby("법인ID")[["l_tf", "l_ex"]].transform("mean")
ok = e.groupby("법인ID")[["l_tf", "l_ex"]].transform("std").gt(0).all(axis=1)
print(f"노출 보유 회사 중 두 값 모두 변동이 있는 회사 {e.loc[ok, '법인ID'].nunique()}곳, "
      f"회사 내 상관 {dm[ok]['l_tf'].corr(dm[ok]['l_ex']):+.2f}")

print("\n=== ③ 지역 수출 감소 달 vs 증가 달, 6개월 무역금융 변화 (보유 회사, 회귀 아님) ===")
d = d.merge(load_shock()[["사업장_시도", "ym", "수출YoY"]], on=["사업장_시도", "ym"], how="left")
d["_m"] = (d["ym"] // 100) * 12 + (d["ym"] % 100 - 1)
base = d[["법인ID", "_m", C]]
prev = base.assign(_m=base["_m"] + 1).rename(columns={C: "t-1"})
fwd = base.assign(_m=base["_m"] - 6).rename(columns={C: "t+6"})
d = d.merge(prev, on=["법인ID", "_m"], how="left").merge(fwd, on=["법인ID", "_m"], how="left")
d["변화"] = np.log1p(d["t+6"]) - np.log1p(d["t-1"])
d = d[d["변화"].notna() & d["수출YoY"].notna()]
d["지역 수출"] = np.where(d["수출YoY"] < 0, "감소 달", "증가 달")
d["구분"] = d["외환노출"].map({1: "노출", 0: "비노출"})
t = d.groupby(["구분", "지역 수출"]).agg(행=("변화", "size"), 회사=("법인ID", "nunique"),
                                      평균변화_로그=("변화", "mean"),
                                      무변동비율=("변화", lambda x: (np.abs(np.exp(x) - 1) <= 0.05).mean() * 100))
t["평균변화(%)"] = (np.exp(t["평균변화_로그"]) - 1) * 100
print(t.drop(columns="평균변화_로그").round(1).to_string())
