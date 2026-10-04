# -*- coding: utf-8 -*-
"""충격 쪽 기술 통계 (결과변수 없음): 업종(KSIC 중분류) × 월 배분 수출, 전년동월비 변동 — 대구·경북 합산과 지역별"""
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H4 = r"C:\test\external\industry_export\hs4"
w = pd.read_csv("hs4_ksic_weights.csv", dtype={"hs4": str, "ksic": str})
h = pd.concat([pd.read_csv(os.path.join(H4, f), encoding="utf-8-sig", dtype={"hsSgn": str}) for f in os.listdir(H4)])
x = h.merge(w, left_on="hsSgn", right_on="hs4")
x["v"] = x["expUsdAmt"] * x["w"]
x = x[x["ksic"].str.isdigit()]
for lab, g in [("합산", x.groupby(["ksic", "기준년월"])["v"].sum()), ("대구", x[x.지역 == "대구"].groupby(["ksic", "기준년월"])["v"].sum()),
               ("경북", x[x.지역 == "경북"].groupby(["ksic", "기준년월"])["v"].sum())]:
    p = g.unstack("ksic").sort_index()
    yoy = (p / p.shift(12) - 1) * 100
    yy = yoy.loc[202301:]
    s = pd.DataFrame({"월평균 수출(천$)": p.mean().round(0), "0인 달": (p == 0).sum(), "YoY 표준편차(%p)": yy.std().round(1),
                      "|YoY|>100% 달": (yy.abs() > 100).sum()})
    c = yy.corr().values; iu = np.triu_indices_from(c, 1)
    print(f"\n[{lab}] 업종 {p.shape[1]}개, 업종 간 YoY 상관 중앙값 {np.nanmedian(c[iu]):.2f}\n{s.to_string()}")
