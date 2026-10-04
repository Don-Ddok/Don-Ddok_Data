# -*- coding: utf-8 -*-
"""기술 통계 (회귀 없음): 노출 법인의 월 요구불 입금/잔액, 출금/잔액 비율 분포와 h=6 입금 차이의 잔액 기준 근사(참고값)
표본: 요구불 표본(10,723곳) 중 노출 법인, 잔액 > 0인 법인×월. 출력: flow_ratio.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\mechanism_inflow"
d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "요구불예금잔액", "요구불입금금액", "요구불출금금액"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)]
d = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0]
e = d[(d["exposed"] == 1) & (d["요구불예금잔액"] > 0)]
rows = []
for c, nm in [("요구불입금금액", "입금/잔액"), ("요구불출금금액", "출금/잔액")]:
    r = e[c] / e["요구불예금잔액"]
    rows.append({"비율": nm, "법인×월": len(r), "법인": e["법인ID"].nunique(), "p25": r.quantile(.25), "중앙값": r.median(), "p75": r.quantile(.75)})
t = pd.DataFrame(rows)
res = pd.read_csv(os.path.join(OUT, "results.csv"))
inflow_pct = float(res[(res["분석"] == "입금") & (res["h"] == 6)]["환산"].iloc[0])
med = float(t.loc[t["비율"] == "입금/잔액", "중앙값"].iloc[0])
q1, q3 = float(t.loc[t["비율"] == "입금/잔액", "p25"].iloc[0]), float(t.loc[t["비율"] == "입금/잔액", "p75"].iloc[0])
t.to_csv(os.path.join(OUT, "flow_ratio.csv"), index=False, encoding="utf-8-sig")
print(t.round(3).to_string(index=False))
print(f"h=6 입금 차이 {inflow_pct:+.2f}% × 입금/잔액 중앙값 {med:.2f} ≈ 잔액의 {inflow_pct * med:+.2f}% (p25~p75 비율로 {inflow_pct * q1:+.2f}% ~ {inflow_pct * q3:+.2f}%) — 참고값")
