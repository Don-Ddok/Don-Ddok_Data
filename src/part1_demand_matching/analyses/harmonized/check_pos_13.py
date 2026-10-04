# -*- coding: utf-8 -*-
"""요구불 POS 재사용 확인 + 운전자금 9,541 vs 9,528 차이 원인"""
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, run_common
import run_harmonized as R
d = R.load_df()
s = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0]
r, _ = run_common(Panel(s, "요구불예금잔액"), R.HS, positive=True, label="POS", log=lambda x: None)
old = pd.read_csv(r"C:\test\outputs\did_demand_deposit\results_table.csv")
old = old[old["모형"] == "A-양수"].set_index("h")
m = r.set_index("h").join(old[["β3", "SE", "N"]], rsuffix="_old")
print("POS vs A-양수: β3 최대차", (m["β3"] - m["β3_old"]).abs().max(), "SE 최대상대차", (m["SE"] / m["SE_old"] - 1).abs().max(), "N 같음", (m["N"] == m["N_old"]).all())
# 13곳
P3 = R.P3
p = pd.read_parquet(os.path.join(P3, "step1_loan_industry_panel.parquet"))
p["사업장_시도"] = p["사업장_시도"].astype(str).str.strip()
hold_df = set(d.loc[d.groupby("법인ID")["여신_운전자금대출잔액"].transform("max") > 0, "법인ID"])
pall = set(p["법인ID"]); p_dg = set(p.loc[p["사업장_시도"].isin(["대구", "경북"]), "법인ID"])
miss = hold_df - p_dg
print("df_ready 운전자금 보유 중 파트3 대구·경북 패널에 없음:", len(miss), "/ 파트3 전체 패널에는 있음:", len(miss & pall))
x = d[d["법인ID"].isin(miss)]
print("  이 법인들의 업종 결측 비율:", round(x["업종_중분류"].isna().mean(), 3), "/ 관측 월 수 중앙값:", x.groupby("법인ID").size().median())
if len(miss & pall):
    print("  파트3에서의 지역:", p.loc[p["법인ID"].isin(miss), "사업장_시도"].value_counts().to_dict())
u = pd.read_csv(R.DATA, usecols=["법인ID", "업종_대분류"])
print("  13곳 업종_대분류:", u[u["법인ID"].isin(miss)].drop_duplicates("법인ID")["업종_대분류"].value_counts().to_dict())
