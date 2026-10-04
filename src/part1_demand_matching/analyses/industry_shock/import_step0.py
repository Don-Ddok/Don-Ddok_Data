# -*- coding: utf-8 -*-
"""수입 0단계 (회귀 없음): HS4 수입 금액 검증, 사용 업종 15개의 업종 수입 계열, 1단계 대상(수입 실적 있는 노출 법인) 수"""
import os, re, sys, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H4 = r"C:\test\external\industry_export\hs4"; HS2 = r"C:\Users\yues7\Downloads\새 폴더"
EXR = r"C:\test\external\Don-Ddok_Data\data\external\export_region.csv"
USE = ["10", "11", "13", "14", "17", "20", "22", "23", "24", "25", "26", "27", "28", "29", "30"]
h = pd.concat([pd.read_csv(os.path.join(H4, f), encoding="utf-8-sig", dtype={"hsSgn": str, "류": str}) for f in os.listdir(H4)])
# 검증 ①: 류별 수입 합계 vs HS2 파일
c4 = h.groupby(["지역", "기준년월", "류"]).agg(s=("impUsdAmt", "sum"), n=("hsSgn", "size")).reset_index()
h2 = []
for f in glob.glob(os.path.join(HS2, "지역별 실적(품목별)*.csv")):
    t = pd.read_csv(f, encoding="utf-8-sig", dtype=str); t = t[t["기간"] != "총계"]
    h2.append(pd.DataFrame({"지역": t["지역"].map({"대구광역시": "대구", "경상북도": "경북"}), "기준년월": t["기간"].str.replace("-", "").astype(int),
                            "류": t["HS코드"], "v": pd.to_numeric(t["수입 금액"].str.replace(",", ""))}))
m = c4.merge(pd.concat(h2), on=["지역", "기준년월", "류"])
print(f"① 수입 류별 대조 {len(m):,}칸: 반올림 허용 안 {(abs(m.s - m.v) <= m.n * 0.5 + 1).mean():.3f}, |차| 최대 {int(abs(m.s - m.v).max())}천 달러")
ex = pd.read_csv(EXR, encoding="utf-8-sig").rename(columns={"region": "지역", "ym": "기준년월"})
t4 = h.groupby(["지역", "기준년월"])["impUsdAmt"].sum().reset_index().merge(ex[["지역", "기준년월", "imp_amt"]])
print(f"② 류 15~97 수입 ÷ 월 수입 총액: 중앙값 {(t4.impUsdAmt / t4.imp_amt).median():.4f}, 최소 {(t4.impUsdAmt / t4.imp_amt).min():.4f}")
w = pd.read_csv("hs4_ksic_weights.csv", dtype={"hs4": str, "ksic": str})
x = h.merge(w, left_on="hsSgn", right_on="hs4"); x["v"] = x["impUsdAmt"] * x["w"]
un = x[x["hsSgn"].isin(["8524", "8485", "8807", "8806", "3827", "2404", "8549", "2424"])]
print(f"③ HS 2022 신설 8개 코드의 수입 비중(류 15~97 대비): {un.impUsdAmt.sum() / h.impUsdAmt.sum():.4f}")
p = x[x["ksic"].isin(USE)].groupby(["ksic", "기준년월"])["v"].sum().unstack("ksic").sort_index()
yy = (p / p.shift(12) - 1).loc[202301:] * 100
s = pd.DataFrame({"월평균 수입(천$)": p.mean().round(0), "YoY 표준편차(%p)": yy.std().round(1), "|YoY|>100% 달": (yy.abs() > 100).sum()})
c = yy.corr().values; iu = np.triu_indices_from(c, 1)
print(f"\n④ 사용 업종 15개 합산 업종 수입 (업종 간 YoY 상관 중앙값 {np.nanmedian(c[iu]):.2f})\n{s.to_string()}")
d = pd.read_csv(r"C:\test\data\processed\df_ready.csv", usecols=["기준년월", "법인ID", "사업장_시도", "업종_중분류", "외환_수입실적금액", "exposed"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip(); d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)]
sys.path.insert(0, "."); from build_mapping import KSIC
norm = lambda z: re.sub(r"[\s,;]", "", str(z)); n2k = {norm(v): k for k, v in KSIC.items()}
last = d.dropna(subset=["업종_중분류"]).sort_values("기준년월").groupby("법인ID")["업종_중분류"].last().map(lambda z: n2k.get(norm(z)))
imp = d.groupby("법인ID")["외환_수입실적금액"].max() > 0
ids = last[last.isin(USE)].index.intersection(imp[imp].index)
e = d[d["법인ID"].isin(ids)]
print(f"\n⑤ 1단계 대상: 수입 실적 있는 사용 업종 법인 {len(ids)}곳 (그중 KSIC 26 {int((last.loc[ids] == '26').sum())}곳), 수입 실적 > 0인 달 비율 {(e['외환_수입실적금액'] > 0).mean():.3f}, "
      f"법인별 수입 > 0인 달 중앙값 {e.groupby('법인ID')['외환_수입실적금액'].apply(lambda z: (z > 0).sum()).median():.0f}")
