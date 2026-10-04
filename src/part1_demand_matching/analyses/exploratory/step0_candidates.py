# -*- coding: utf-8 -*-
"""0단계 (회귀 없음): 탐색 후보 변수의 존재·0 비율·t−1→t+6 값이 바뀐 관측 비율·예상 표본 / 이질성 집단 크기 / 노출 쪼개기 크기"""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
allc = pd.read_csv(DATA, nrows=1).columns.tolist()
DEP = ["요구불예금잔액", "거치식예금잔액", "적립식예금잔액", "수익증권잔액", "신탁잔액", "퇴직연금잔액"]
OP8 = ["운전_할인어음잔액", "운전_당좌대출잔액", "운전_일반자금대출잔액", "운전_무역금융잔액", "운전_주택자금대출잔액",
       "운전_기업구매자금대출잔액", "운전_외상매출채권담보대출잔액", "운전_기타운전자금대출잔액"]
C = ["신용카드사용금액", "체크카드사용금액", "자동이체금액", "창구거래금액", "인터넷뱅킹거래금액", "스마트뱅킹거래금액"]
G = ["법인_고객등급", "전담고객여부", "업종_대분류"]
need = ["기준년월", "법인ID", "사업장_시도", "exposed", "외환_수출실적금액", "외환_수입실적금액", "여신한도금액",
        "여신_운전자금대출잔액", "여신_시설자금대출잔액"] + DEP + OP8 + C + G
miss = [c for c in need if c not in allc]
print("df_ready에 없는 열:", miss)
d = pd.read_csv(DATA, usecols=[c for c in need if c in allc])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
d["midx"] = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
fid = pd.factorize(d["법인ID"])[0].astype(np.int64)
key = pd.Index(fid * 10000 + d["midx"].to_numpy())
def at(v, off):
    pos = key.get_indexer(fid * 10000 + d["midx"].to_numpy() + off); o = np.full(len(d), np.nan); ok = pos >= 0; o[ok] = v[pos[ok]]; return o
d["총수신"] = d[DEP].sum(axis=1)
d["요구불비중"] = np.where(d["총수신"] > 0, d["요구불예금잔액"] / d["총수신"], np.nan)
d["한도소진율"] = np.where(d["여신한도금액"] > 0, d["여신_운전자금대출잔액"] / d["여신한도금액"], np.nan)
rows = []
def cand(group, var, ratio=False):
    v = d[var].to_numpy(float)
    if ratio:
        smp = d[var].notna()
        firms = d.loc[smp, "법인ID"].unique()
    else:
        firms = d.loc[d.groupby("법인ID")[var].transform("max") > 0, "법인ID"].unique()
    inS = d["법인ID"].isin(firms).to_numpy()
    a, b = at(v, -1), at(v, 6)
    m = inS & ~np.isnan(a) & ~np.isnan(b)
    ch = np.mean(a[m] != b[m]) if m.sum() else np.nan
    ex = d.loc[inS & (d["exposed"] == 1).to_numpy(), "법인ID"].nunique()
    rows.append({"묶음": group, "변수": var, "단위": "수준(%p)" if ratio else "ln(+1)", "0 비율(표본 내)": np.mean(v[inS] == 0) if not ratio else np.mean(v[inS & ~np.isnan(v)] == 0),
                 "t−1→t+6 바뀐 비율": ch, "표본 법인": len(firms), "노출 법인": ex, "h=6 가능 행": int(m.sum()),
                 "제외 후보(<5%)": (ch < 0.05) if not np.isnan(ch) else True})
for v in ["총수신"]: cand("A", v)
cand("A", "요구불비중", ratio=True)
for v in ["수익증권잔액", "신탁잔액", "퇴직연금잔액"]: cand("A", v)
cand("B", "여신한도금액"); cand("B", "한도소진율", ratio=True); cand("B", "여신_시설자금대출잔액")
for v in OP8: cand("B", v)
for v in C: cand("C", v)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
t.to_csv("step0_candidates.csv", index=False, encoding="utf-8-sig")
# 한도소진율 분포
x = d["한도소진율"].dropna(); print(f"\n한도소진율: 관측 {len(x):,}, >1 비율 {np.mean(x > 1)*100:.1f}%, p50 {x.median():.2f}, p99 {x.quantile(.99):.2f}")
# D 이질성: 요구불 표본, 2023-01~03 값
s = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0]
e = s[s["기준년월"] <= 202303]
first = e.sort_values("기준년월").groupby("법인ID").first()
print(f"\n[D] 요구불 표본 {s['법인ID'].nunique():,}곳 중 2023-01~03 관측 있는 법인 {len(first):,}곳")
expo = s.groupby("법인ID")["exposed"].first()
for g in G + ["사업장_시도"]:
    if g not in first: print(g, "없음"); continue
    c = pd.crosstab(first[g], expo.reindex(first.index)); c.columns = ["비노출", "노출"]
    print(f"\n{g}:\n{c.to_string()}")
size = e.groupby("법인ID")["총수신"].mean()
med = size.median()
c = pd.crosstab(np.where(size >= med, "상위", "하위"), expo.reindex(size.index)); c.columns = ["비노출", "노출"]
print(f"\n규모(2023-01~03 평균 총수신, 중앙값 {med:.3g} 기준):\n{c.to_string()}")
# E 노출 쪼개기
f = d.groupby("법인ID").agg(x=("외환_수출실적금액", "max"), m=("외환_수입실적금액", "max"), e=("exposed", "first"))
print(f"\n[E] 수출 실적 있음(수출만 또는 둘 다) {(f.x > 0).sum()}, 수출 없이 수입만 {((f.x <= 0) & (f.m > 0)).sum()}, 수출만(수입 없음) {((f.x > 0) & (f.m <= 0)).sum()}, 둘 다 {((f.x > 0) & (f.m > 0)).sum()}, 비노출 {(f.e == 0).sum()}")
