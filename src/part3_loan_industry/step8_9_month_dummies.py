# -*- coding: utf-8 -*-
"""
8단계-9: 달력월(1~12월) 더미 — 사후 보조. 계절 패턴이 β3(그리고 8-7의 사전 추세)를 설명하나
  설계·판정 규칙은 08_달력통제_이후_결과.md 5절에 실행 전 확정. 주 결과·등급 판정에는 쓰지 않는다.
  월 더미 = 충격 달 t의 달력월 더미 11개(1월 기준). 월 고정효과(29개월 각각)가 아니다.
  모형 A: 3-3 식 + 월 더미 / 모형 B: 8-6 식(영업일수 2항) + 월 더미
  종속변수: 주 결과(h=6, t-1→t+6), 사전 6개월(t-7→t-1)
입력: 패널(common.py), step3_psm_matched.parquet, 외부데이터/workdays_2021_2025.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, XCOLS, fe_reg, load_panel, lsdv_check, match_weights, twoway_p

sys.stdout.reconfigure(encoding="utf-8")

BASE_B3 = -0.231
HALF = abs(BASE_B3) / 2
WORKDAYS = HERE.parents[1] / "외부데이터" / "workdays_2021_2025.csv"

df = load_panel()
wd = pd.read_csv(WORKDAYS, encoding="utf-8-sig")[["ym", "전년동월차"]].rename(columns={"전년동월차": "영업일수차"})
df = df.merge(wd, on="ym", how="left")
assert df["영업일수차"].notna().all(), "영업일수가 빠진 달이 있음"
df["영업일수×노출"] = df["영업일수차"] * df["외환노출"]
MONTHS = [f"월{m}" for m in range(2, 13)]
for m in range(2, 13):
    df[f"월{m}"] = (df["ym"] % 100 == m).astype(float)

df["_m"] = (df["ym"] // 100) * 12 + (df["ym"] % 100 - 1)


def value_at(offset):
    s = df[["법인ID", "_m", "여신_운전자금대출잔액"]].copy()
    s["_m"] -= offset
    return df[["법인ID", "_m"]].merge(s, on=["법인ID", "_m"], how="left")["여신_운전자금대출잔액"].to_numpy()


df["사전6"] = np.log(value_at(-1) + 1) - np.log(value_at(-7) + 1)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))


def twoway_V(r):
    """법인·월 이중 클러스터 분산 행렬(Cameron-Gelbach-Miller), twoway_p와 같은 계산"""
    X, W, e, inv = r["X"], r["W"], r["e"], r["XtWX_inv"]
    n, k = X.shape
    V = np.zeros((k, k))
    for keys, sign in [(["법인ID"], 1), (["ym"], 1), (["법인ID", "ym"], -1)]:
        g = pd.factorize(pd.MultiIndex.from_frame(r["d"][keys]))[0] if len(keys) > 1 \
            else pd.factorize(r["d"][keys[0]])[0]
        G = g.max() + 1
        s = np.zeros((G, k))
        np.add.at(s, g, X * (W * e)[:, None])
        V += sign * inv @ (s.T @ s) @ inv * G / (G - 1) * (n - 1) / (n - k)
    return V


def joint_month_p(r, xcols):
    """월 더미 11개가 함께 0인지: 이중 클러스터 Wald → F(11, 적은 쪽 클러스터 수 − 1). 분산 행렬이 불안정하면 None"""
    idx = [xcols.index(c) for c in MONTHS]
    V = twoway_V(r)[np.ix_(idx, idx)]
    if np.linalg.eigvalsh(V).min() <= 0:
        return None
    b = r["b"][idx]
    F = float(b @ np.linalg.solve(V, b)) / len(idx)
    return float(stats.f.sf(F, len(idx), min(r["법인ID"][2], r["ym"][2]) - 1))


OUTCOMES = [("주 결과 h=6", "대출증감률_h6"), ("사전 6개월", "사전6")]
SPECS = [
    ("기준(월 더미 없음)", XCOLS),
    ("모형 A: + 월 더미", XCOLS + MONTHS),
    ("모형 B: 영업일수 2항 + 월 더미", XCOLS + ["영업일수차", "영업일수×노출"] + MONTHS),
]

rows, keep, base_se = [], {}, {}
for oname, y in OUTCOMES:
    for sname, xcols in SPECS:
        r = fe_reg(df, y, W_MATCH, xcols)
        se, p3 = twoway_p(r, 1)
        if sname.startswith("기준"):
            base_se[oname] = se
        joint = joint_month_p(r, xcols) if any(c in xcols for c in MONTHS) else None
        rows.append({"종속변수": oname, "모형": sname, "회사": r["firms"], "월": r["ym"][2],
                     "β1(비노출)": r["b"][0], "β1+β3(노출)": r["b"][0] + r["b"][1], "β3": r["b"][1],
                     "SE": se, "SE배수": se / base_se[oname], "p(이중)": p3,
                     "월더미 공동p": joint if joint is not None else np.nan})
        keep[(oname, sname)] = (r, xcols, y)

res = pd.DataFrame(rows)
print("=== 8-9 결과 (사후 보조, 매칭 1:3 가중, 법인·월 이중 클러스터) ===")
with pd.option_context("display.width", 260, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))

print("\n=== 판정 (사전 규칙) ===")
for oname, _ in OUTCOMES:
    a = res[(res["종속변수"] == oname) & (res["모형"] == "모형 A: + 월 더미")].iloc[0]
    b3, p, mult = a["β3"], a["p(이중)"], a["SE배수"]
    if oname.startswith("주"):
        explained = not ((b3 < 0) and (abs(b3) >= HALF))
        verdict = "계절과 분리되지 않음" if explained else "월 더미로 설명되지 않음"
    else:
        explained = (p >= 0.10) and (abs(b3) < HALF)
        verdict = "사전 추세가 계절로 설명됨(사라짐)" if explained else "사전 추세는 계절로 설명되지 않음(남음)"
    if explained and mult > 1.5:
        verdict = "판정 불가(정보 손실)"
    print(f"{oname}: β3 {b3:+.3f}, p {p:.3f}, SE배수 {mult:.2f} → {verdict}")

for (oname, sname), (r, xcols, y) in keep.items():
    if sname.startswith("모형 A"):
        chk = lsdv_check(r, y, xcols)
        print(f"이중 확인 ({oname}, 모형 A): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}")
