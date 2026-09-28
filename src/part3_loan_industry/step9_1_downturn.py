# -*- coding: utf-8 -*-
"""
9단계-1: 하락 국면 β3⁻ — 노출·비노출 차이가 수출이 "떨어질 때" 생기나
  (설계는 06_추가분석_설계와_결과.md 1절, 2026-09-23 실행 전 확정)
  충격 분리: 수출YoY⁻ = min(수출YoY, 0), 수출YoY⁺ = max(수출YoY, 0)
  식: 대출증감률_h6 = 법인 FE + 연도 + β1⁻·YoY⁻ + β1⁺·YoY⁺ + β3⁻·YoY⁻×노출 + β3⁺·YoY⁺×노출
  표본: 3-2 매칭 표본(1:3 가중), 법인·월 이중 클러스터(자유도 = 적은 쪽 클러스터 − 1)
  주 검정: β3⁻ (기대 부호 음수). Holm 보정은 9-1~9-4가 모두 끝난 뒤 한다
  보조(판정에 안 씀): β3⁺, 비대칭 검정 β3⁻ − β3⁺
  사후 보조(설계에 없음, 9/28 팀 판정표의 "달력 통제" 열을 채우려는 용도): 같은 식 + 영업일수차 + 영업일수차×노출
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, fe_reg, load_panel, lsdv_check, match_weights

sys.stdout.reconfigure(encoding="utf-8")

Y = "대출증감률_h6"
WORKDAYS = HERE.parents[1] / "외부데이터" / "workdays_2021_2025.csv"


def twoway_V(r):
    """법인·월 이중 클러스터 분산행렬 전체(Cameron-Gelbach-Miller)와 자유도"""
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
    return V, min(r["법인ID"][2], r["ym"][2]) - 1


def test(b, V, df, c):
    """선형결합 c'b = 0 검정: (추정치, 표준오차, p)"""
    est = c @ b
    se = np.sqrt(c @ V @ c)
    return est, se, 2 * stats.t.sf(abs(est / se), df)


df = load_panel()
df["YoY⁻"] = df["수출YoY"].clip(upper=0)
df["YoY⁺"] = df["수출YoY"].clip(lower=0)
df["YoY⁻×노출"] = df["YoY⁻"] * df["외환노출"]
df["YoY⁺×노출"] = df["YoY⁺"] * df["외환노출"]
wd = pd.read_csv(WORKDAYS, encoding="utf-8-sig")[["ym", "전년동월차"]].rename(columns={"전년동월차": "영업일수차"})
df = df.merge(wd, on="ym", how="left")
assert df["영업일수차"].notna().all()
df["영업일수×노출"] = df["영업일수차"] * df["외환노출"]

W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

# ── 추정 표본에서 하락·상승 달이 몇 개인가 ───────────────────────
est = df[df[Y].notna() & df["법인ID"].isin(W_MATCH.index)].drop_duplicates(["사업장_시도", "ym"])
print("=== 추정 표본(h=6)의 달 구성 ===")
for r, g in est.groupby("사업장_시도"):
    print(f"  {r}: 하락 {int((g['수출YoY'] < 0).sum())}개월 / 상승 {int((g['수출YoY'] >= 0).sum())}개월 (전체 {len(g)})")

X_MAIN = ["YoY⁻", "YoY⁺", "YoY⁻×노출", "YoY⁺×노출", "연도2024", "연도2025"]
X_CAL = X_MAIN + ["영업일수차", "영업일수×노출"]
I = {c: i for i, c in enumerate(X_CAL)}

out = {}
for name, xcols in [("주 모형(사전 설계)", X_MAIN), ("사후 보조: + 영업일수 2항", X_CAL)]:
    r = fe_reg(df, Y, W_MATCH, xcols)
    V, dof = twoway_V(r)
    b = r["b"]
    k = len(xcols)

    def c_of(**w):
        c = np.zeros(k)
        for key, val in w.items():
            c[I[key]] = val
        return c

    rows = []
    for label, c in [
        ("β1⁻ 비노출, 하락", c_of(**{"YoY⁻": 1})),
        ("β1⁻+β3⁻ 노출, 하락", c_of(**{"YoY⁻": 1, "YoY⁻×노출": 1})),
        ("β3⁻ 하락 국면 차이 [주 검정]", c_of(**{"YoY⁻×노출": 1})),
        ("β1⁺ 비노출, 상승", c_of(**{"YoY⁺": 1})),
        ("β1⁺+β3⁺ 노출, 상승", c_of(**{"YoY⁺": 1, "YoY⁺×노출": 1})),
        ("β3⁺ 상승 국면 차이 [보조]", c_of(**{"YoY⁺×노출": 1})),
        ("β3⁻ − β3⁺ 비대칭 [보조]", c_of(**{"YoY⁻×노출": 1, "YoY⁺×노출": -1})),
    ]:
        e_, se, p = test(b, V, dof, c)
        rows.append({"항목": label, "계수": e_, "SE(이중)": se, "p(이중)": p, "10%p 하락당 대출 변화(%)": -0.1 * e_ * 100})
    t = pd.DataFrame(rows)
    print(f"\n=== 9-1 {name} — 회사 {r['firms']}, 월 클러스터 {r['ym'][2]}, 자유도 {dof} ===")
    with pd.option_context("display.width", 200):
        print(t.round(4).to_string(index=False))
    out[name] = (r, xcols, t)

t = out["주 모형(사전 설계)"][2]
# 더미 회귀 이중 확인: lsdv_check는 xcols의 두 번째 계수(인덱스 1)를 돌려주므로 β3⁻가 인덱스 1이 되게 순서를 바꿔 푼다
alt = ["YoY⁻", "YoY⁻×노출", "YoY⁺", "YoY⁺×노출", "연도2024", "연도2025"]
r_alt = fe_reg(df, Y, W_MATCH, alt)
chk = lsdv_check(r_alt, Y, alt)
b3m = t.loc[t["항목"].str.startswith("β3⁻ 하락"), "계수"].iloc[0]
print(f"\n이중 확인(주 모형 β3⁻): 평균빼기 {b3m:.5f} / 더미회귀 {chk:.5f} / 차이 {abs(b3m - chk):.1e}")
