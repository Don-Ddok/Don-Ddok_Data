# -*- coding: utf-8 -*-
"""
8단계-7: 충격 이전 추세(사전 추세) — 수출이 꺾이기 전부터 노출·비노출 회사가 이미 다르게 움직였나
  설계는 08_달력통제_이후_결과.md 3절에 실행 전 확정.
  주 종속변수: ln(운전자금 t-1 +1) - ln(운전자금 t-7 +1)  (충격 달 t 이전 6개월, 주 결과 구간과 겹치지 않음)
  보조(판정 제외): t-4 -> t-1
  식·표본·가중은 3-3과 같고 종속변수만 다르다.
  판정: p(이중)>=0.10 이고 |β3pre|<0.1155(기준 -0.231의 절반) → 사전 추세 없음
        p<0.10 → 사전 추세 의심 / p>=0.10인데 크기 >=0.1155 → 판정 불가
입력: 패널(common.py), step3_psm_matched.parquet
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, XCOLS, fe_reg, load_panel, lsdv_check, match_weights, twoway_p

sys.stdout.reconfigure(encoding="utf-8")

BASE_B3 = -0.231
HALF = abs(BASE_B3) / 2

df = load_panel()
df["_m"] = (df["ym"] // 100) * 12 + (df["ym"] % 100 - 1)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))


def value_at(offset):
    """각 행 기준 offset개월 떨어진 달의 운전자금 잔액(관측 안 됐으면 빈칸)"""
    s = df[["법인ID", "_m", "여신_운전자금대출잔액"]].copy()
    s["_m"] -= offset
    return df[["법인ID", "_m"]].merge(s, on=["법인ID", "_m"], how="left")["여신_운전자금대출잔액"].to_numpy()


x_m1 = value_at(-1)
df["사전6"] = np.log(x_m1 + 1) - np.log(value_at(-7) + 1)
df["사전3"] = np.log(x_m1 + 1) - np.log(value_at(-4) + 1)
df["h6_같은표본"] = df["대출증감률_h6"].where(df["사전6"].notna())

n_ok = df["사전6"].notna().sum()
print(f"패널 {len(df):,}행 중 사전6 계산 가능 {n_ok:,}행, 그중 h6도 가능 {df['h6_같은표본'].notna().sum():,}행")
print(f"사전6 계산 가능한 달 범위: {df.loc[df['사전6'].notna(), 'ym'].min()} ~ {df.loc[df['사전6'].notna(), 'ym'].max()}")

models = [
    ("기준: 주 결과 h=6 (전체 표본)", "대출증감률_h6"),
    ("같은 표본에서 주 결과 h=6", "h6_같은표본"),
    ("주 검정: 사전 6개월 (t-7→t-1)", "사전6"),
    ("보조: 사전 3개월 (t-4→t-1)", "사전3"),
]
rows, keep = [], {}
for name, y in models:
    r = fe_reg(df, y, W_MATCH, XCOLS)
    se, p3 = twoway_p(r, 1)
    g_min = min(r["법인ID"][2], r["ym"][2])
    tcrit = stats.t.ppf(0.975, g_min - 1)
    b3 = r["b"][1]
    rows.append({"모형": name, "회사": r["firms"], "월": r["ym"][2], "β1(비노출)": r["b"][0],
                 "β1+β3(노출)": r["b"][0] + b3, "β3": b3, "SE(이중)": se, "p(이중)": p3,
                 "95%CI 하한": b3 - tcrit * se, "95%CI 상한": b3 + tcrit * se})
    keep[name] = (r, y)

res = pd.DataFrame(rows)
print("\n=== 8-7 결과 (매칭 1:3 가중, 법인·월 이중 클러스터) ===")
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(res.round(3).to_string(index=False))

main = res.iloc[2]
b, p = main["β3"], main["p(이중)"]
print("\n=== 판정 (사전 기준, 주 검정 = 사전 6개월) ===")
print(f"β3pre {b:+.3f}, p(이중) {p:.3f}, 기준 크기 {HALF:.4f}")
if p < 0.10:
    verdict = "사전 추세 의심"
elif abs(b) < HALF:
    verdict = "사전 추세 없음(통과)"
else:
    verdict = "판정 불가(추정이 불확실한데 크기는 큼)"
print(f"판정: {verdict}")

r, y = keep["주 검정: 사전 6개월 (t-7→t-1)"]
chk = lsdv_check(r, y, XCOLS)
print(f"\n이중 확인 (주 검정): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}")
