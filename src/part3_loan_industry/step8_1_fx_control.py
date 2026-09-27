# -*- coding: utf-8 -*-
"""
8단계-1: 환율 통제 — β3가 수출 충격이 아니라 환율 반응은 아닌가 (설계는 작업순서 8단계에 사전 확정)
  환율YoY = 원/달러 월평균 전년동월비 (한국은행 ECOS 원자료, 전국 단일 시계열)
  주 모형:  3-3 기준(h=6, 매칭 1:3 가중) + 환율YoY + 환율YoY×외환노출
  보조 모형: 4-3 이중강건 주 모형 + 환율 2항
  판정: β3 부호 같고 크기가 기준(−0.231)의 절반 이상 → 환율로 설명되지 않음
입력: 패널(common.py), step3_psm_matched.parquet, step3_propensity.parquet, 외부데이터/환율_ECOS원자료_202101_202512.csv
"""
import sys

import pandas as pd

from common import (HERE, XCOLS, add_cov_interactions, add_fx, fe_reg, load_panel, lsdv_check,
                    match_weights, twoway_p)

sys.stdout.reconfigure(encoding="utf-8")

Y = "대출증감률_h6"
BASE_B3 = -0.231

df = load_panel()
df = add_fx(df)
ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
df, INT_COV = add_cov_interactions(df, ps)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

# ── 참고: 수출 충격과 환율이 얼마나 같이 움직이나 ─────────────────
print("=== 수출YoY vs 환율YoY 상관 (2023~2025 월별) ===")
m = df.drop_duplicates(["사업장_시도", "ym"])
for r, g in m.groupby("사업장_시도"):
    print(f"  {r}: {g['수출YoY'].corr(g['환율YoY']):+.2f}")

FX = ["환율YoY", "환율×노출"]
models = [
    ("기준(3-3)", XCOLS),
    ("주 모형: + 환율 2항", XCOLS + FX),
    ("보조: 이중강건 + 환율 2항", XCOLS + INT_COV + FX),
]
rows, keep = [], {}
for name, xcols in models:
    r = fe_reg(df, Y, W_MATCH, xcols)
    _, p3 = twoway_p(r, 1)
    row = {"모형": name, "회사": r["firms"], "β1(비노출)": r["b"][0], "β1+β3(노출)": r["b"][0] + r["b"][1],
           "β3": r["b"][1], "p(이중)": p3}
    if "환율×노출" in xcols:
        j = xcols.index("환율×노출")
        row["환율×노출"] = r["b"][j]
        row["p 환율×노출"] = twoway_p(r, j)[1]
    rows.append(row)
    keep[name] = (r, xcols)

res = pd.DataFrame(rows)
print("\n=== 8-1 결과 (h=6, 매칭 1:3 가중) ===")
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(res.round(3).to_string(index=False))

print("\n=== 판정 (사전 기준: 부호 같고 |β3| ≥ 기준의 절반) ===")
for _, row in res.iloc[1:].iterrows():
    keep_ok = (row["β3"] < 0) and (abs(row["β3"]) >= abs(BASE_B3) / 2)
    print(f"{row['모형']}: β3 {row['β3']:+.3f} (기준 대비 {row['β3'] / BASE_B3 * 100:.0f}%) → "
          f"{'환율로 설명되지 않음' if keep_ok else '환율과 분리되지 않음'}")

r, xcols = keep["주 모형: + 환율 2항"]
chk = lsdv_check(r, Y, xcols)
print(f"\n이중 확인 (주 모형): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}")
