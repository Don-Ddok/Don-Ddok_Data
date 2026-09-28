# -*- coding: utf-8 -*-
"""
8단계-6: 달력(영업일수) 통제 — β3가 수출 충격이 아니라 영업일수 효과는 아닌가
  (설계는 작업순서 8단계 "8-6"에 2026-09-28 사전 확정)
  영업일수차 = 이번 달 영업일수 − 작년 같은 달 영업일수(일, 전국 단일 시계열)
  주 모형:  3-3 기준(h=6, 매칭 1:3 가중, 법인 FE + 연도 더미) + 영업일수차 + 영업일수차×외환노출
  보조 모형: 4-3 이중강건 주 모형 + 영업일수 2항, h=3 기준·통제(참고, 주 결과 아님)
  판정: 8-1과 같음 — β3 부호 같고 크기가 기준(−0.231)의 절반 이상 → 영업일수로 설명되지 않음
입력: 패널(common.py), step3_psm_matched.parquet, step3_propensity.parquet,
      외부데이터/workdays_2021_2025.csv (holidays 라이브러리로 60개월 재계산해 일치 확인, 2026-09-28)
"""
import sys

import pandas as pd

from common import (HERE, XCOLS, add_cov_interactions, fe_reg, load_panel, lsdv_check,
                    match_weights, twoway_p)

sys.stdout.reconfigure(encoding="utf-8")

BASE_B3 = -0.231
WORKDAYS = HERE.parents[1] / "외부데이터" / "workdays_2021_2025.csv"


def add_workdays(df):
    """영업일수 전년동월차(일)와 × 외환노출 교호항 추가"""
    wd = pd.read_csv(WORKDAYS, encoding="utf-8-sig")[["ym", "전년동월차"]]
    df = df.merge(wd.rename(columns={"전년동월차": "영업일수차"}), on="ym", how="left")
    assert df["영업일수차"].notna().all(), "영업일수가 빠진 달이 있음"
    df["영업일수×노출"] = df["영업일수차"] * df["외환노출"]
    return df


df = load_panel()
df = add_workdays(df)
ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
df, INT_COV = add_cov_interactions(df, ps)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

# ── 참고: 수출 충격과 영업일수가 얼마나 같이 움직이나 ─────────────
print("=== 수출YoY vs 영업일수차 상관 (2023~2025 월별) ===")
m = df.drop_duplicates(["사업장_시도", "ym"])
for r, g in m.groupby("사업장_시도"):
    print(f"  {r}: {g['수출YoY'].corr(g['영업일수차']):+.2f}  (달 {len(g)}개)")

CAL = ["영업일수차", "영업일수×노출"]
models = [
    ("h=6 기준(3-3)", "대출증감률_h6", XCOLS),
    ("h=6 주 모형: + 영업일수 2항", "대출증감률_h6", XCOLS + CAL),
    ("h=6 보조: 이중강건 + 영업일수 2항", "대출증감률_h6", XCOLS + INT_COV + CAL),
    ("h=3 기준(참고)", "대출증감률_h3", XCOLS),
    ("h=3 + 영업일수 2항(참고)", "대출증감률_h3", XCOLS + CAL),
]
rows, keep = [], {}
for name, y, xcols in models:
    r = fe_reg(df, y, W_MATCH, xcols)
    _, p3 = twoway_p(r, 1)
    row = {"모형": name, "회사": r["firms"], "월": r["ym"][2], "β1(비노출)": r["b"][0],
           "β1+β3(노출)": r["b"][0] + r["b"][1], "β3": r["b"][1], "p(이중)": p3}
    if "영업일수×노출" in xcols:
        j = xcols.index("영업일수×노출")
        row["영업일수×노출"] = r["b"][j]
        row["p 영업일수×노출"] = twoway_p(r, j)[1]
        j0 = xcols.index("영업일수차")
        row["영업일수차"] = r["b"][j0]
        row["p 영업일수차"] = twoway_p(r, j0)[1]
    rows.append(row)
    keep[name] = (r, y, xcols)

res = pd.DataFrame(rows)
print("\n=== 8-6 결과 (매칭 1:3 가중, 이중 클러스터) ===")
with pd.option_context("display.width", 260, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))

print("\n=== 판정 (사전 기준: 부호 같고 |β3| ≥ 기준(−0.231)의 절반) — h=6 모형만 ===")
for _, row in res.iloc[1:3].iterrows():
    ok = (row["β3"] < 0) and (abs(row["β3"]) >= abs(BASE_B3) / 2)
    print(f"{row['모형']}: β3 {row['β3']:+.3f} (기준 대비 {row['β3'] / BASE_B3 * 100:.0f}%), p(이중) {row['p(이중)']:.3f} → "
          f"{'영업일수로 설명되지 않음' if ok else '영업일수와 분리되지 않음'}")

r, y, xcols = keep["h=6 주 모형: + 영업일수 2항"]
chk = lsdv_check(r, y, xcols)
print(f"\n이중 확인 (h=6 주 모형): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}")
