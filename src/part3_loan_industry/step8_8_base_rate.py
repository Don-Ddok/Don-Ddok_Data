# -*- coding: utf-8 -*-
"""
8단계-8: 기준금리 통제 — β3가 수출 충격이 아니라 금리 반응은 아닌가 (설계는 08_달력통제_이후_결과.md 4절에 실행 전 확정)
  금리변화 = 기준금리(t+6) - 기준금리(t-1), %p  (종속변수와 같은 기간, 한국은행 ECOS 722Y001 원자료, 전국 단일 시계열)
  주 모형:  3-3 기준(h=6, 매칭 1:3 가중) + 금리변화 + 금리변화×외환노출
  보조 모형: 4-3 이중강건 주 모형 + 금리 2항
  판정: β3 부호 같고 크기가 기준(-0.231)의 절반 이상 → 기준금리로 설명되지 않음
입력: 패널(common.py), step3_psm_matched.parquet, step3_propensity.parquet, 외부데이터/기준금리_ECOS원자료_202101_202512.csv
"""
import sys

import pandas as pd

from common import (HERE, XCOLS, add_cov_interactions, fe_reg, load_panel, lsdv_check, match_weights,
                    twoway_p)

sys.stdout.reconfigure(encoding="utf-8")

Y = "대출증감률_h6"
BASE_B3 = -0.231
RATE = HERE.parents[1] / "외부데이터" / "기준금리_ECOS원자료_202101_202512.csv"

df = load_panel()
rate = pd.read_csv(RATE, encoding="utf-8-sig")
assert len(rate) == 60 and rate["ym"].is_unique
rate["_m"] = (rate["ym"] // 100) * 12 + (rate["ym"] % 100 - 1)
r_at = rate.set_index("_m")["기준금리"]

df["_m"] = (df["ym"] // 100) * 12 + (df["ym"] % 100 - 1)
df["금리변화"] = (df["_m"] + 6).map(r_at) - (df["_m"] - 1).map(r_at)
assert df.loc[df[Y].notna(), "금리변화"].notna().all(), "종속변수가 있는 행에 금리변화가 비어 있음"
df["금리×노출"] = df["금리변화"] * df["외환노출"]

ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
df, INT_COV = add_cov_interactions(df, ps)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

# ── 참고: 금리변화 분포와 수출 충격과의 관계 ─────────────────────
mm = df[df[Y].notna()].drop_duplicates(["사업장_시도", "ym"])
ch = mm.drop_duplicates("ym").set_index("ym")["금리변화"].sort_index()
print("=== 금리변화(t+6 − t−1, %p) — 종속변수가 계산되는 달 ===")
print(f"달 수 {len(ch)}, 0이 아닌 달 {int((ch != 0).sum())}개, 분포: " +
      ", ".join(f"{k:+.2f}%p {v}개월" for k, v in ch.value_counts().sort_index().items()))
print(f"0이 아닌 달 범위: {ch[ch != 0].index.min()} ~ {ch[ch != 0].index.max()}")
print("\n=== 수출YoY vs 금리변화 상관 (종속변수가 있는 달) ===")
for r, g in mm.groupby("사업장_시도"):
    print(f"  {r}: {g['수출YoY'].corr(g['금리변화']):+.2f} (월 {len(g)}개)")

FX = ["금리변화", "금리×노출"]
models = [
    ("기준(3-3)", XCOLS),
    ("주 모형: + 금리 2항", XCOLS + FX),
    ("보조: 이중강건 + 금리 2항", XCOLS + INT_COV + FX),
]
rows, keep = [], {}
for name, xcols in models:
    r = fe_reg(df, Y, W_MATCH, xcols)
    _, p3 = twoway_p(r, 1)
    row = {"모형": name, "회사": r["firms"], "월": r["ym"][2], "β1(비노출)": r["b"][0],
           "β1+β3(노출)": r["b"][0] + r["b"][1], "β3": r["b"][1], "p(이중)": p3}
    if "금리×노출" in xcols:
        j = xcols.index("금리변화")
        k = xcols.index("금리×노출")
        row["금리변화"] = r["b"][j]
        row["p 금리"] = twoway_p(r, j)[1]
        row["금리×노출"] = r["b"][k]
        row["p 금리×노출"] = twoway_p(r, k)[1]
    rows.append(row)
    keep[name] = (r, xcols)

res = pd.DataFrame(rows)
print("\n=== 8-8 결과 (h=6, 매칭 1:3 가중) ===")
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))

print("\n=== 판정 (사전 기준: 부호 같고 |β3| ≥ 기준의 절반) ===")
for _, row in res.iloc[1:].iterrows():
    keep_ok = (row["β3"] < 0) and (abs(row["β3"]) >= abs(BASE_B3) / 2)
    print(f"{row['모형']}: β3 {row['β3']:+.3f} (기준 대비 {row['β3'] / BASE_B3 * 100:.0f}%) → "
          f"{'기준금리로 설명되지 않음' if keep_ok else '기준금리와 분리되지 않음'}")

r, xcols = keep["주 모형: + 금리 2항"]
chk = lsdv_check(r, Y, xcols)
print(f"\n이중 확인 (주 모형): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}")
