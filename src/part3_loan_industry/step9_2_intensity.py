# -*- coding: utf-8 -*-
"""
9단계-2: 노출 강도(연속형) — 더 자주 수출입하는 회사일수록 더 크게 반응하나 (용량-반응)
  (설계는 06_추가분석_설계와_결과.md 1절, 2026-09-23 실행 전 확정)
  노출 강도 = 회사가 관측된 달 중 외환 수출 또는 수입 실적이 있는 달의 비율(0~1). 비노출 회사는 0
             (금액이 아니라 빈도라 단위 문제·결과 변수 오염 없음)
  식: 3-3 식(대출증감률_h6 = FE + 연도 + β1·수출YoY + β3·수출YoY×외환노출) + γ·수출YoY×c_강도
     (c_강도 = 강도 − 노출 회사 평균, 운전자금 보유 회사 전체 기준으로 중심화)
     → β3는 "평균 강도 노출 회사"의 반응, γ는 강도가 평균보다 높을수록 추가되는 반응
  표본: 3-2 매칭 표본(1:3 가중), 재매칭 없음
  보조: 4-3 이중강건 교호항 5개 추가 버전
  주 검정: γ (Holm 보정은 9-1~9-4가 모두 끝난 뒤 한다)
  미리 적어 둔 한계: 강도는 36개월 전체로 계산해 충격 이후 활동도 반영된다(섞임 가능).
    회사 단위 강도라 기준 문서 H3(업종 수출지향도)와는 다른 양이다
입력: 패널(common.py), step1_loan_industry_panel.parquet(원본, 강도 계산용), step3_psm_matched.parquet, step3_propensity.parquet
"""
import sys

import pandas as pd

from common import HERE, XCOLS, add_cov_interactions, fe_reg, load_panel, lsdv_check, match_weights, twoway_p

sys.stdout.reconfigure(encoding="utf-8")

BASE_B3 = -0.231

# ── 노출 강도: 원본(필터 전) 패널로 계산 — 회사가 실제 관측된 모든 달 기준 ──
raw = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
raw["당월외환"] = (raw["외환_수출실적금액"] > 0) | (raw["외환_수입실적금액"] > 0)
intensity = raw.groupby("법인ID")["당월외환"].mean().rename("강도")
print(f"강도 계산: 회사 {len(intensity):,}개, 관측 달 수 평균 {raw.groupby('법인ID').size().mean():.1f}개월")
print(f"강도>0인 회사 {(intensity > 0).sum():,}개 (외환노출 회사 수와 대략 같아야 함)")

df = load_panel()
df = df.join(intensity, on="법인ID")
assert df["강도"].notna().all()

# 중심화: 운전자금 보유 회사(추정 모집단) 중 노출 회사의 강도 평균
firm_int = df.drop_duplicates("법인ID").set_index("법인ID")[["외환노출", "강도"]]
center = firm_int.loc[firm_int["외환노출"] == 1, "강도"].mean()
print(f"노출 회사 강도 평균(중심값): {center:.3f} (0=한 번, 1=매달 외환 거래)")
df["c_강도"] = df["강도"] - center
df["강도교호"] = df["수출YoY"] * df["c_강도"]

ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
df, INT_COV = add_cov_interactions(df, ps)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

Y = "대출증감률_h6"
models = [
    ("기준(3-3)", XCOLS),
    ("주 모형: + 강도교호", XCOLS + ["강도교호"]),
    ("보조: 이중강건 + 강도교호", XCOLS + INT_COV + ["강도교호"]),
]
rows, keep = [], {}
for name, xcols in models:
    r = fe_reg(df, Y, W_MATCH, xcols)
    _, p3 = twoway_p(r, 1)
    row = {"모형": name, "회사": r["firms"], "β1(비노출)": r["b"][0], "β1+β3(평균강도 노출)": r["b"][0] + r["b"][1],
           "β3": r["b"][1], "p(β3, 이중)": p3}
    if "강도교호" in xcols:
        j = xcols.index("강도교호")
        row["γ(강도)"] = r["b"][j]
        row["p(γ, 이중)"] = twoway_p(r, j)[1]
    rows.append(row)
    keep[name] = (r, xcols)

res = pd.DataFrame(rows)
print("\n=== 9-2 결과 (h=6, 매칭 1:3 가중) ===")
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))

main = res.iloc[1]
print(f"\n주 검정 γ = {main['γ(강도)']:+.4f}, p(이중, 보정 전) = {main['p(γ, 이중)']:.4f}")
print("해석: γ>0이면 강도가 평균보다 높은 노출 회사일수록 β3(음수) 효과가 옅어진다(반응이 작아짐).")
print("      γ<0이면 강도가 높을수록 효과가 더 커진다(용량-반응 가설과 같은 방향).")

r, xcols = keep["주 모형: + 강도교호"]
chk = lsdv_check(r, Y, xcols)
print(f"\n이중 확인 (주 모형 β3): 평균빼기 {r['b'][1]:.5f} / 더미회귀 {chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}")
print("Holm 보정 판정은 9-1~9-4를 모두 돌린 뒤 별도로 한다.")
