# -*- coding: utf-8 -*-
"""
3단계-3: 매칭 표본 회귀 — 수출이 꺾일 때 노출 회사가 다르게 반응하는가
  식: 대출증감률_h(i,t) = 법인FE + 연도더미 + β1·수출YoY(r,t) + β3·수출YoY(r,t)×외환노출(i)
      대출증감률_h = ln 운전자금(t+h) − ln 운전자금(t−1),  h = 3, 6
      수출YoY = 회사 소재지(대구/경북) 월별 수출액 전년동월비 (소수, 0.1 = 10%)
  표본: [매칭 전] 운전자금 보유 회사 9,528개(가중치 없음)
        [매칭 후] 3-2 매칭 회사 (노출 가중치 1, 대조 = 뽑힌 횟수 ÷ 3)
  표준오차: 법인 클러스터 / 월 클러스터(충격이 36개 달뿐이라 — 열린 이슈 3)
  이중 확인: 핵심 계수를 희소행렬 더미 회귀(LSDV)로 다시 계산해 일치하는지
입력: step1_loan_industry_panel.parquet, step3_psm_matched.parquet, ../../외부데이터/export_region.csv
공통 함수: common.py
"""
import sys

import numpy as np
import pandas as pd

from common import EXPORT, HERE, fe_reg, load_panel, load_shock, lsdv_check, match_weights

sys.stdout.reconfigure(encoding="utf-8")

# ── ① 수출 데이터 점검: 파일의 전년동월비가 원자료로 재현되는지 ─────────
ex = pd.read_csv(EXPORT, encoding="utf-8-sig")
prev = ex[["region", "ym", "exp_amt"]].assign(ym=lambda d: d["ym"] + 100)
chk = ex.merge(prev, on=["region", "ym"], suffixes=("", "_작년"))
recalc = (chk["exp_amt"] / chk["exp_amt_작년"] - 1) * 100
gap = (recalc - chk["yoy_exp_amt"]).abs().max()
print("=== ① 수출 데이터 점검 ===")
print(f"전년동월비 재계산 vs 파일 값: 최대 차이 {gap:.3f}%p ({len(chk)}개월×지역)")
for r, g in load_shock().groupby("사업장_시도"):
    print(f"  {r}: 평균 {g['수출YoY'].mean()*100:+.1f}%, 마이너스인 달 {(g['수출YoY'] < 0).sum()}/36")

df = load_panel()
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

print("\n=== ② 회귀 결과 (β3 = 수출YoY × 외환노출) ===")
print("β는 수출YoY가 1(=100%p) 오를 때의 로그 변화. 아래 '10%p 하락 시'는 −0.1×β3를 %로 환산")
rows = []
for h in (3, 6):
    y = f"대출증감률_h{h}"
    for label, w in [("매칭 전", None), ("매칭 후", W_MATCH)]:
        r = fe_reg(df, y, w)
        b1, b3 = r["b"][0], r["b"][1]
        se_f, p_f = r["법인ID"][0][1], r["법인ID"][1][1]
        se_m, p_m, Gm = r["ym"][0][1], r["ym"][1][1], r["ym"][2]
        eff = (np.exp(-0.1 * b3) - 1) * 100
        rows.append(dict(h=h, 표본=label, 회사=r["firms"], 행=r["n"], β1=b1, β3=b3,
                         SE_법인=se_f, p_법인=p_f, SE_월=se_m, p_월=p_m,
                         **{"10%p 하락 시 노출회사 추가변화(%)": eff}))
        if h == 6 and label == "매칭 후":
            chk_b3 = lsdv_check(r, y)
            chk_line = f"이중 확인 (h=6 매칭 후): 평균빼기 β3={b3:.5f} / 더미회귀 β3={chk_b3:.5f} / 차이 {abs(b3-chk_b3):.1e}"
res = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))
print(f"\n{chk_line}")
print(f"월 클러스터 수 {Gm}개 (표준오차가 과소추정될 수 있는 개수 — 4단계 wild bootstrap 예정)")

# ── ③ 참고: 수출이 마이너스인 달/플러스인 달의 평균 대출증감률 ──
print("\n=== ③ 단순 비교 (회귀 아님, 감 잡기용) — 매칭 후 가중 평균 h=6 ===")
d = df[df["대출증감률_h6"].notna()].copy()
d["_w"] = d["법인ID"].map(W_MATCH)
d = d[d["_w"] > 0]
d["수출"] = np.where(d["수출YoY"] < 0, "수출 감소 달", "수출 증가 달")
t = (d.groupby(["수출", "외환노출"])
       .apply(lambda g: np.average(g["대출증감률_h6"], weights=g["_w"]) * 100, include_groups=False)
       .unstack().rename(columns={0: "비노출", 1: "노출"}))
t["차이(노출−비노출)"] = t["노출"] - t["비노출"]
print(t.round(2).to_string())
