# -*- coding: utf-8 -*-
"""
8단계-3: 최소검출효과(MDE) — "효과가 없다"가 아니라 "이 표본으로 확정할 수 없다"를 수치로
  MDE = (t_0.975 + t_0.80) × SE,  유의수준 5%(양측), 검정력 80%
  자유도 = 적은 쪽 클러스터 수 − 1, SE = 법인·월 이중 클러스터
  함께 보고: 참값이 추정치와 같을 때 5% 수준에서 유의하게 나올 확률(검정력, 비중심 t)
  표현: 수출 10%p 하락 시 노출 회사 대출의 추가 변화(%) = exp(0.1×|β|) − 1
입력: 패널(common.py), step3_psm_matched.parquet, step3_propensity.parquet, step5_2_item_results.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import HERE, XCOLS, add_cov_interactions, fe_reg, load_panel, match_weights, twoway_p

sys.stdout.reconfigure(encoding="utf-8")


def mde_row(name, b3, se, G):
    dof = G - 1
    tc = stats.t.ppf(0.975, dof)
    mde = (tc + stats.t.ppf(0.80, dof)) * se
    ncp = abs(b3) / se
    power = stats.nct.sf(tc, dof, ncp) + stats.nct.cdf(-tc, dof, ncp)
    pct = lambda v: (np.exp(0.1 * abs(v)) - 1) * 100
    return {"대상": name, "β3": b3, "SE(이중)": se, "자유도": dof, "MDE(β)": mde,
            "추정 효과(10%p당 %)": pct(b3), "MDE(10%p당 %)": pct(mde), "MDE÷추정": mde / abs(b3),
            "추정치 크기에서 검정력(%)": power * 100}


df = load_panel()
ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
df, INT_COV = add_cov_interactions(df, ps)
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))
T = ps["외환노출"] == 1
W_IPW = pd.Series(np.where(T, 1.0, ps["성향점수"] / (1 - ps["성향점수"])), index=ps.index)

rows = []
for name, y, w, xcols in [
    ("주 결과: h=6 매칭", "대출증감률_h6", W_MATCH, XCOLS),
    ("h=6 이중강건", "대출증감률_h6", W_MATCH, XCOLS + INT_COV),
    ("h=6 IPW", "대출증감률_h6", W_IPW, XCOLS),
    ("h=3 매칭", "대출증감률_h3", W_MATCH, XCOLS),
]:
    r = fe_reg(df, y, w, xcols)
    se, _ = twoway_p(r, 1)
    rows.append(mde_row(name, r["b"][1], se, min(r["법인ID"][2], r["ym"][2])))

main = pd.DataFrame(rows)
print("=== 8-3 최소검출효과: 운전자금 합계 ===")
with pd.option_context("display.width", 240, "display.max_columns", 20):
    print(main.round(3).to_string(index=False))

items = pd.read_csv(HERE / "step5_2_item_results.csv", encoding="utf-8-sig")
it = pd.DataFrame([mde_row(f"{a} {b.split('(')[0]}", s, e, g) for a, b, s, e, g in
                   items[["항목", "구간", "β3", "SE(이중)", "클러스터(적은 쪽)"]].itertuples(index=False)])
print("\n=== 8-3 최소검출효과: 세부 항목 20개 (5-2) ===")
with pd.option_context("display.width", 240, "display.max_columns", 20):
    print(it[["대상", "β3", "MDE(β)", "MDE(10%p당 %)", "MDE÷추정", "추정치 크기에서 검정력(%)"]].round(2).to_string(index=False))
print(f"\n세부 항목 MDE(10%p당 %): 중앙값 {it['MDE(10%p당 %)'].median():.1f}%, 범위 "
      f"{it['MDE(10%p당 %)'].min():.1f}~{it['MDE(10%p당 %)'].max():.1f}% "
      f"(주 결과 추정 효과 {main.loc[0, '추정 효과(10%p당 %)']:.1f}%)")
