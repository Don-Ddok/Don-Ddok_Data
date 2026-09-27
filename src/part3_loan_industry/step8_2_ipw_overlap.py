# -*- coding: utf-8 -*-
"""
8단계-2: 매칭 없이도 재현되는가 (설계는 작업순서 8단계에 사전 확정)
  배경: 매칭 표본 β3=−0.231, 매칭 없이 전체 표본 회귀 보정만 하면 β3=−0.149(p=0.25)
  (가) 겹침 구간 + 회귀 보정: 성향점수가 [노출 회사 1% 분위수, 비노출 최댓값] 안인 회사만,
       가중치 없음, 4-3 보조②와 같은 식(기준값 × 수출YoY 교호항 5개)
  (나) 역확률가중(IPW, 노출 회사 기준 효과): 노출 1, 비노출 ps/(1−ps), 전체 표본, 3-3 식
       가중 후 기준값 SMD, 최대 가중치, 유효 표본 크기 보고
  판정: (가)(나) 모두 부호 같고 |β3| ≥ 매칭(0.231)의 절반 → 다른 방법도 매칭 결과를 지지
        둘 다 절반 미만 → 매칭 표본에 의존 / 엇갈리면 그대로 보고
입력: 패널(common.py), step3_propensity.parquet
"""
import sys

import numpy as np
import pandas as pd

from common import COV, HERE, XCOLS, add_cov_interactions, fe_reg, load_panel, lsdv_check, twoway_p

sys.stdout.reconfigure(encoding="utf-8")

Y = "대출증감률_h6"
MATCH_B3 = -0.231

df = load_panel()
ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
df, INT_COV = add_cov_interactions(df, ps)
T = ps["외환노출"] == 1
p = ps["성향점수"]


def smd_w(w):
    out = {}
    for c in COV:
        x = ps[c]
        sd = np.sqrt((x[T].var() + x[~T].var()) / 2)
        out[c] = (np.average(x[T], weights=w[T]) - np.average(x[~T], weights=w[~T])) / sd
    return pd.Series(out)


# ── (가) 겹침 구간 ────────────────────────────────────────────────
lo, hi = p[T].quantile(0.01), p[~T].max()
inside = p.between(lo, hi)
print("=== (가) 겹침 구간 ===")
print(f"구간 [{lo:.4f}, {hi:.4f}] — 노출 {int((T & inside).sum())}/{int(T.sum())}, "
      f"비노출 {int((~T & inside).sum())}/{int((~T).sum())} 남음")
print(f"비노출 성향점수 분포: 중앙값 {p[~T].median():.4f}, 노출 1% 분위수보다 낮은 비노출 {int((~T & (p < lo)).sum())}곳 제외")
w_a = pd.Series(1.0, index=ps.index[inside])
print("겹침 구간 안 기준값 SMD (가중치 없음):", smd_w(pd.Series(np.where(inside, 1.0, 0.0), index=ps.index)).round(3).to_dict())

# ── (나) IPW ─────────────────────────────────────────────────────
w_b = pd.Series(np.where(T, 1.0, p / (1 - p)), index=ps.index)
wc = w_b[~T]
print("\n=== (나) IPW ===")
print(f"비노출 가중치: 합 {wc.sum():.1f} (노출 {int(T.sum())}곳), 최대 {wc.max():.2f}, "
      f"유효 표본 크기 {wc.sum() ** 2 / (wc ** 2).sum():.0f}곳 / {len(wc)}곳")
top = wc.sort_values(ascending=False)
print(f"가중치 상위 10곳이 비노출 가중치 합에서 차지하는 비율 {top.iloc[:10].sum() / wc.sum() * 100:.1f}%")
print("가중 후 기준값 SMD:", smd_w(w_b).round(3).to_dict())
print("가중 전 기준값 SMD:", smd_w(pd.Series(1.0, index=ps.index)).round(3).to_dict())

# ── 회귀 ────────────────────────────────────────────────────────
rows = []
specs = [
    ("참고: 매칭 전 + 회귀 보정(4-3 보조②)", None, XCOLS + INT_COV),
    ("(가) 겹침 구간 + 회귀 보정", w_a, XCOLS + INT_COV),
    ("(나) IPW", w_b, XCOLS),
]
for name, w, xcols in specs:
    r = fe_reg(df, Y, w, xcols)
    _, p3 = twoway_p(r, 1)
    rows.append({"방법": name, "회사": r["firms"], "β1(비노출)": r["b"][0], "β1+β3(노출)": r["b"][0] + r["b"][1],
                 "β3": r["b"][1], "p(이중)": p3, "매칭 대비": r["b"][1] / MATCH_B3 * 100})
    if name.startswith("(나)"):
        chk = lsdv_check(r, Y, xcols)
        chk_line = f"이중 확인 (IPW): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1] - chk):.1e}"

res = pd.DataFrame(rows)
print("\n=== 8-2 결과 (h=6, 매칭 β3 = −0.231 기준) ===")
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(res.round(3).to_string(index=False))
print(chk_line)

ok = [(row["β3"] < 0) and (abs(row["β3"]) >= abs(MATCH_B3) / 2) for _, row in res.iloc[1:].iterrows()]
verdict = ("다른 방법도 매칭 결과를 지지" if all(ok) else
           "매칭 표본에 의존하는 결과" if not any(ok) else "방법에 따라 엇갈림")
print(f"\n판정 (사전 기준): (가) {'충족' if ok[0] else '미충족'}, (나) {'충족' if ok[1] else '미충족'} → {verdict}")
