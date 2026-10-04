# -*- coding: utf-8 -*-
"""최소검출효과(MDE) — 새 추정 없음, results_all.csv에 저장된 h=6 β3·SE만 사용
공식: 파트3 step8_3_mde.py와 같음
  dof = G_min − 1 (G_min = min(노출+비노출 법인 수, 월 수)), MDE(β) = (t_0.975 + t_0.80) × SE
  검정력(참값 b) = nct.sf(t_c, dof, |b|/SE) + nct.cdf(−t_c, dof, |b|/SE)
주 지표: (a) MDE, (b) 원래 결과 크기에서의 검정력 (요구불 +0.362: C1·E23 / 대출 −0.231: C1·C2)
참고: 추정치 기준 검정력(사후 검정력)은 p값과 같은 정보라 해석에 쓰지 않음
% 환산: 감소 방향 exp(−0.1·MDE) − 1, 증가 방향 exp(+0.1·MDE) − 1 (주 값 = 원래 주장 방향, 원래 결과가 없으면 추정치 방향)
HOLD: %p = 0.1 × MDE × 100 (대칭)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\test\outputs\harmonized"
ORIG_B = {"요구불": 0.362, "운전자금": -0.231}          # SPEC §7 원래 결과 (100%p당)
TARGETS = [("요구불", "C1"), ("요구불", "E23"), ("운전자금", "C1"), ("운전자금", "C2"),
           ("거치식", "C1"), ("적립식", "C1"), ("거치식", "HOLD"), ("적립식", "HOLD")]


def power(b, se, dof):
    tc = stats.t.ppf(0.975, dof); ncp = abs(b) / se
    return stats.nct.sf(tc, dof, ncp) + stats.nct.cdf(-tc, dof, ncp)


r = pd.read_csv(os.path.join(OUT, "results_all.csv"))
rows = []
for a, m in TARGETS:
    x = r[(r["계정"] == a) & (r["모형"] == m) & (r["h"] == 6)].iloc[0]
    G = int(min(x["노출법인"] + x["비노출법인"], x["월"])); dof = G - 1
    mde = (stats.t.ppf(0.975, dof) + stats.t.ppf(0.80, dof)) * x["SE"]
    hold = m == "HOLD"
    row = {"계정": a, "모형": m, "β3": x["β3"], "SE": x["SE"], "G_min": G, "dof": dof, "MDE(β)": mde}
    if hold:
        row.update({"MDE_감소방향": -10 * mde, "MDE_증가방향": 10 * mde, "단위": "%p"})
        main_dir = "증가" if x["β3"] < 0 else "감소"
    else:
        row.update({"MDE_감소방향": (np.exp(-0.1 * mde) - 1) * 100, "MDE_증가방향": (np.exp(0.1 * mde) - 1) * 100, "단위": "%"})
        main_dir = {"요구불": "감소", "운전자금": "증가"}.get(a, "증가" if x["β3"] < 0 else "감소")
    row["주 방향"] = main_dir + ("(원래 주장)" if a in ORIG_B else "(추정치)")
    row["MDE 주 값"] = row["MDE_감소방향"] if main_dir == "감소" else row["MDE_증가방향"]
    row["추정치(환산)"] = x["환산"]
    if a in ORIG_B and m in ("C1", "E23", "C2"):
        bo = ORIG_B[a]
        row["원래 크기 β"] = bo
        row["원래 크기 검정력"] = power(bo, x["SE"], dof)
        row["원래 크기 ≥ MDE"] = abs(bo) >= mde
    row["사후 검정력(참고)"] = power(x["β3"], x["SE"], dof)
    rows.append(row)
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "mde.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
