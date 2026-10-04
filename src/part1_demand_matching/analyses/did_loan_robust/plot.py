# -*- coding: utf-8 -*-
"""irf_plot.png — h=-6~12 β3 (L1, L2 나란히), 95% CI 음영, h=-1 기준선, 원본 -0.231 점선. 강건성(사후)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = r"C:\test\outputs\did_loan_robust"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
r = pd.read_csv(os.path.join(OUT, "results_table.csv"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (m, title) in zip(axes, [("L1", "L1 매칭 표본 (판정)"), ("L2", "L2 매칭 없음 (참고)")]):
    d = r[r["모형"] == m].sort_values("h")
    d = pd.concat([d, pd.DataFrame([{"h": -1, "β3": 0.0, "CI_lo": 0.0, "CI_hi": 0.0}])]).sort_values("h")
    ax.fill_between(d["h"], d["CI_lo"], d["CI_hi"], color="#2D5FC4", alpha=0.15, linewidth=0)
    ax.plot(d["h"], d["β3"], color="#2D5FC4", lw=2, marker="o", ms=4)
    ax.axhline(0, color="#62625d", lw=0.8)
    ax.axvline(-1, color="#9d9d97", lw=0.8, ls=":")
    ax.axhline(-0.231, color="#D26FC2", lw=1.2, ls="--")
    ax.text(12, -0.231, " 원본 -0.231 (h=6)", va="bottom", ha="right", fontsize=8, color="#62625d")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("h (개월, -1 = 기준)")
    ax.set_xticks(range(-6, 13, 2))
    ax.grid(axis="y", color="#efefeb")
axes[0].set_ylabel("β3 (수출 YoY 100%p당 로그 변화)")
fig.suptitle("강건성(사후) · 운전자금 대출 β3 (노출×보정YoY), 법인 FE + 지역×연월 FE + 노출×영업일수차, 이중 군집 95% CI", fontsize=10)
fig.text(0.5, 0.005, "주: h<0의 종속변수는 ln_{t+h} - ln_{t-1}(이전 - 나중). 사후 효과와 같은 방향의 사전 추세가 있으면 h<0 β3는 반대 부호(양수)로 나온다.",
         ha="center", fontsize=8, color="#62625d")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(os.path.join(OUT, "irf_plot.png"), dpi=150)
print("saved")
