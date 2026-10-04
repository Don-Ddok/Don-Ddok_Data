# -*- coding: utf-8 -*-
"""그림: irf_inflow_outflow.png (입금·출금 + 요구불 잔액 C1 참고), irf_shock_validity.png (§3). 사후 분석(메커니즘)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = r"C:\test\outputs\mechanism_inflow"
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
r = pd.read_csv(os.path.join(OUT, "results.csv"))
bal = pd.read_csv(r"C:\test\outputs\harmonized\results_all.csv")
bal = bal[(bal["계정"] == "요구불") & (bal["모형"] == "C1")].sort_values("h")
cv = lambda b: (np.exp(-0.1 * b) - 1) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (a, col, title) in zip(axes, [("입금", "#2D5FC4", "요구불 입금 (판정: h=6)"), ("출금", "#D26FC2", "요구불 출금 (보고만)")]):
    d = r[r["분석"] == a].sort_values("h")
    d = pd.concat([d, pd.DataFrame([{"h": -1, "β3": 0.0, "CI_lo": 0.0, "CI_hi": 0.0}])]).sort_values("h")
    ax.fill_between(d["h"], cv(d["CI_hi"]), cv(d["CI_lo"]), color=col, alpha=0.15, linewidth=0)
    ax.plot(d["h"], cv(d["β3"]), color=col, lw=2, marker="o", ms=3.5, label=a)
    ax.plot(bal["h"], bal["환산"], color="#62625d", lw=1.2, ls="--", label="요구불 잔액 C1 (참고)")
    ax.axhline(0, color="#62625d", lw=0.8); ax.axvline(-1, color="#9d9d97", lw=0.8, ls=":")
    ax.set_title(title, fontsize=11); ax.set_xticks(range(-6, 13, 2)); ax.set_xlabel("h (개월, -1 = 기준)")
    ax.grid(axis="y", color="#efefeb"); ax.legend(fontsize=8, loc="lower left")
axes[0].set_ylabel("10%p 하락 시 노출-비노출 차이 (%)")
fig.suptitle("사후 분석(메커니즘) · 요구불 입금·출금, C1 사양 (법인 FE + 지역×연월 FE + 노출×영업일수차, 이중 군집 95% CI)", fontsize=10)
fig.text(0.5, 0.005, "주: h<0은 이전 - 나중이라 사후 효과와 같은 방향의 사전 추세는 반대 부호로 나온다. 입금·출금은 월 흐름 금액의 t+h월 vs t-1월 비교.", ha="center", fontsize=8, color="#62625d")
fig.tight_layout(rect=(0, 0.04, 1, 0.94)); fig.savefig(os.path.join(OUT, "irf_inflow_outflow.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.4))
for a, col, ls in [("수출점검_3개월합", "#00789A", "-"), ("수출점검_월(참고)", "#9d9d97", "--")]:
    d = r[r["분석"] == a].sort_values("h")
    if a.startswith("수출점검_3"):
        ax.fill_between(d["h"], cv(d["CI_hi"]), cv(d["CI_lo"]), color=col, alpha=0.15, linewidth=0)
    ax.plot(d["h"], cv(d["β3"]), color=col, lw=2 if ls == "-" else 1.3, ls=ls, marker="o" if ls == "-" else None, ms=3.5,
            label="3개월 합계 (주)" if ls == "-" else "월 단위 (참고)")
ax.axhline(0, color="#62625d", lw=0.8); ax.set_xticks(range(0, 13)); ax.set_xlabel("h (개월)")
ax.set_ylabel("지역 수출 YoY 10%p 하락 시\n노출 기업 수출 실적 변화 (%)"); ax.grid(axis="y", color="#efefeb"); ax.legend(fontsize=8)
ax.set_title("충격 변수 점검 · EX 노출 법인 525곳 (법인 FE + 달력월 더미 + 영업일수차, 이중 군집 95% CI)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "irf_shock_validity.png"), dpi=150); plt.close(fig)
print("saved")
