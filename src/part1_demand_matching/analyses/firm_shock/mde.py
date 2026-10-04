# -*- coding: utf-8 -*-
"""MDE (SPEC §2, 추정치보다 먼저): results.csv에서 SE·자유도 관련 열만 읽어 계정별 h=6 MDE
MDE = (t.975 + t.80) × SE, 자유도 G_min − 1 (G_min = min(법인, 월))
보고 단위: (a') 외환 실적 10% 감소(S = ln 0.9) → 잔액 (exp(MDE·ln 0.9)−1)×100 %, HOLD |MDE·ln 0.9|×100 %p
           (b) 급감 = 1 → 잔액 (exp(MDE)−1)×100 %, HOLD MDE×100 %p
출력: mde.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\test\outputs\firm_shock"
r = pd.read_csv(os.path.join(OUT, "results.csv"), usecols=["계정", "충격", "h", "SE", "N", "법인", "월"])   # β·p는 읽지 않음
r = r[r["h"] == 6].copy()
r["자유도"] = np.minimum(r["법인"], r["월"]) - 1
r["배수"] = stats.t.ppf(0.975, r["자유도"]) + stats.t.ppf(0.80, r["자유도"])
r["MDE_β"] = r["배수"] * r["SE"]
hold = r["계정"].str.endswith("HOLD")
main = r["충격"] == "주_a'"
L = np.log(0.9)
r["MDE_환산"] = np.where(main, np.where(hold, np.abs(r["MDE_β"] * L) * 100, (1 - np.exp(-r["MDE_β"] * abs(L))) * 100),
                        np.where(hold, r["MDE_β"] * 100, (np.exp(r["MDE_β"]) - 1) * 100))
r["단위"] = np.where(hold, "%p", "%")
r["환산_기준"] = np.where(main, "외환 실적 10% 감소 시 |변화|", "급감(=1) 시 |변화|")
r = r[["계정", "충격", "h", "N", "법인", "월", "자유도", "SE", "배수", "MDE_β", "MDE_환산", "단위", "환산_기준"]]
r.to_csv(os.path.join(OUT, "mde.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(r.round(4).to_string(index=False))
