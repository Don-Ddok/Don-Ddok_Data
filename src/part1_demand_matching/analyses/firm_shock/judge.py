# -*- coding: utf-8 -*-
"""판정 (SPEC §3): 주 충격 (a') h=6, 계정 6개 Holm / LP 결합 h=−6..−3 / 원본 방식 ln_{t−3}−ln_{t−9}
확정: Holm p<0.05 & LP p(χ²)≥0.10 & 연장 표시 없음 / 약한 증거: h=6 p<0.10 / 근거 없음: 그 외
연장 표시: 원본 β_pre가 h=6 β와 같은 부호이고 |β_pre| ≥ |β|/2
역인과 의심: LP p(χ²)<0.10 또는 원본 p<0.10 / 모든 줄에 "인과 아님"
반대 부호 유의: h=6 p<0.10이고 부호가 같은 계정 harmonized h=6(C1, HOLD는 HOLD)과 반대
강건성 (b): 주 사양과 같은 방향 / 반대 / 0 근처(p≥0.10) — 판정을 바꾸지 않음
출력: judgment.csv, robust_b.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import holm

OUT = r"C:\test\outputs\firm_shock"
HARM = r"C:\test\outputs\harmonized\results_all.csv"
L = np.log(0.9)
r = pd.read_csv(os.path.join(OUT, "results.csv"))
pre = pd.read_csv(os.path.join(OUT, "pretrend.csv"))
mde = pd.read_csv(os.path.join(OUT, "mde.csv"))
hz = pd.read_csv(HARM)
hz = hz[(hz["h"] == 6) & hz["업종"].isna() & hz["모형"].isin(["C1", "HOLD"])]
hsign = {(a + ("_HOLD" if m == "HOLD" else "")): np.sign(b) for a, m, b in zip(hz["계정"], hz["모형"], hz["β3"])}


def conv(beta, acct, shock):
    hold = acct.endswith("HOLD")
    if shock == "주_a'":
        return beta * L * 100 if hold else (np.exp(beta * L) - 1) * 100
    return beta * 100 if hold else (np.exp(beta) - 1) * 100


rows = []
for shock in ["주_a'", "강건_b"]:
    g = r[(r["h"] == 6) & (r["충격"] == shock)].copy()
    g["p_holm6"] = holm(g["p"].to_numpy())
    for _, x in g.iterrows():
        p = pre[(pre["계정"] == x["계정"]) & (pre["충격"] == shock)].iloc[0]
        m2 = r[(r["계정"] == x["계정"]) & (r["충격"] == shock) & (r["h"] == -2)]
        md = mde[(mde["계정"] == x["계정"]) & (mde["충격"] == shock)].iloc[0]
        lp = p["LP_p_chi2"]
        ext = (np.sign(p["원본_β"]) == np.sign(x["β"])) and abs(p["원본_β"]) >= abs(x["β"]) / 2
        if x["p_holm6"] < 0.05 and not np.isnan(lp) and lp >= 0.10 and not ext:
            grade = "확정"
        elif x["p"] < 0.10:
            grade = "약한 증거"
        else:
            grade = "근거 없음"
        flags = ["인과 아님"]
        if (not np.isnan(lp) and lp < 0.10) or p["원본_p"] < 0.10:
            flags.append("역인과 의심")
        if ext:
            flags.append("사전 추세 연장 가능성")
        if np.isnan(lp):
            flags.append("LP 결합 계산 불가")
        if x["p"] < 0.10 and np.sign(x["β"]) != hsign[x["계정"]]:
            flags.append("반대 부호 유의(harmonized 대비)")
        rows.append({"충격": shock, "계정": x["계정"], "β_h6": x["β"], "SE": x["SE"], "p": x["p"], "p_holm6": x["p_holm6"],
                     "환산": conv(x["β"], x["계정"], shock),
                     "CI_환산_lo": min(conv(x["CI_lo"], x["계정"], shock), conv(x["CI_hi"], x["계정"], shock)),
                     "CI_환산_hi": max(conv(x["CI_lo"], x["계정"], shock), conv(x["CI_hi"], x["계정"], shock)),"단위": "%p" if x["계정"].endswith("HOLD") else "%",
                     "MDE_환산": md["MDE_환산"], "|환산|<MDE": abs(conv(x["β"], x["계정"], shock)) < md["MDE_환산"],
                     "LP_p_chi2_h6_3": lp, "원본_β_t3_t9": p["원본_β"], "원본_p": p["원본_p"],
                     "참고_h-2_β": m2["β"].iloc[0] if len(m2) else np.nan, "참고_h-2_p": m2["p"].iloc[0] if len(m2) else np.nan,
                     "참고_원본harm_β": p["원본_harmonized형태_β"], "참고_원본harm_p": p["원본_harmonized형태_p"],
                     "Holm_계정내_h1_12_최소": r[(r["계정"] == x["계정"]) & (r["충격"] == shock)]["p_holm_계정내h1_12"].min(),
                     "법인": x["법인"], "월": x["월"], "등급": grade, "표시": " · ".join(flags)})
j = pd.DataFrame(rows)
main = j[j["충격"] == "주_a'"].copy()
b = j[j["충격"] == "강건_b"].set_index("계정")
main["강건_b"] = [("0 근처(p≥0.10)" if b.loc[a, "p"] >= 0.10 else ("같은 방향" if np.sign(b.loc[a, "β_h6"]) == np.sign(bm) else "반대"))
                  for a, bm in zip(main["계정"], main["β_h6"])]
main.to_csv(os.path.join(OUT, "judgment.csv"), index=False, encoding="utf-8-sig")
j[j["충격"] == "강건_b"].to_csv(os.path.join(OUT, "robust_b.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(main[["계정", "환산", "CI_환산_lo", "CI_환산_hi", "단위", "MDE_환산", "p", "p_holm6", "LP_p_chi2_h6_3", "원본_β_t3_t9", "원본_p",
            "등급", "표시", "강건_b"]].round(4).to_string(index=False))
print()
print(j[j["충격"] == "강건_b"][["계정", "환산", "단위", "MDE_환산", "p", "p_holm6", "LP_p_chi2_h6_3", "원본_p"]].round(4).to_string(index=False))
print()
print(main[["계정", "β_h6", "SE", "참고_h-2_β", "참고_h-2_p", "참고_원본harm_β", "참고_원본harm_p", "Holm_계정내_h1_12_최소", "법인"]].round(4).to_string(index=False))
