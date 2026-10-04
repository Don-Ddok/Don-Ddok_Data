# -*- coding: utf-8 -*-
"""외환 실적 거래건수(구간) 분포 점검 (회귀 없음)
1. 구간 종류와 관측 비율
2. t−1 → t+6 사이 구간이 바뀐 관측 비율 vs 금액이 바뀐 비율 (같은 표본·같은 쌍으로 계산)
3. 금액 0인데 건수 ≠ 0건, 건수 0건인데 금액 > 0 비율
4. 건수 구간(순서)과 금액의 Spearman
표본: 대구·경북 2023-01~2025-12, 전체 법인과 외환노출 법인 두 가지
출력: count_check.txt (집계만)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\firm_shock"
LOG = open(os.path.join(OUT, "count_check.txt"), "w", encoding="utf-8")
ORDER = ["0건", "1건", "2건", "2건초과 5건이하", "5건초과 10건이하", "10건초과 20건이하", "20건초과 30건이하",
         "30건초과 40건이하", "40건초과 50건이하", "50건 초과"]
KIND = {"수출": ("외환_수출실적거래건수", "외환_수출실적금액"), "수입": ("외환_수입실적거래건수", "외환_수입실적금액")}


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed"] + [c for v in KIND.values() for c in v])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
for k, (cc, _) in KIND.items():
    extra = set(d[cc].unique()) - set(ORDER)
    assert not extra, f"알 수 없는 구간: {extra}"
    d[k + "_구간"] = d[cc].map({v: i for i, v in enumerate(ORDER)})
midx = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
fid = pd.factorize(d["법인ID"])[0].astype(np.int64)
key = pd.Index(fid * 10000 + midx.to_numpy())


def at(arr, off):
    pos = key.get_indexer(fid * 10000 + midx.to_numpy() + off)
    out = np.full(len(arr), np.nan); ok = pos >= 0; out[ok] = np.asarray(arr, float)[pos[ok]]
    return out


for k, (cc, ac) in KIND.items():
    d[k + "_b_m1"], d[k + "_b_p6"] = at(d[k + "_구간"], -1), at(d[k + "_구간"], 6)
    d[k + "_a_m1"], d[k + "_a_p6"] = at(d[ac], -1), at(d[ac], 6)

for sname, s in [("전체 법인", d), ("외환노출 법인", d[d["exposed"] == 1])]:
    log(f"\n# 표본: 대구·경북 {sname} {s['법인ID'].nunique():,}곳, 법인×월 {len(s):,}행")
    log("\n## 1. 구간별 관측 비율")
    tab = pd.DataFrame({k: s[cc].value_counts(normalize=True).reindex(ORDER).fillna(0) for k, (cc, _) in KIND.items()})
    for b, row in tab.iterrows():
        log(f"  {b:<18} 수출 {row['수출']:7.3%}   수입 {row['수입']:7.3%}")
    log("\n## 2. t−1 → t+6 변화 비율 (t−1과 t+6 모두 관측되는 같은 쌍)")
    both = {}
    for k in KIND:
        ok = s[k + "_b_m1"].notna() & s[k + "_b_p6"].notna()
        x = s[ok]
        bch = (x[k + "_b_m1"] != x[k + "_b_p6"])
        ach = ~np.isclose(x[k + "_a_m1"], x[k + "_a_p6"])
        b0 = (x[k + "_b_m1"] == 0) & (x[k + "_b_p6"] == 0)
        a0 = np.isclose(x[k + "_a_m1"], 0) & np.isclose(x[k + "_a_p6"], 0)
        both[k] = (ok, bch, ach)
        log(f"  {k}: 쌍 {len(x):,} / 구간이 바뀜 {bch.mean():.1%} / 금액이 바뀜 {ach.mean():.1%} "
            f"/ 금액 변화 중 구간도 바뀜 {bch[ach].mean():.1%} / 구간 변화 중 금액도 바뀜 {ach[bch].mean():.1%}")
        nz = ~(b0 & a0)
        log(f"      (두 시점 모두 0건·0원인 쌍 제외 {int(nz.sum()):,}쌍) 구간 바뀜 {bch[nz].mean():.1%} / 금액 바뀜 {ach[nz].mean():.1%}")
    ok = both["수출"][0] & both["수입"][0]
    x = s[ok]
    bany = (x["수출_b_m1"] != x["수출_b_p6"]) | (x["수입_b_m1"] != x["수입_b_p6"])
    aany = ~np.isclose(x["수출_a_m1"], x["수출_a_p6"]) | ~np.isclose(x["수입_a_m1"], x["수입_a_p6"])
    fx_m1, fx_p6 = x["수출_a_m1"] + x["수입_a_m1"], x["수출_a_p6"] + x["수입_a_p6"]
    log(f"  수출 또는 수입: 구간이 바뀜 {bany.mean():.1%} / 금액이 바뀜 {aany.mean():.1%} "
        f"/ (참고) 수출+수입 금액 합이 바뀜 {(~np.isclose(fx_m1, fx_p6)).mean():.1%}")
    log("\n## 3. 금액과 건수의 불일치")
    for k, (cc, ac) in KIND.items():
        a0 = np.isclose(s[ac], 0); b0 = s[k + "_구간"] == 0
        log(f"  {k}: 금액 0인데 건수 ≠ 0건 {(a0 & ~b0).mean():.3%} ({int((a0 & ~b0).sum()):,}행) "
            f"/ 건수 0건인데 금액 > 0 {(~a0 & b0).mean():.3%} ({int((~a0 & b0).sum()):,}행)")
        if (a0 & ~b0).any():
            log(f"      금액 0·건수 ≠ 0건의 구간 분포: " + ", ".join(f"{ORDER[int(i)]} {n}" for i, n in s.loc[a0 & ~b0, k + "_구간"].value_counts().sort_index().items()))
    log("\n## 4. 건수 구간(순서 0~9)과 금액의 Spearman")
    for k, (cc, ac) in KIND.items():
        r_all = stats.spearmanr(s[k + "_구간"], s[ac])[0]
        pos = (s[k + "_구간"] > 0) & (s[ac] > 0)
        r_pos = stats.spearmanr(s.loc[pos, k + "_구간"], s.loc[pos, ac])[0]
        med = s[s[ac] > 0].groupby(k + "_구간")[ac].median()
        log(f"  {k}: 전체 {r_all:+.3f} / 둘 다 양수인 {int(pos.sum()):,}행 {r_pos:+.3f}")
        log("      구간별 금액 중앙값(금액 > 0): " + ", ".join(f"{ORDER[int(i)]} {v:g}" for i, v in med.items()))
LOG.close()
