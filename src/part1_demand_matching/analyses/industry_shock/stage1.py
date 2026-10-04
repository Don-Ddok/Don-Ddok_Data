# -*- coding: utf-8 -*-
"""1단계 (SPEC §2, 커밋 ef32bba): 업종 수출 충격이 개별 기업 수출에 닿는가
충격 S_{k,t} = 대구·경북 합산 업종(KSIC 중분류) 수출 전년동월비 (HS4 → KSIC 배분표 hs4_ksic_weights.csv)
대상: 수출 실적 있는 노출 법인 중 사용 업종 15개에 연결된 법인
종속: 3개월 합계 ln 수출 변화, h=0~3 / 법인 FE + 지역×연월 FE / 이중 군집, t(G_min−1) / Holm(h=0~3)
보고용: REG(지역별 업종 YoY), NO26(KSIC 26 제외), 비교용 지역 충격(법인 FE + 달력월 + 영업일수차, mechanism §3와 같은 통제)
출력: shock_series.csv, stage1_results.csv, stage1_judgment.txt
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, holm, run_common

OUT = r"C:\test\outputs\industry_shock"
H4 = r"C:\test\external\industry_export\hs4"
DATA = r"C:\test\data\processed\df_ready.csv"
USE = ["10", "11", "13", "14", "17", "20", "22", "23", "24", "25", "26", "27", "28", "29", "30"]
from build_mapping import KSIC   # 업종명 (같은 폴더)


def shocks():
    w = pd.read_csv(os.path.join(OUT, "hs4_ksic_weights.csv"), dtype={"hs4": str, "ksic": str})
    h = pd.concat([pd.read_csv(os.path.join(H4, f), encoding="utf-8-sig", dtype={"hsSgn": str}) for f in os.listdir(H4)])
    x = h.merge(w, left_on="hsSgn", right_on="hs4")
    x["v"] = x["expUsdAmt"] * x["w"]
    x = x[x["ksic"].isin(USE)]
    comb = x.groupby(["ksic", "기준년월"])["v"].sum().unstack("ksic").sort_index()
    reg = x.groupby(["지역", "ksic", "기준년월"])["v"].sum()
    rows = []
    for k in USE:
        s = comb[k]
        yoy = s / s.shift(12) - 1
        for ym, v in yoy.loc[202301:].items():
            rows.append({"ksic": k, "ym": ym, "지역": "합산", "수출": s[ym], "S": v})
        for r in ["대구", "경북"]:
            sr = reg.loc[r, k].sort_index()
            yr = sr / sr.shift(12) - 1
            for ym, v in yr.loc[202301:].items():
                rows.append({"ksic": k, "ym": ym, "지역": r, "수출": sr[ym], "S": v})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "shock_series.csv"), index=False, encoding="utf-8-sig")
    return t


def main():
    sh = shocks()
    norm = lambda s: re.sub(r"[\s,;]", "", str(s))
    name2k = {norm(v): k for k, v in KSIC.items()}
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "업종_중분류", "외환_수출실적금액"])
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].rename(columns={"기준년월": "ym"})
    last = d.dropna(subset=["업종_중분류"]).sort_values("ym").groupby("법인ID")["업종_중분류"].last().map(lambda s: name2k.get(norm(s)))
    exporters = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
    ids = last[last.isin(USE)].index.intersection(exporters[exporters].index)
    e = d[d["법인ID"].isin(ids)].copy()
    e["ksic"] = e["법인ID"].map(last)
    e["노출_수출"] = 1
    print(f"1단계 표본: 수출 실적 있는 사용 업종 법인 {e['법인ID'].nunique()}곳 (SPEC 예상 422), 행 {len(e):,}")
    P = Panel(e[["법인ID", "ym", "사업장_시도", "노출_수출", "외환_수출실적금액", "ksic"]].rename(columns={"ksic": "_keep_ksic"}),
              "외환_수출실적금액", expo_col="노출_수출")
    kk = P.d["_keep_ksic"].to_numpy()
    comb = sh[sh["지역"] == "합산"].set_index(["ksic", "ym"])["S"]
    regs = sh[sh["지역"] != "합산"].set_index(["지역", "ksic", "ym"])["S"]
    P.X["업종"] = np.array([comb.get((k, ym), np.nan) for k, ym in zip(kk, P.ymv)], float)
    P.X["업종_REG"] = np.array([regs.get((r, k, ym), np.nan) for r, k, ym in zip(P.d["사업장_시도"], kk, P.ymv)], float)
    assert not np.isnan(P.X["업종"]).any(), "합산 업종 충격 결측"
    x = np.clip(P.bal, 0, None)
    S3 = x + P.at(x, -1) + P.at(x, -2)
    lnS = np.log(S3 + 1); lag3 = P.at(lnS, -1)
    dep = lambda h: P.at(lnS, h) - lag3
    HS = [0, 1, 2, 3]
    res = []
    for name, kw in [("주_업종충격", dict(xkey="업종", fe2="regym", ctrl="none")),
                     ("REG_지역별업종충격", dict(xkey="업종_REG", fe2="regym", ctrl="none")),
                     ("NO26", dict(xkey="업종", fe2="regym", ctrl="none", keep=kk != "26")),
                     ("비교_지역충격", dict(xkey="보정", fe2="month", ctrl="biz"))]:
        r, _ = run_common(P, HS, dep_fn=dep, need_two=False, label=name, **kw)
        r["p_holm_h0_3"] = holm(r["p"].to_numpy())
        res.append(r)
    out = pd.concat(res, ignore_index=True)
    out.to_csv(os.path.join(OUT, "stage1_results.csv"), index=False, encoding="utf-8-sig")
    m = out[out["모형"] == "주_업종충격"]
    ok = bool(((m["β3"] > 0) & (m["p_holm_h0_3"] < 0.05)).any())
    verdict = ("업종 수출 충격은 은행 거래 법인의 실제 수출에 반영된다 (지역 충격보다 적합)" if ok
               else "업종 수출로도 개별 기업 수출과의 연결이 약하다")
    open(os.path.join(OUT, "stage1_judgment.txt"), "w", encoding="utf-8").write(verdict + "\n")
    pd.set_option("display.width", 220)
    print(out[["모형", "h", "β3", "SE", "p", "p_holm_h0_3", "CI_lo", "CI_hi", "환산", "N", "노출법인", "월"]].round(4).to_string(index=False))
    print("\n판정:", verdict)


if __name__ == "__main__":
    main()
