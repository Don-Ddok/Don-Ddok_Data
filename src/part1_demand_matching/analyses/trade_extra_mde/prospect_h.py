# -*- coding: utf-8 -*-
"""진단용: A·B·C 사전 MDE를 h = 1, 3, 6, 9, 12에서 (회귀 없음, β 계산 안 함)
prospect.py와 같은 근사. k는 h마다 harmonized 요구불 C1의 같은 h 실제 SE에 맞춘다.
이 결과는 판정 시차(h=6)를 바꾸는 근거로 쓰지 않는다.
출력: mde_by_h.csv, calibration_by_h.csv, prospect_h_log.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel
from prospect import ACC, FXF, MULT, components, load

OUT = r"C:\test\outputs\trade_extra_mde"
HARM = r"C:\test\outputs\harmonized\results_all.csv"
HS = [1, 3, 6, 9, 12]
LOGF = open(os.path.join(OUT, "prospect_h_log.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def dep(P, h):
    return P.at(P.ln, h) - P.at(P.ln, -1)


per10 = lambda m: (np.exp(0.1 * m) - 1) * 100          # 큰 쪽(증가 방향)
ev = lambda m: (np.exp(m) - 1) * 100


def main():
    d = load()
    hz = pd.read_csv(HARM)
    hz = hz[hz["업종"].isna() & (hz["모형"] == "C1") & (hz["계정"] == "요구불")].set_index("h")
    # k_h: harmonized 요구불 C1
    s = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0]
    P = Panel(s, "요구불예금잔액")
    K, cal = {}, []
    for h in HS:
        y = dep(P, h)
        c = components(P, y, P.expo * P.X["보정"], ~np.isnan(y), P.expo * P.biz)
        K[h] = hz.loc[h, "SE"] / c["raw"]
        cal.append({"h": h, "N": c["N"], "N_harmonized": hz.loc[h, "N"], "월": c["월"], "σ_y": c["σ_y"], "SE_harmonized": hz.loc[h, "SE"],
                    "k": K[h], "샌드위치/실제": c["SE_샌드위치"] / hz.loc[h, "SE"],
                    "MDE_실제(10%p당, 큰쪽)": per10(MULT(c["자유도"]) * hz.loc[h, "SE"])})
    cal = pd.DataFrame(cal)
    cal.to_csv(os.path.join(OUT, "calibration_by_h.csv"), index=False, encoding="utf-8-sig")
    log("## 보정: harmonized 요구불 C1 (h별)"); log(cal.round(4).to_string(index=False))

    rows = []

    def add(label, acct, h, c, conv, unit, n_events=np.nan):
        sw = c["SE_샌드위치"] if c["SE_샌드위치"] > 0 else np.nan     # 이중 군집 분산이 음수(준정부호 아님) → 계산 불가
        for how, se in [("근사(k)", K[h] * c["raw"]), ("샌드위치", sw)]:
            rows.append({"분석": label, "계정": acct, "h": h, "방식": how, "N": c["N"], "사건수": n_events, "법인": c["법인"], "월": c["월"],
                         "자유도": c["자유도"], "σ_y": c["σ_y"], "sd_X": c["sd_X"], "SE": se, "MDE_큰쪽": conv(MULT(c["자유도"]) * se), "단위": unit})

    # A
    for acct in ["할인어음", "외상매출채권담보", "기업구매자금"]:
        col = ACC[acct]
        s = d[d.groupby("법인ID")[col].transform("max") > 0]
        P = Panel(s, col)
        for h in HS:
            y = dep(P, h)
            add("A 결제채널", acct, h, components(P, y, P.expo * P.X["보정"], ~np.isnan(y), P.expo * P.biz), per10, "% (지역 수출 10%p당)")
    # B
    exp_ids = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
    dx = d[d["법인ID"].isin(exp_ids[exp_ids].index)]
    for acct in ["요구불", "운전자금", "적립식"]:
        col = ACC[acct]
        s = dx[dx.groupby("법인ID")[col].transform("max") > 0]
        P = Panel(s, col)
        ee = P.d["_keep_exp"].to_numpy(float)
        ps = lambda off: P.at(ee, off) > 0
        zr = lambda off: P.at(ee, off) == 0
        Ds = {"중단": (ps(-1) & np.all([zr(o) for o in range(0, 6)], axis=0)).astype(float),
              "시작": (ps(0) & np.all([zr(-o) for o in range(1, 7)], axis=0)).astype(float)}
        for h in HS:
            y = dep(P, h)
            m = ~np.isnan(y) & (P.ymv >= 202307)
            for name, D in Ds.items():
                add(f"B 수출{name}", acct, h, components(P, y, D, m), ev, "% (사건 1회당)", int(D[m].sum()))
    # C
    fx = pd.read_csv(FXF, encoding="utf-8-sig").set_index("ym")["전년동월비_%"] / 100
    s = d[(d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0) & d["_keep_net"].notna()]
    P = Panel(s, "요구불예금잔액", expo_col="노출_2023")
    X = P.d["_keep_net"].to_numpy(float) * P.d["ym"].map(fx).to_numpy(float)
    for h in HS:
        y = dep(P, h)
        add("C 순노출×환율", "요구불", h, components(P, y, X, ~np.isnan(y) & (P.ymv >= 202401)), per10, "% (순노출 1, 환율 YoY 10%p당)")

    out = pd.DataFrame(rows)
    out["10%미만"] = out["MDE_큰쪽"] < 10
    out.to_csv(os.path.join(OUT, "mde_by_h.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    wide = out.pivot_table(index=["분석", "계정", "방식"], columns="h", values="MDE_큰쪽").round(1)
    log("\n## MDE (큰 쪽, %)"); log(wide.to_string())
    k = out[out["방식"] == "근사(k)"]
    log("\n## 표본 수 N (법인×월) / B 사건 수"); log(k.pivot_table(index=["분석", "계정"], columns="h", values="N").to_string())
    log(k[k["분석"].str.startswith("B")].pivot_table(index=["분석", "계정"], columns="h", values="사건수").to_string())
    log("\n## 월 수"); log(k.pivot_table(index=["분석"], columns="h", values="월", aggfunc="first").to_string())
    log("\n## 종속변수 잔차 SD σ(ỹ)"); log(k.pivot_table(index=["분석", "계정"], columns="h", values="σ_y", aggfunc="first").round(3).to_string())
    log("\n## MDE < 10%"); log(out[out["10%미만"]][["분석", "계정", "h", "방식", "MDE_큰쪽"]].round(2).to_string(index=False) or "(없음)")


if __name__ == "__main__":
    main()
