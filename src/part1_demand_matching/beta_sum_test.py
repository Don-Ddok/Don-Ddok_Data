# -*- coding: utf-8 -*-
"""β₁+β₃ 합 검정 (사전등록: 분석결과\\민영_β합검정_사전등록.md)

beta1_common.py와 같은 식의 복사본:
  ln(Y_{t+h}+1) − ln(Y_{t−1}+1) = α_i + δ_m(달력월) + β1·S + β3·(S×D) + ε
  법인·월 이중 군집 (lib.twoway_vcov), p = t(G월−1), 가설별 h=0~12 Holm 보정.
원본(beta1_common.py, lib.py)은 수정하지 않는다. 데이터 폴더에는 쓰지 않는다.
실행: py -3.11 beta_sum_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r"C:\Users\yues7\Downloads")   # lib.py 위치 (수정하지 않음)
from lib import twoway_vcov

DATA = r"C:\test\data\processed\df_ready.csv"
TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
OUT = r"C:\test\분석결과"
REGIONS = {"대구": "대구", "경북": "경북"}    # df_ready 사업장_시도 → 트리거 지역
PERIOD = (202301, 202512)
FAMILY = list(range(0, 13))                  # Holm 묶음 (h = 0~12)
PRE = [-3, -2]                               # 사전추세 점검
ALPHA = 0.05
WINSOR = (0.01, 0.99)
STAGE_H = {"적기": range(3, 7), "정점·유지": range(7, 13)}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def load():
    cols = ["기준년월", "법인ID", "사업장_시도", "요구불예금잔액", "exp_yoy", "exposed"]
    d = pd.read_csv(DATA, usecols=cols)
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(REGIONS) & d["기준년월"].between(*PERIOD)].copy()
    d["지역"] = d["사업장_시도"].map(REGIONS)
    t = pd.read_csv(TRIG)[["지역", "연월", "보정YoY"]].rename(columns={"연월": "기준년월"})
    d = d.merge(t, on=["지역", "기준년월"], how="left")
    d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
    d["midx"] = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
    d["ln_y"] = np.log(d["요구불예금잔액"].clip(lower=0) + 1)
    return d


def lead(d, h):
    """정확히 h개월 뒤(또는 −1개월) 관측값만 쓴다."""
    key = d[["법인ID", "midx", "ln_y"]]
    f = key.assign(midx=key["midx"] - h).rename(columns={"ln_y": "y_lead"})
    b = key.assign(midx=key["midx"] + 1).rename(columns={"ln_y": "y_lag1"})
    return d.merge(f, on=["법인ID", "midx"], how="left").merge(b, on=["법인ID", "midx"], how="left")


def est(d, h, shock_col):
    x = lead(d, h)
    x["y"] = x["y_lead"] - x["y_lag1"]
    x["S"] = x[shock_col] / 10.0
    x = x.dropna(subset=["y", "S"]).copy()
    lo, hi = x["y"].quantile(list(WINSOR)); x["y"] = x["y"].clip(lo, hi)
    x["SD"] = x["S"] * x["exposed"]
    md = pd.get_dummies(x["기준년월"] % 100, prefix="m", drop_first=True).astype(float)
    Z = pd.concat([x[["y", "S", "SD"]], md], axis=1)
    Z = Z - Z.groupby(x["법인ID"].values).transform("mean")         # 법인 고정효과
    y = Z["y"].values; X = Z.drop(columns="y").values
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ (X.T @ y); u = y - X @ b
    g1 = pd.factorize(x["법인ID"])[0]; g2 = pd.factorize(x["기준년월"])[0]
    _, _, V, _, Gm = twoway_vcov(X, u, XtXi, g1, g2)
    dfree = Gm - 1
    se1, se3 = np.sqrt(V[0, 0]), np.sqrt(V[1, 1])
    se13 = np.sqrt(V[0, 0] + V[1, 1] + 2 * V[0, 1])
    b13 = b[0] + b[1]
    p = lambda t: 2 * stats.t.sf(abs(t), dfree)
    return {"h": h, "b1": b[0], "p1": p(b[0] / se1), "b3": b[1], "p3": p(b[1] / se3),
            "b13": b13, "se13": se13, "t13": b13 / se13, "p13": p(b13 / se13),
            "ci13_lo": b13 - stats.t.ppf(.975, dfree) * se13, "ci13_hi": b13 + stats.t.ppf(.975, dfree) * se13,
            "G월": Gm, "N": len(y), "법인": x["법인ID"].nunique()}


def label(res):
    """사전등록 규칙: 주 사양 H1 Holm 결과 → 단계 라벨."""
    sig = set(res.loc[res["h"].isin(FAMILY) & (res["p13_holm"] < ALPHA), "h"])
    out = {}
    for stage, hs in STAGE_H.items():
        hs = list(hs); s = [h for h in hs if h in sig]
        if len(s) == len(hs):
            out[stage] = f"요구불 반응 추정 (h={hs[0]}~{hs[-1]} 유의, 상품 효과 아님)"
        elif s:
            out[stage] = f"요구불 반응 일부 시차만 유의 (h={','.join(map(str, s))})"
        else:
            out[stage] = "요구불 반응 근거 약함 (Holm 보정 후 유의 시차 없음)"
    return out


def main():
    d = load()
    print(f"표본: 법인 {d['법인ID'].nunique():,} / 법인×월 {len(d):,} / 노출 법인 "
          f"{d.loc[d['exposed'] == 1, '법인ID'].nunique():,} / 보정YoY 결측 {d['보정YoY'].isna().sum()}", flush=True)
    allres = {}
    for spec, col in [("주_보정충격", "보정YoY"), ("보조_원충격", "exp_yoy")]:
        rows = []
        for h in PRE + FAMILY:
            r = est(d, h, col); rows.append(r)
            print(f"[{spec}] h={h:+3d} β1+β3 {r['b13']:+.4f} (p {r['p13']:.3f}) | β3 {r['b3']:+.4f} (p {r['p3']:.3f}) "
                  f"| β1 {r['b1']:+.4f} | N {r['N']:,}", flush=True)
        res = pd.DataFrame(rows)
        fam = res["h"].isin(FAMILY)
        for k in ["p13", "p3"]:
            res[f"{k}_holm"] = np.nan
            res.loc[fam, f"{k}_holm"] = holm(res.loc[fam, k])
        res["효과_%"] = -res["b13"] * 100              # 수출 YoY 10%p 하락 시 누적 로그 변화×100
        res.insert(0, "사양", spec)
        allres[spec] = res
    out = pd.concat(allres.values())
    out.to_csv(os.path.join(OUT, "민영_β합검정.csv"), index=False, encoding="utf-8-sig")
    lab = label(allres["주_보정충격"])
    pd.Series(lab).to_json(os.path.join(OUT, "민영_β합검정_라벨.json"), force_ascii=False, indent=1)
    print("\n[라벨 (주 사양 H1, Holm)]", lab)
    print("[보조 사양 라벨 (참고)]", label(allres["보조_원충격"]))


if __name__ == "__main__":
    main()
