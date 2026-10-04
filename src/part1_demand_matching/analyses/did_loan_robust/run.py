# -*- coding: utf-8 -*-
"""강건성(사후): 운전자금 대출 × 수출 충격 — 요구불 새 버전과 같은 통제 (사양: SPEC.md, 커밋 48b2056)

  y_h = ln(운전+1)_{t+h} − ln(운전+1)_{t−1} = α_i + τ_{지역×연월} + β3·(S×X) + γ·(S×영업일수차) + ε
  L1 매칭(가중) · L1-원YoY · L1-W(1/99) · L2 매칭 없음, h=−6~12 (−1 제외)
  이중 군집(법인·연월, CGM, 원본 twoway_p와 같은 보정), p = t(G_min−1), Holm h=1~12
  사전추세: LP h=−6~−2 결합 Wald(영향함수 합산) + 원본 방식(ln_{t−1} − ln_{t−7}) 재추정
원본(파트3 코드·결과, 요구불 주 결과, 데이터 폴더)은 수정하지 않는다.
실행: py -3.11 run.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

P3 = r"C:\test\external\part3_work\통계 프로젝트\여신업종분석\단계별 분석"
TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
WORK = r"C:\test\data\workdays_2021_2025.csv"
OUT = r"C:\test\outputs\did_loan_robust"
Y = "여신_운전자금대출잔액"
HS = [h for h in range(-6, 13) if h != -1]
PRE = [-6, -5, -4, -3, -2]
FAM = list(range(1, 13))
BASE_B3, BASE_PRE = -0.2311, -0.282
K = 3


def load():
    d = pd.read_parquet(os.path.join(P3, "step1_loan_industry_panel.parquet"))
    d = d[d["운전자금_보유이력"] == 1][["법인ID", "ym", "사업장_시도", "외환노출", Y]].copy()
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    t = pd.read_csv(TRIG)[["지역", "연월", "원YoY", "보정YoY"]].rename(columns={"지역": "사업장_시도", "연월": "ym"})
    w = pd.read_csv(WORK, encoding="utf-8-sig")[["ym", "전년동월차"]]
    d = d.merge(t, on=["사업장_시도", "ym"], how="left").merge(w, on="ym", how="left")
    assert d[["원YoY", "보정YoY", "전년동월차"]].notna().all().all()
    d["X_adj"], d["X_raw"] = d["보정YoY"] / 100, d["원YoY"] / 100
    d["midx"] = (d["ym"] // 100) * 12 + d["ym"] % 100
    d["ln"] = np.log(d[Y] + 1)
    d["regym"] = d["사업장_시도"] + "_" + d["ym"].astype(str)
    print(f"패널: 운전자금 보유 법인 {d['법인ID'].nunique():,}, 행 {len(d):,}, 노출 {d.loc[d['외환노출'] == 1, '법인ID'].nunique():,}")
    print(f"  0 이하 잔액 행 {(d[Y] <= 0).sum():,} (빼지 않음, ln(+1)) / 두 지역에 걸친 법인 {(d.groupby('법인ID')['사업장_시도'].nunique() > 1).sum()}")
    return d


def weights():
    m = pd.read_parquet(os.path.join(P3, "step3_psm_matched.parquet"))
    wc = m["법인ID_대조"].value_counts() / K
    wt = pd.Series(1.0, index=m["법인ID_노출"].unique())
    return pd.concat([wt, wc]).groupby(level=0).sum()   # 원본 match_weights와 같음 (노출·대조 겹침 없음)


def shifted(d, off):
    s = d[["법인ID", "midx", "ln"]].assign(midx=lambda x: x["midx"] - off)
    return d[["법인ID", "midx"]].merge(s, on=["법인ID", "midx"], how="left")["ln"].to_numpy()


def demean(M, w, groups, tol=1e-12, it=10000):
    """가중 교대투영으로 여러 FE 제거"""
    M = M.copy()
    codes = [pd.factorize(g)[0] for g in groups]
    for _ in range(it):
        old = M.copy()
        for c in codes:
            sw = np.bincount(c, w)
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, w * M[:, j]) / sw)[c]
        if np.max(np.abs(M - old)) < tol:
            break
    return M


def fit(x, w):
    """자체 추정: β, 이중 군집 V, 관측치별 영향 성분(점수)과 보정계수"""
    M = demean(x[["y", "SX", "SB"]].to_numpy(float), w, [x["법인ID"].to_numpy(), x["regym"].to_numpy()])
    yv, X = M[:, 0], M[:, 1:]
    inv = np.linalg.inv(X.T @ (X * w[:, None]))
    b = inv @ (X.T @ (w * yv))
    e = yv - X @ b
    n, k = X.shape
    psi = (X * (w * e)[:, None]) @ inv.T              # 관측치별 inv·x·w·e
    dims = {}
    V = np.zeros((k, k))
    for name, keys, sign in [("법인", ["법인ID"], 1), ("월", ["ym"], 1), ("법인월", ["법인ID", "ym"], -1)]:
        g = pd.factorize(pd.MultiIndex.from_frame(x[keys]))[0] if len(keys) > 1 else pd.factorize(x[keys[0]])[0]
        G = g.max() + 1
        c = G / (G - 1) * (n - 1) / (n - k)
        s = np.zeros((G, k)); np.add.at(s, g, psi)
        V += sign * c * (s.T @ s)
        dims[name] = (x[keys].astype(str).agg("|".join, axis=1).to_numpy(), c, sign, G)
    return b, V, psi[:, 0], dims, n


def run_model(d, name, xcol, w_map, winsor=False, hs=HS, dep=None):
    rows, parts = [], {}
    for h in hs:
        x = d.copy()
        if dep is None:
            x["y"] = shifted(d, h) - shifted(d, -1)
        else:
            x["y"] = dep
        x["SX"] = x[xcol] * x["외환노출"]
        x["SB"] = x["전년동월차"] * x["외환노출"]
        x["w"] = 1.0 if w_map is None else x["법인ID"].map(w_map)
        n0 = len(x)
        x = x[x["y"].notna() & (x["w"] > 0)].copy()
        if winsor:
            lo, hi = x["y"].quantile([0.01, 0.99]); x["y"] = x["y"].clip(lo, hi)
        w = x["w"].to_numpy(float)
        b, V, psi, dims, n = fit(x, w)
        Gf, Gm = x["법인ID"].nunique(), x["ym"].nunique()
        df_ = min(Gf, Gm) - 1
        se = np.sqrt(V[0, 0]); tq = stats.t.ppf(0.975, df_)
        # pyfixest 교차 확인
        fm = pf.feols("y ~ SX + SB | 법인ID + regym", data=x, weights="w" if w_map is not None else None,
                      vcov={"CRV1": "법인ID + ym"}, fixef_maxiter=100000)
        rows.append({"모형": name, "h": h, "β3": b[0], "SE": se, "p": 2 * stats.t.sf(abs(b[0] / se), df_),
                     "CI_lo": b[0] - tq * se, "CI_hi": b[0] + tq * se,
                     "10%p하락_차이%": (np.exp(-0.1 * b[0]) - 1) * 100, "γ": b[1],
                     "N": n, "법인": Gf, "월": Gm, "제외_미관측": n0 - n,
                     "pf_β3": float(fm.coef()["SX"]), "pf_SE": float(fm.se()["SX"])})
        parts[h] = (x[["법인ID", "ym"]].reset_index(drop=True), psi, dims, se)
    r = pd.DataFrame(rows)
    fam = r["h"].isin(FAM)
    if fam.any():
        r["p_holm"] = np.nan
        o = np.argsort(r.loc[fam, "p"].to_numpy()); p = r.loc[fam, "p"].to_numpy(); m = len(p)
        adj = np.empty(m); run = 0.0
        for rk, i in enumerate(o):
            run = max(run, (m - rk) * p[i]); adj[i] = min(1.0, run)
        r.loc[fam, "p_holm"] = adj
    return r, parts


def joint(parts, hs):
    """h별 영향 성분을 법인·월·법인월 군집으로 합쳐 교차 공분산 (각 h의 보정계수 기하평균)"""
    k = len(hs)
    V = np.zeros((k, k))
    for dim in ["법인", "월", "법인월"]:
        sums = []
        for h in hs:
            key, c, sign, _ = parts[h][2][dim]
            sums.append((pd.Series(parts[h][1]).groupby(key).sum(), c, sign))
        for a in range(k):
            for bb in range(k):
                sa, ca, sg = sums[a]; sb, cb, _ = sums[bb]
                common = sa.index.intersection(sb.index)
                V[a, bb] += sg * np.sqrt(ca * cb) * float((sa[common] * sb[common]).sum())
    return V


def main():
    d = load()
    W = weights()
    print(f"매칭 가중치: 노출 {int((W.index.isin(d.loc[d['외환노출'] == 1, '법인ID'])).sum())}곳, 전체 {len(W):,}곳")
    res, parts_all = [], {}
    for name, xcol, wm, ws in [("L1", "X_adj", W, False), ("L1-원YoY", "X_raw", W, False),
                               ("L1-W", "X_adj", W, True), ("L2", "X_adj", None, False)]:
        r, parts = run_model(d, name, xcol, wm, ws)
        res.append(r); parts_all[name] = parts
        print(f"[{name}] 완료, h=6 β3 {r.loc[r.h == 6, 'β3'].item():+.4f}", flush=True)
    out = pd.concat(res, ignore_index=True)

    # 사전추세 ① LP 결합
    pre = {}
    for name in ["L1", "L2"]:
        V = joint(parts_all[name], PRE)
        r = out[(out["모형"] == name) & out["h"].isin(PRE)].set_index("h").loc[PRE]
        chk = pd.DataFrame({"h": PRE, "SE_합친대각": np.sqrt(np.diag(V)), "SE_h별": r["SE"].to_numpy()})
        chk["상대차"] = (chk["SE_합친대각"] / chk["SE_h별"] - 1)
        print(f"\n[결합검정 검산 {name}]\n{chk.round(6).to_string(index=False)}")
        bvec = r["β3"].to_numpy()
        wald = float(bvec @ np.linalg.pinv(V) @ bvec)
        G = int(r["월"].min())
        pre[name] = {"Wald": wald, "p_chi2": float(stats.chi2.sf(wald, 5)),
                     "p_F": float(stats.f.sf(wald / 5, 5, G - 1)), "검산_최대상대차": float(chk["상대차"].abs().max())}
    # 사전추세 ② 원본 방식 (ln_{t−1} − ln_{t−7}), 새 통제
    dep = shifted(d, -1) - shifted(d, -7)
    orig = []
    for name, wm in [("L1", W), ("L2", None)]:
        r, _ = run_model(d, name + "_원본방식사전", "X_adj", wm, hs=[0], dep=dep)
        orig.append(r)
    orig = pd.concat(orig, ignore_index=True)

    out.to_csv(os.path.join(OUT, "results_table.csv"), index=False, encoding="utf-8-sig")
    orig.to_csv(os.path.join(OUT, "pretrend_original_method.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pre).T.to_csv(os.path.join(OUT, "pretrend_joint.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n[h별 결과]\n", out[["모형", "h", "β3", "SE", "p", "p_holm", "CI_lo", "CI_hi", "10%p하락_차이%", "N", "법인", "월", "pf_β3", "pf_SE"]].round(4).to_string(index=False))
    print("\n[사전추세 LP 결합]\n", pd.DataFrame(pre).T.round(4))
    print("\n[사전추세 원본 방식]\n", orig[["모형", "β3", "SE", "p", "CI_lo", "CI_hi", "N", "법인", "월"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
