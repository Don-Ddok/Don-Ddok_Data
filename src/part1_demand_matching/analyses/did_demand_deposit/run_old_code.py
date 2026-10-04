# -*- coding: utf-8 -*-
"""사후 보조 분석: 요구불예금 × 수출 충격 — 통제를 강화한 이중차분형 국소투영 (사양: SPEC.md, 커밋 c8e48c3)

  y_h = ln(요구불+1)_{t+h} − ln(요구불+1)_{t−1} = α_i + τ_{지역×연월} + β3·(S×X) + γ·(S×영업일수차) + ε
  A 주(1/99) · A-원YoY · A-NW(윈저 없음) · A-양수(t−1·t+h 잔액 > 0) · B 매칭 1:3(가중), h=−6~12 (−1 제외)
  이중 군집(법인·연월, CGM, 파트3 twoway_p와 같은 보정), p = t(G_min−1), Holm h=1~12
  사전추세: LP h=−6~−2 결합 Wald(영향함수 합산, 검산 포함) + 원본 방식(ln_{t−1} − ln_{t−7})
원본(기존 주 결과 코드·결과, 파트3 코드, 데이터 폴더)은 수정하지 않는다. 파트3 common.py는 import만 한다.
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
sys.path.insert(0, P3)
from common import build_firm_table, match_weights, psm_match   # 수정 없이 import

DATA = r"C:\test\data\processed\df_ready.csv"
TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
WORK = r"C:\test\data\workdays_2021_2025.csv"
OUT = r"C:\test\outputs\did_demand_deposit"
Y = "요구불예금잔액"
HS = [h for h in range(-6, 13) if h != -1]
PRE = [-6, -5, -4, -3, -2]
FAM = list(range(1, 13))
PF_A = {-6, -2, 0, 6, 12}


def load():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", Y, "exposed"])
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    t = pd.read_csv(TRIG)[["지역", "연월", "원YoY", "보정YoY"]].rename(columns={"지역": "사업장_시도", "연월": "ym"})
    w = pd.read_csv(WORK, encoding="utf-8-sig")[["ym", "전년동월차"]]
    d = d.merge(t, on=["사업장_시도", "ym"], how="left").merge(w, on="ym", how="left")
    assert d[["원YoY", "보정YoY", "전년동월차"]].notna().all().all()
    d = d.sort_values(["법인ID", "ym"]).reset_index(drop=True)
    d["X_adj"], d["X_raw"] = d["보정YoY"] / 100, d["원YoY"] / 100
    d["midx"] = (d["ym"] // 100) * 12 + d["ym"] % 100
    d["ln"] = np.log(d[Y].clip(lower=0) + 1)
    d["regym"] = d["사업장_시도"] + "_" + d["ym"].astype(str)
    print(f"패널: 대구·경북 법인 {d['법인ID'].nunique():,}, 행 {len(d):,}, 노출 {d.loc[d['외환노출'] == 1, '법인ID'].nunique():,}")
    print(f"  0 이하 잔액 행 {(d[Y] <= 0).sum():,} (빼지 않음, ln(+1)) / 두 지역에 걸친 법인 {(d.groupby('법인ID')['사업장_시도'].nunique() > 1).sum()}")
    return d


def matching(d):
    """파트3 함수 그대로: 대구·경북 전체 법인(운전자금 한정 없음)으로 1:3 매칭"""
    p = pd.read_parquet(os.path.join(P3, "step1_loan_industry_panel.parquet"))
    p["사업장_시도"] = p["사업장_시도"].astype(str).str.strip()
    p = p[p["사업장_시도"].isin(["대구", "경북"])]
    firm, ind_cols = build_firm_table(p)
    pairs, ps = psm_match(firm, ind_cols, k=3)
    W = match_weights(pairs)
    ids_d = set(d["법인ID"])
    print(f"매칭: 파트3 패널 대구·경북 법인 {len(firm):,} (노출 {int(firm['외환노출'].sum()):,}) / "
          f"df_ready에만 있는 법인 {len(ids_d - set(firm.index)):,} (매칭 대상 아님) / "
          f"공통지지 밖 노출 {int(firm['외환노출'].sum()) - pairs['법인ID_노출'].nunique()} / "
          f"매칭 노출 {pairs['법인ID_노출'].nunique():,}, 대조 서로 다른 {pairs['법인ID_대조'].nunique():,}곳")
    print(f"  운전자금 초기값 0인 법인 {(firm['log_초기운전'] == 0).sum():,} / {len(firm):,}")
    # 균형표 (파트3 step3_2와 같은 SMD: 분모는 매칭 전 두 집단 분산 평균)
    T = firm["외환노출"].to_numpy() == 1
    wv = firm.index.map(W).fillna(0).to_numpy(float)
    mt = firm.index.isin(pairs["법인ID_노출"])
    rows = []
    for c in ["log_초기수신", "log_초기운전", "최우수", "우수", "거래기간"] + ind_cols:
        x = firm[c].to_numpy(float)
        sd = np.sqrt((x[T].var(ddof=1) + x[~T].var(ddof=1)) / 2)
        before = (x[T].mean() - x[~T].mean()) / sd
        ctrl_w = np.where(~T, wv, 0)
        after = (x[mt].mean() - np.average(x, weights=ctrl_w)) / sd
        rows.append({"변수": c, "SMD_매칭전": round(before, 4), "SMD_매칭후": round(after, 4)})
    bal = pd.DataFrame(rows)
    bal.to_csv(os.path.join(OUT, "balance_table.csv"), index=False, encoding="utf-8-sig")
    print(f"  균형: 매칭후 |SMD| 최대 {bal['SMD_매칭후'].abs().max():.3f}, 0.1 초과 {(bal['SMD_매칭후'].abs() > 0.1).sum()}개")
    return W


def shifted(d, off):
    s = d[["법인ID", "midx", "ln"]].assign(midx=lambda x: x["midx"] - off)
    return d[["법인ID", "midx"]].merge(s, on=["법인ID", "midx"], how="left")["ln"].to_numpy()


def demean(M, w, groups, tol=1e-12, it=10000):
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
    M = demean(x[["y", "SX", "SB"]].to_numpy(float), w, [x["법인ID"].to_numpy(), x["regym"].to_numpy()])
    yv, X = M[:, 0], M[:, 1:]
    inv = np.linalg.inv(X.T @ (X * w[:, None]))
    b = inv @ (X.T @ (w * yv))
    e = yv - X @ b
    n, k = X.shape
    psi = (X * (w * e)[:, None]) @ inv.T
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


def run_model(d, name, xcol, w_map, winsor=True, positive=False, hs=HS, dep=None, pf_hs=()):
    rows, parts = [], {}
    for h in hs:
        x = d.copy()
        if dep is None:
            fwd, lag = shifted(d, h), shifted(d, -1)
            x["y"] = fwd - lag
            if positive:                                   # t−1·t+h 잔액 모두 > 0 (ln(x+1) > 0 ⇔ x > 0)
                x.loc[~((lag > 0) & (fwd > 0)), "y"] = np.nan
        else:
            x["y"] = dep
        x["SX"] = x[xcol] * x["외환노출"]
        x["SB"] = x["전년동월차"] * x["외환노출"]
        x["w"] = 1.0 if w_map is None else x["법인ID"].map(w_map).fillna(0)
        n0 = len(x)
        x = x[x["y"].notna() & (x["w"] > 0)].copy()
        if winsor:
            lo, hi = x["y"].quantile([0.01, 0.99]); x["y"] = x["y"].clip(lo, hi)
        w = x["w"].to_numpy(float)
        b, V, psi, dims, n = fit(x, w)
        Gf, Gm = x["법인ID"].nunique(), x["ym"].nunique()
        df_ = min(Gf, Gm) - 1
        se = np.sqrt(V[0, 0]); tq = stats.t.ppf(0.975, df_)
        row = {"모형": name, "h": h, "β3": b[0], "SE": se, "p": 2 * stats.t.sf(abs(b[0] / se), df_),
               "CI_lo": b[0] - tq * se, "CI_hi": b[0] + tq * se,
               "10%p하락_차이%": (np.exp(-0.1 * b[0]) - 1) * 100, "γ": b[1],
               "N": n, "법인": Gf, "월": Gm, "제외_미관측": n0 - n, "pf_β3": np.nan, "pf_SE": np.nan}
        if h in pf_hs:
            fm = pf.feols("y ~ SX + SB | 법인ID + regym", data=x, weights="w" if w_map is not None else None,
                          vcov={"CRV1": "법인ID + ym"}, fixef_maxiter=100000)
            row["pf_β3"], row["pf_SE"] = float(fm.coef()["SX"]), float(fm.se()["SX"])
        rows.append(row)
        parts[h] = (psi, dims)
        print(f"  [{name}] h={h:+d} β3 {b[0]:+.4f} (SE {se:.4f})", flush=True)
    r = pd.DataFrame(rows)
    fam = r["h"].isin(FAM)
    if fam.any():
        r["p_holm"] = np.nan
        p = r.loc[fam, "p"].to_numpy(); o = np.argsort(p); m = len(p)
        adj = np.empty(m); run = 0.0
        for rk, i in enumerate(o):
            run = max(run, (m - rk) * p[i]); adj[i] = min(1.0, run)
        r.loc[fam, "p_holm"] = adj
    return r, parts


def joint(parts, hs):
    k = len(hs)
    V = np.zeros((k, k))
    for dim in ["법인", "월", "법인월"]:
        sums = []
        for h in hs:
            key, c, sign, _ = parts[h][1][dim]
            sums.append((pd.Series(parts[h][0]).groupby(key).sum(), c, sign))
        for a in range(k):
            for bb in range(k):
                sa, ca, sg = sums[a]; sb, cb, _ = sums[bb]
                common = sa.index.intersection(sb.index)
                V[a, bb] += sg * np.sqrt(ca * cb) * float((sa[common] * sb[common]).sum())
    return V


def main():
    d = load()
    W = matching(d)
    res, parts_all = [], {}
    for name, xcol, wm, ws, pos, pfh in [("A", "X_adj", None, True, False, PF_A),
                                         ("A-원YoY", "X_raw", None, True, False, ()),
                                         ("A-NW", "X_adj", None, False, False, ()),
                                         ("A-양수", "X_adj", None, True, True, ()),
                                         ("B", "X_adj", W, True, False, set(HS))]:
        r, parts = run_model(d, name, xcol, wm, ws, pos, pf_hs=pfh)
        res.append(r); parts_all[name] = parts
        print(f"[{name}] 완료, h=6 β3 {r.loc[r.h == 6, 'β3'].item():+.4f}", flush=True)
    out = pd.concat(res, ignore_index=True)
    out.to_csv(os.path.join(OUT, "results_table.csv"), index=False, encoding="utf-8-sig")

    pre = {}
    for name in ["A", "B"]:
        V = joint(parts_all[name], PRE)
        r = out[(out["모형"] == name) & out["h"].isin(PRE)].set_index("h").loc[PRE]
        chk = pd.DataFrame({"h": PRE, "SE_합친대각": np.sqrt(np.diag(V)), "SE_h별": r["SE"].to_numpy()})
        chk["상대차"] = chk["SE_합친대각"] / chk["SE_h별"] - 1
        print(f"\n[결합검정 검산 {name}]\n{chk.round(6).to_string(index=False)}", flush=True)
        bvec = r["β3"].to_numpy()
        wald = float(bvec @ np.linalg.pinv(V) @ bvec)
        G = int(r["월"].min())
        pre[name] = {"Wald": wald, "p_chi2": float(stats.chi2.sf(wald, 5)),
                     "p_F": float(stats.f.sf(wald / 5, 5, G - 1)), "검산_최대상대차": float(chk["상대차"].abs().max())}
    pd.DataFrame(pre).T.to_csv(os.path.join(OUT, "pretrend_joint.csv"), encoding="utf-8-sig")

    dep = shifted(d, -1) - shifted(d, -7)
    orig = pd.concat([run_model(d, nm + "_원본방식사전", "X_adj", wm, True, hs=[0], dep=dep)[0]
                      for nm, wm in [("A", None), ("B", W)]], ignore_index=True)
    orig.to_csv(os.path.join(OUT, "pretrend_original_method.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("\n[h별 결과]\n", out[["모형", "h", "β3", "SE", "p", "p_holm", "CI_lo", "CI_hi", "10%p하락_차이%", "N", "법인", "월", "pf_β3", "pf_SE"]].round(4).to_string(index=False))
    print("\n[사전추세 LP 결합]\n", pd.DataFrame(pre).T.round(4))
    print("\n[사전추세 원본 방식]\n", orig[["모형", "β3", "SE", "p", "CI_lo", "CI_hi", "N", "법인", "월"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
