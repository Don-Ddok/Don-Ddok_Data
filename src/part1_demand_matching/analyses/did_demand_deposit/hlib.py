# -*- coding: utf-8 -*-
"""사후 재추정 공통 함수 (SPEC.md, 커밋 58502f3)

run_common: 계정·모형 공통 LP 추정 하나.
  y_h = f(잔액_{t+h}) − f(잔액_{t−1}) = α_i + τ_{지역×연월} + β3·(노출×X) + γ·(노출×영업일수차) + ε
  f = ln(+1) (잔액 모형) 또는 1[>0] (HOLD)
  가중 교대투영 FE 제거(수렴 1e-10) + CGM 이중 군집(법인·연월, 각 차원 G/(G−1)·(n−1)/(n−k)), p = t(G_min−1)
  군집 키는 패널 준비 때 숫자 코드로 한 번만 만든다.
"""
import numpy as np
import pandas as pd
from scipy import stats

TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
WORK = r"C:\test\data\workdays_2021_2025.csv"
FAM = list(range(1, 13))


class Panel:
    """법인ID·ym·사업장_시도·노출·잔액 열이 있는 월 패널 → 충격·영업일수·숫자 코드·시차 색인 준비"""

    def __init__(self, df, bal_col, expo_col="외환노출"):
        d = df[["법인ID", "ym", "사업장_시도", expo_col, bal_col] + [c for c in df.columns if c.startswith("_keep_")]].copy()
        d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
        t = pd.read_csv(TRIG)[["지역", "연월", "원YoY", "보정YoY"]].rename(columns={"지역": "사업장_시도", "연월": "ym"})
        w = pd.read_csv(WORK, encoding="utf-8-sig")[["ym", "전년동월차"]]
        d = d.merge(t, on=["사업장_시도", "ym"], how="left").merge(w, on="ym", how="left")
        assert d[["원YoY", "보정YoY", "전년동월차"]].notna().all().all(), "충격·영업일수 결측"
        d = d.sort_values(["법인ID", "ym"]).reset_index(drop=True)
        assert not d.duplicated(["법인ID", "ym"]).any(), "법인×월 중복"
        self.d = d
        self.bal = d[bal_col].to_numpy(float)
        self.ln = np.log(np.clip(self.bal, 0, None) + 1)
        self.pos = (self.bal > 0).astype(float)
        self.expo = d[expo_col].to_numpy(float)
        self.X = {"보정": d["보정YoY"].to_numpy(float) / 100, "원": d["원YoY"].to_numpy(float) / 100}
        self.biz = d["전년동월차"].to_numpy(float)
        self.firm = pd.factorize(d["법인ID"])[0]
        self.ymv = d["ym"].to_numpy()
        self.month = pd.factorize(d["ym"])[0]
        self.regym = pd.factorize(d["사업장_시도"] + "_" + d["ym"].astype(str))[0]
        self.obs = np.arange(len(d))                      # 법인×월 군집 = 관측 하나 (중복 없음 확인)
        midx = (d["ym"] // 100).to_numpy() * 12 + (d["ym"] % 100).to_numpy()
        key = pd.Index(self.firm.astype(np.int64) * 10000 + midx)
        self._key, self._firm, self._midx = key, self.firm.astype(np.int64), midx

    def at(self, arr, off):
        """각 행 기준 off개월 뒤 값 (관측 없으면 nan)"""
        pos = self._key.get_indexer(self._firm * 10000 + self._midx + off)
        out = np.full(len(arr), np.nan)
        ok = pos >= 0
        out[ok] = arr[pos[ok]]
        return out


def demean(M, w, codes, tol=1e-10, it=100000):
    M = M.copy()
    sws = [np.bincount(c, w) for c in codes]
    for _ in range(it):
        old = M.copy()
        for c, sw in zip(codes, sws):
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, w * M[:, j], minlength=len(sw)) / np.where(sw > 0, sw, 1))[c]
        if np.max(np.abs(M - old)) < tol:
            return M
    raise RuntimeError("FE 제거 수렴 실패")


def partial_out(M, w, fi, ri):
    """법인 FE + 지역×연월 FE를 반복 없이 정확히 제거 (FWL): 법인 가중평균을 빼고,
    같은 방식으로 법인 평균을 뺀 지역×연월 더미(72개 이하)에 가중 최소제곱으로 한 번 투영한 잔차.
    수렴한 교대투영(demean)과 같은 값 (CHANGES.md 검증)."""
    from scipy import sparse
    n = len(w)
    nf, nr = fi.max() + 1, ri.max() + 1
    S = sparse.csr_matrix((w, (fi, np.arange(n))), shape=(nf, n))          # 법인 × 관측 (가중)
    sw = np.asarray(S.sum(axis=1)).ravel()
    D = sparse.csr_matrix((np.ones(n), (np.arange(n), ri)), shape=(n, nr))
    Mf = M - (S @ M / sw[:, None])[fi]
    Df = D.toarray() - (np.asarray((S @ D).todense()) / sw[:, None])[fi]
    sq = np.sqrt(w)[:, None]
    coef = np.linalg.lstsq(Df * sq, Mf * sq, rcond=None)[0]
    return Mf - Df @ coef


def fit(y, SX, SB, w, firm, regym, month, obs):
    fc, fi = np.unique(firm, return_inverse=True)
    rc, ri = np.unique(regym, return_inverse=True)
    M = partial_out(np.column_stack([y, SX, SB]), w, fi, ri)
    yv, X = M[:, 0], M[:, 1:]
    inv = np.linalg.inv(X.T @ (X * w[:, None]))
    b = inv @ (X.T @ (w * yv))
    e = yv - X @ b
    n, k = X.shape
    psi = (X * (w * e)[:, None]) @ inv.T
    V = np.zeros((k, k)); dims = {}
    for name, g, sign in [("법인", firm, 1), ("월", month, 1), ("법인월", obs, -1)]:
        gu, gi = np.unique(g, return_inverse=True)
        G = len(gu)
        c = G / (G - 1) * (n - 1) / (n - k)
        s = np.zeros((G, k)); np.add.at(s, gi, psi)
        V += sign * c * (s.T @ s)
        dims[name] = (g, c, sign)
    return b, V, psi[:, 0], dims, n


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def run_common(P, hs, *, xkey="보정", weights=None, winsor=True, positive=False, hold=False,
               start=None, keep=None, dep=None, pf_hs=(), label="", log=print):
    """P: Panel. weights: 행 가중치 배열(없으면 1). keep: 행 사용 여부 bool 배열. dep: 종속변수를 직접 줄 때(원본 방식 사전추세).
    반환: (결과 DataFrame, h별 영향 성분 dict)"""
    base = P.pos if hold else P.ln
    lag = P.at(base, -1)
    lag_ln = P.at(P.ln, -1)
    SX = P.expo * P.X[xkey]
    SB = P.expo * P.biz
    w_all = np.ones(len(P.d)) if weights is None else np.asarray(weights, float)
    rows, parts = [], {}
    for h in hs:
        try:
            if dep is not None:
                y = dep.copy()
            else:
                y = P.at(base, h) - lag
                if positive:
                    fwd_ln = P.at(P.ln, h)
                    y[~((lag_ln > 0) & (fwd_ln > 0))] = np.nan
            m = ~np.isnan(y) & (w_all > 0)
            if keep is not None:
                m &= keep
            if start is not None:
                m &= P.ymv >= start
            yy = y[m]
            if winsor:
                lo, hi = np.quantile(yy, [0.01, 0.99]); yy = np.clip(yy, lo, hi)
            if len(np.unique(P.expo[m])) < 2:
                raise ValueError("처치군 또는 대조군 없음")
            b, V, psi, dims, n = fit(yy, SX[m], SB[m], w_all[m], P.firm[m], P.regym[m], P.month[m], P.obs[m])
            Gf, Gm = len(np.unique(P.firm[m])), len(np.unique(P.month[m]))
            df_ = min(Gf, Gm) - 1
            se = np.sqrt(V[0, 0]); tq = stats.t.ppf(0.975, df_)
            fe = pd.Series(P.expo[m]).groupby(P.firm[m]).first()
            row = {"모형": label, "h": h, "β3": b[0], "SE": se, "p": 2 * stats.t.sf(abs(b[0] / se), df_),
                   "CI_lo": b[0] - tq * se, "CI_hi": b[0] + tq * se,
                   "환산": (-0.1 * b[0] * 100) if hold else (np.exp(-0.1 * b[0]) - 1) * 100,
                   "단위": "%p" if hold else "%", "γ": b[1], "N": n, "노출법인": int((fe == 1).sum()),
                   "비노출법인": int((fe == 0).sum()), "월": Gm, "pf_β3": np.nan, "pf_SE": np.nan}
            if h in pf_hs:
                import pyfixest as pf
                x = pd.DataFrame({"y": yy, "SX": SX[m], "SB": SB[m], "w": w_all[m], "firm": P.firm[m],
                                  "regym": P.regym[m], "month": P.month[m]})
                fm = pf.feols("y ~ SX + SB | firm + regym", data=x, weights="w" if weights is not None else None,
                              vcov={"CRV1": "firm + month"}, fixef_maxiter=100000)
                row["pf_β3"], row["pf_SE"] = float(fm.coef()["SX"]), float(fm.se()["SX"])
            rows.append(row)
            parts[h] = (psi, dims)
            log(f"  [{label}] h={h:+d} β3 {b[0]:+.4f} (SE {se:.4f}) N {n:,}")
        except Exception as ex:
            log(f"  [{label}] h={h:+d} 실패: {type(ex).__name__}: {ex}")
    r = pd.DataFrame(rows)
    if len(r):
        fam = r["h"].isin(FAM)
        r["p_holm"] = np.nan
        if fam.any():
            r.loc[fam, "p_holm"] = holm(r.loc[fam, "p"])
    return r, parts


def joint(parts, hs):
    """h별 영향 성분을 법인·월·법인월 군집으로 합친 교차 공분산 (각 h 보정계수의 기하평균)"""
    k = len(hs); V = np.zeros((k, k))
    for dim in ["법인", "월", "법인월"]:
        S = []
        for h in hs:
            psi, dims = parts[h]
            g, c, sign = dims[dim]
            S.append((np.bincount(g, psi, minlength=int(g.max()) + 1), c, sign))
        L = max(len(s[0]) for s in S)
        S = [(np.pad(s, (0, L - len(s))), c, sg) for s, c, sg in S]
        for a in range(k):
            for bb in range(k):
                V[a, bb] += S[a][2] * np.sqrt(S[a][1] * S[bb][1]) * float(S[a][0] @ S[bb][0])
    return V


def joint_test(res, parts, hs=(-6, -5, -4, -3, -2)):
    hs = list(hs)
    V = joint(parts, hs)
    r = res.set_index("h").loc[hs]
    chk = pd.DataFrame({"h": hs, "SE_합친대각": np.sqrt(np.diag(V)), "SE_h별": r["SE"].to_numpy()})
    chk["상대차"] = chk["SE_합친대각"] / chk["SE_h별"] - 1
    bvec = r["β3"].to_numpy()
    wald = float(bvec @ np.linalg.pinv(V) @ bvec)
    G = int(r["월"].min())
    return {"Wald": wald, "p_chi2": float(stats.chi2.sf(wald, len(hs))),
            "p_F": float(stats.f.sf(wald / len(hs), len(hs), G - 1)),
            "검산_최대상대차": float(chk["상대차"].abs().max())}, chk
