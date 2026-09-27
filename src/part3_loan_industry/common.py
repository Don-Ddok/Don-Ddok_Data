# -*- coding: utf-8 -*-
"""
3·4단계 공통 함수 — 패널 + 수출 충격 준비, 고정효과 회귀
  식: 대출증감률_h(i,t) = 법인FE + 연도더미 + β1·수출YoY(r,t) + β3·수출YoY(r,t)×외환노출(i)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import lsqr

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parents[1] / "외부데이터" / "export_region.csv"
XCOLS = ["수출YoY", "교호", "연도2024", "연도2025"]


def load_shock():
    """회사 소재지(대구/경북) 월별 수출액 전년동월비, 소수(0.1 = 10%)"""
    ex = pd.read_csv(EXPORT, encoding="utf-8-sig")
    s = ex[(ex["ym"] >= 202301) & (ex["ym"] <= 202512)][["region", "ym", "yoy_exp_amt"]]
    return s.rename(columns={"region": "사업장_시도"}).assign(수출YoY=lambda d: d["yoy_exp_amt"] / 100)


def load_panel():
    """운전자금 보유 회사 패널 + 수출YoY + 교호항 + 연도 더미"""
    df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
    df = df[df["운전자금_보유이력"] == 1]
    df = df.merge(load_shock()[["사업장_시도", "ym", "수출YoY"]], on=["사업장_시도", "ym"], how="left")
    assert df["수출YoY"].notna().all(), "수출YoY가 비어 있는 행이 있음"
    df["교호"] = df["수출YoY"] * df["외환노출"]
    for y in (2024, 2025):
        df[f"연도{y}"] = (df["ym"] // 100 == y).astype(float)
    return df


COV = ["log_초기수신", "log_초기운전", "최우수", "우수", "거래기간"]
FX = HERE.parents[1] / "외부데이터" / "환율_ECOS원자료_202101_202512.csv"


def add_cov_interactions(df, ps):
    """기준값(노출 회사 평균으로 중심화) × 수출YoY 교호항 추가. 반환: (df, 교호항 이름 목록)"""
    center = ps.loc[ps["외환노출"] == 1, COV].mean()
    cov_c = (ps[COV] - center).add_prefix("c_")
    df = df.join(cov_c, on="법인ID")
    assert df[cov_c.columns].notna().all().all(), "기준값이 빠진 회사가 있음"
    names = []
    for c in COV:
        df[f"충격×{c}"] = df["수출YoY"] * df[f"c_{c}"]
        names.append(f"충격×{c}")
    return df, names


def add_fx(df):
    """원/달러 월평균 전년동월비(ECOS 원자료)와 × 외환노출 교호항 추가"""
    fx = pd.read_csv(FX, encoding="utf-8-sig")[["ym", "전년동월비_%"]]
    df = df.merge(fx.rename(columns={"전년동월비_%": "환율YoY"}), on="ym", how="left")
    assert df["환율YoY"].notna().all(), "환율이 빠진 달이 있음"
    df["환율YoY"] /= 100
    df["환율×노출"] = df["환율YoY"] * df["외환노출"]
    return df


def build_firm_table(panel, min_ind=20):
    """3-2와 같은 회사 단위 매칭 기준값 표 (첫 3개월 수신·운전자금 로그, 첫 달 등급, 거래기간, 업종 더미)"""
    d = panel.sort_values(["법인ID", "ym"]).copy()
    d["수신합"] = d[["요구불예금잔액", "거치식예금잔액", "적립식예금잔액"]].sum(axis=1)
    d["순번"] = d.groupby("법인ID").cumcount()
    early = d[d["순번"] < 3].groupby("법인ID")[["수신합", "여신_운전자금대출잔액"]].mean()
    firm = d.groupby("법인ID").agg(외환노출=("외환노출", "first"), 거래기간=("거래기간", "first"),
                                  업종=("업종_중분류", "first"), 등급=("법인_고객등급", "first"),
                                  첫관측=("ym", "min"))
    firm["log_초기수신"] = np.log1p(early["수신합"])
    firm["log_초기운전"] = np.log1p(early["여신_운전자금대출잔액"])
    cnt = firm["업종"].value_counts()
    firm["업종"] = firm["업종"].where(firm["업종"].map(cnt) >= min_ind, "기타")
    firm["최우수"] = (firm["등급"] == "최우수").astype(int)
    firm["우수"] = (firm["등급"] == "우수").astype(int)
    ind = pd.get_dummies(firm["업종"], prefix="업종", drop_first=True, dtype=int)
    return firm.join(ind), list(ind.columns)


def psm_match(firm, ind_cols, k=3):
    """3-2와 같은 매칭: 로지스틱 성향점수(표준화, 벌점 없음) → 공통지지 → 1:k 최근접이웃(복원)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    cols = ["최우수", "우수", "log_초기수신", "log_초기운전", "거래기간"] + ind_cols
    X = firm[cols].to_numpy(float)
    sd = X.std(0)
    X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    T = firm["외환노출"].to_numpy() == 1
    ps = LogisticRegression(C=np.inf, max_iter=5000).fit(X, T).predict_proba(X)[:, 1]
    lo, hi = ps[~T].min(), ps[~T].max()
    mt = T & (ps >= lo) & (ps <= hi)
    ctrl = np.where(~T)[0]
    dist, pos = NearestNeighbors(n_neighbors=k).fit(ps[ctrl].reshape(-1, 1)).kneighbors(ps[mt].reshape(-1, 1))
    ids = firm.index.to_numpy()
    pairs = pd.DataFrame({"법인ID_노출": np.repeat(ids[mt], k), "법인ID_대조": ids[ctrl[pos]].ravel(),
                          "성향점수차": dist.ravel()})
    return pairs, pd.Series(ps, index=firm.index, name="성향점수")


def match_weights(m, k=3):
    """매칭 쌍 → 회사별 가중치 (노출 1, 대조 = 뽑힌 횟수 ÷ k)"""
    w_ctrl = m["법인ID_대조"].value_counts() / k
    w_treat = pd.Series(1.0, index=m["법인ID_노출"].unique())
    return pd.concat([w_treat, w_ctrl])


def fe_reg(d, y, w=None, xcols=XCOLS):
    """법인 고정효과(평균 빼기) + 가중 최소제곱 + 법인·월 클러스터 표준오차"""
    d = d[d[y].notna()].copy()
    d["_w"] = 1.0 if w is None else d["법인ID"].map(w)
    d = d[d["_w"] > 0]
    cols = [y] + xcols
    dm = d[cols] - d.groupby("법인ID")[cols].transform("mean")   # 가중치가 회사 안에서 일정하므로 단순 평균
    X, Y, W = dm[xcols].to_numpy(), dm[y].to_numpy(), d["_w"].to_numpy()
    XtWX_inv = np.linalg.inv(X.T @ (X * W[:, None]))
    b = XtWX_inv @ (X.T @ (W * Y))
    e = Y - X @ b
    n, k = X.shape
    out = {"b": b, "n": n, "firms": d["법인ID"].nunique(), "d": d, "X": X, "Y": Y, "W": W, "e": e,
           "XtWX_inv": XtWX_inv}
    for cl in ["법인ID", "ym"]:
        g = pd.factorize(d[cl])[0]
        G = g.max() + 1
        score = np.zeros((G, k))
        np.add.at(score, g, X * (W * e)[:, None])
        V = XtWX_inv @ (score.T @ score) @ XtWX_inv * G / (G - 1) * (n - 1) / (n - k)
        se = np.sqrt(np.diag(V))
        out[cl] = (se, 2 * stats.t.sf(np.abs(b / se), G - 1), G)
    return out


def twoway_p(r, j=1):
    """법인·월 이중 클러스터(Cameron-Gelbach-Miller): V = V_법인 + V_월 − V_(법인×월).
    자유도는 적은 쪽 클러스터 수 − 1. 반환: (표준오차, p)"""
    X, W, e, inv = r["X"], r["W"], r["e"], r["XtWX_inv"]
    n, k = X.shape
    V = np.zeros((k, k))
    for keys, sign in [(["법인ID"], 1), (["ym"], 1), (["법인ID", "ym"], -1)]:
        g = pd.factorize(pd.MultiIndex.from_frame(r["d"][keys]))[0] if len(keys) > 1 \
            else pd.factorize(r["d"][keys[0]])[0]
        G = g.max() + 1
        s = np.zeros((G, k))
        np.add.at(s, g, X * (W * e)[:, None])
        V += sign * inv @ (s.T @ s) @ inv * G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(V[j, j])
    G_min = min(r["법인ID"][2], r["ym"][2])
    return se, 2 * stats.t.sf(abs(r["b"][j] / se), G_min - 1)


def lsdv_check(res, y, xcols=XCOLS):
    """같은 식을 회사 더미를 직접 넣은 희소행렬 회귀로 다시 풀어 β3 반환 (이중 확인용)"""
    d = res["d"]
    fid = pd.factorize(d["법인ID"])[0]
    D = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), fid)))
    Xf = sparse.hstack([sparse.csr_matrix(d[xcols].to_numpy()), D]).tocsr()
    sw = np.sqrt(d["_w"].to_numpy())
    sol = lsqr(sparse.diags(sw) @ Xf, sw * d[y].to_numpy(), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return sol[1]
