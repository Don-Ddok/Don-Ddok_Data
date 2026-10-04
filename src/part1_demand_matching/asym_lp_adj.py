# -*- coding: utf-8 -*-
"""요구불 하방 비대칭 LP 검정 — YoY.ipynb 셀 3·4·5의 복사본 (민영 파트 ③ 재확인용)
원본 계산은 그대로 옮겼다:
  - 종속변수: 누적 ln(Y_{t+h}+1) − ln(Y_{t−1}+1), h마다 1%/99% 윈저라이즈
  - 설명변수: shock×exposed×국면더미 + shock  (shock = exp_yoy/10)
  - 법인 평균 → 연월 평균을 한 번씩 빼는 고정효과, 상수항 없는 OLS
  - 직접 검정: 두 교호작용 계수 차이 = 0
추가한 것: cov="twoway" 옵션 (lib.twoway_vcov, 법인·월 이중 군집, 임계값 t(G월−1)).
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, r"C:\Users\yues7\Downloads")   # lib.py 위치 (수정하지 않음)
from lib import twoway_vcov

HORIZONS = [-3, -2, 0, 1, 2, 3, 4, 5, 6]


def demean(data, cols):
    """원본과 같은 방식: 법인 평균 → 연월 평균을 한 번씩 뺌."""
    d = data.copy()
    d[cols] = d[cols] - d.groupby("법인ID")[cols].transform("mean")
    d[cols] = d[cols] - d.groupby("기준년월")[cols].transform("mean")
    return d


def run_lp(df, y_src, groups, test_pair, cov="R0", phase_col="phase"):
    """groups: {"증가": ["증가"], "깊은하락": ["깊은하락"], ...} — 교호작용 항 이름 → 포함 국면.
    test_pair: 직접 검정할 두 항 이름 (a, b) → H0: b_a − b_b = 0.
    cov: "R0" = 원본 법인 군집(statsmodels, 정규분포 p), "twoway" = 법인·월 이중 군집 + t(G월−1)."""
    d0 = df.sort_values(["법인ID", "기준년월"]).reset_index(drop=True).copy()
    d0["shock"] = d0["exp_yoy"] / 10.0
    d0["ln_y"] = np.log(d0[y_src].clip(lower=0) + 1)
    d0["ln_y_lag1"] = d0.groupby("법인ID")["ln_y"].shift(1)
    xnames = []
    for name, phases in groups.items():
        col = f"x_{name}"
        d0[col] = d0["shock"] * d0["exposed"] * d0[phase_col].isin(phases).astype(int)
        xnames.append(col)
    xvars = xnames + ["shock"]
    a, b = f"x_{test_pair[0]}", f"x_{test_pair[1]}"

    rows = []
    for h in HORIZONS:
        d = d0.copy()
        d["y_h"] = d.groupby("법인ID")["ln_y"].shift(-h) - d["ln_y_lag1"]
        lo, hi = d["y_h"].quantile([0.01, 0.99])
        d["y_h_w"] = d["y_h"].clip(lo, hi)
        cols = ["y_h_w"] + xvars
        v = d.dropna(subset=cols + ["법인ID", "기준년월"]).copy()
        dm = demean(v, cols)
        row = {"h": h, "n": len(v)}
        if cov == "R0":
            m = sm.OLS(dm["y_h_w"], dm[xvars]).fit(cov_type="cluster", cov_kwds={"groups": v["법인ID"]})
            for x in xnames:
                row[f"{x[2:]}_b"], row[f"{x[2:]}_t"], row[f"{x[2:]}_p"] = m.params[x], m.tvalues[x], m.pvalues[x]
            w = m.t_test(f"{a} - {b} = 0")
            row["차이"], row["직접검정_p"] = float(np.ravel(w.effect)[0]), float(np.ravel(w.pvalue)[0])
            row["G월"] = v["기준년월"].nunique()
        else:
            X = dm[xvars].values.astype(float); y = dm["y_h_w"].values.astype(float)
            XtXi = np.linalg.pinv(X.T @ X); beta = XtXi @ (X.T @ y); u = y - X @ beta
            g1 = pd.factorize(v["법인ID"])[0]; g2 = pd.factorize(v["기준년월"])[0]
            _, _, V, G1, G2 = twoway_vcov(X, u, XtXi, g1, g2)
            se = np.sqrt(np.diag(V)); dfree = G2 - 1
            for j, x in enumerate(xnames):
                t = beta[j] / se[j]
                row[f"{x[2:]}_b"], row[f"{x[2:]}_t"], row[f"{x[2:]}_p"] = beta[j], t, 2 * stats.t.sf(abs(t), dfree)
            cvec = np.zeros(len(xvars)); cvec[xvars.index(a)] = 1; cvec[xvars.index(b)] = -1
            diff = cvec @ beta; sed = np.sqrt(cvec @ V @ cvec)
            row["차이"], row["직접검정_p"] = diff, 2 * stats.t.sf(abs(diff / sed), dfree)
            row["G월"] = G2
        rows.append(row)
    return pd.DataFrame(rows)


# 원본 셀별 사양
MAIN = dict(y_src="요구불예금잔액",
            groups={"증가": ["증가"], "하락": ["하락"], "깊은하락": ["깊은하락"]},
            test_pair=("증가", "깊은하락"))
ROBUST = {
    "요구불 증가 vs 하락전체": dict(y_src="요구불예금잔액", groups={"pos": ["증가"], "neg": ["하락", "깊은하락"]}, test_pair=("pos", "neg")),
    "저축성수신 증가 vs 하락전체": dict(y_src="저축성수신잔액", groups={"pos": ["증가"], "neg": ["하락", "깊은하락"]}, test_pair=("pos", "neg")),
    "운전자금 증가 vs 하락전체": dict(y_src="여신_운전자금대출잔액", groups={"pos": ["증가"], "neg": ["하락", "깊은하락"]}, test_pair=("pos", "neg")),
    "요구불 증가 vs 하락만": dict(y_src="요구불예금잔액", groups={"pos": ["증가"], "neg": ["하락"]}, test_pair=("pos", "neg")),
}


def load_df():
    use = ["법인ID", "기준년월", "사업장_시도", "exp_yoy", "phase", "exposed",
           "요구불예금잔액", "거치식예금잔액", "적립식예금잔액", "여신_운전자금대출잔액"]
    df = pd.read_csv(r"C:\test\data\processed\df_ready.csv", usecols=use, encoding="utf-8-sig")
    df["저축성수신잔액"] = df["거치식예금잔액"].fillna(0) + df["적립식예금잔액"].fillna(0)   # 셀 4와 같음
    return df
