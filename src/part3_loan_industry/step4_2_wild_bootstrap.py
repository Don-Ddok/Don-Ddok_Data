# -*- coding: utf-8 -*-
"""
4단계-2: 월 단위 wild cluster bootstrap — 월 클러스터 29개로 p값이 과소추정됐는지
  방법: WCR(귀무가설 β3=0을 부과한 잔차) bootstrap-t, 월 클러스터 단위로 부호를 무작위로 뒤집음
        가중치 2종: Rademacher(±1), Webb(6점 — 클러스터가 적을 때 권장)
        9,999회 반복, p = |t*| ≥ |t 관측값|인 비율
  추가: 법인·월 이중 클러스터 표준오차(Cameron-Gelbach-Miller, common.twoway_p)
  이중 확인: 모든 부호를 +1로 두면 bootstrap 계수가 원래 계수와 같아야 함
입력: 패널(common.py), step3_psm_matched.parquet
"""
import sys

import numpy as np
import pandas as pd
from common import HERE, fe_reg, load_panel, match_weights, twoway_p

sys.stdout.reconfigure(encoding="utf-8")

B = 9999
J = 1                                   # β3 = 수출YoY×외환노출 (XCOLS 두 번째)
rng = np.random.default_rng(20260923)
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


def wild_bootstrap(r, draws):
    X, Y, W = r["X"], r["Y"], r["W"]
    n, k = X.shape
    g = pd.factorize(r["d"]["ym"])[0]
    G = g.max() + 1
    corr = G / (G - 1) * (n - 1) / (n - k)
    inv = r["XtWX_inv"]

    def cl_se(scores):                   # scores: (G, k) → β3 표준오차
        return np.sqrt((inv @ (scores.T @ scores) @ inv)[J, J] * corr)

    # 관측 t (월 클러스터)
    s_obs = np.zeros((G, k)); np.add.at(s_obs, g, X * (W * r["e"])[:, None])
    t_obs = r["b"][J] / cl_se(s_obs)

    # 귀무가설(β3=0) 부과: β3 열을 뺀 회귀
    keep = [c for c in range(k) if c != J]
    Xr = X[:, keep]
    br = np.linalg.solve(Xr.T @ (Xr * W[:, None]), Xr.T @ (W * Y))
    yhat_r, u_r = Xr @ br, Y - Xr @ br

    # 클러스터별로 미리 합쳐 두기 (반복마다 전체 행을 다시 계산하지 않도록)
    WX = X * W[:, None]
    s1 = np.zeros((G, k)); np.add.at(s1, g, WX * yhat_r[:, None])     # X'W ŷ_r
    s2 = np.zeros((G, k)); np.add.at(s2, g, WX * u_r[:, None])        # X'W u_r
    M = np.zeros((G, k, k)); np.add.at(M, g, WX[:, :, None] * X[:, None, :])  # X'W X
    b_r_full = inv @ s1.sum(0)
    C = inv @ s2.T                                                    # (k, G)

    def one(v):
        b_star = b_r_full + C @ v
        scores = s1 + v[:, None] * s2 - M @ b_star
        return b_star, b_star[J] / cl_se(scores)

    # 이중 확인: 부호 전부 +1 → 원래 데이터 → 원래 계수
    b_chk, t_chk = one(np.ones(G))
    assert np.allclose(b_chk, r["b"]) and np.isclose(t_chk, t_obs), "bootstrap 구성 오류"

    out = {"t_obs": t_obs, "G": G}
    for name in draws:
        V = rng.choice([-1.0, 1.0], (B, G)) if name == "Rademacher" else rng.choice(WEBB, (B, G))
        t_star = np.array([one(v)[1] for v in V])
        out[name] = np.mean(np.abs(t_star) >= np.abs(t_obs))
    return out


df = load_panel()
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

rows = []
for label, h, w in [("매칭 후", 6, W_MATCH), ("매칭 후", 3, W_MATCH), ("매칭 전", 6, None)]:
    r = fe_reg(df, f"대출증감률_h{h}", w)
    bs = wild_bootstrap(r, ["Rademacher", "Webb"])
    se2, p2 = twoway_p(r, J)
    rows.append({"표본": label, "h": h, "β3": r["b"][J], "월 클러스터": bs["G"],
                 "p_법인(공식)": r["법인ID"][1][J], "p_월(공식)": r["ym"][1][J], "p_이중클러스터": p2,
                 "p_월 WCB Rademacher": bs["Rademacher"], "p_월 WCB Webb": bs["Webb"]})
    print(f"완료: {label} h={h}")

res = pd.DataFrame(rows)
print(f"\n=== 4-2 결과 (β3, 반복 {B:,}회, 이중 확인 통과) ===")
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(res.round(4).to_string(index=False))
