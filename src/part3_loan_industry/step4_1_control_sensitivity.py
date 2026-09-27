# -*- coding: utf-8 -*-
"""
4단계-1: 대조군 민감도 — 대조군을 바꿔도 β3가 유지되는가
  3-3에서 매칭 전(β3=−0.084, 무의미)과 매칭 후(β3=−0.231, 경계선 유의)가 달랐다.
  대조 회사 한 곳이 최대 17번 뽑혔으므로, 소수 대조 회사가 결과를 끌고 갔는지 확인한다.
  방식:
    A 기준(3-2)          1:3 복원, 뽑힌 횟수만큼 가중
    B 가중치 없음        A와 같은 회사들, 대조 회사 한 번씩만
    C 캘리퍼 0.01        성향점수 차이 0.01 넘는 짝 제외
    D 캘리퍼 0.005       성향점수 차이 0.005 넘는 짝 제외
    E 많이 뽑힌 대조 제외 5번 이상 뽑힌 대조 회사를 빼고 남은 짝만
    F 1:1 비복원          한 대조 회사는 한 번만 (무작위 순서, 가장 가까운 미사용 회사)
    G 1:5 복원            대조를 더 많이
입력: step3_psm_matched.parquet, step3_propensity.parquet, 패널(common.py)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import HERE, fe_reg, load_panel, match_weights

sys.stdout.reconfigure(encoding="utf-8")

df = load_panel()
m3 = pd.read_parquet(HERE / "step3_psm_matched.parquet")
ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
T = ps["외환노출"] == 1
COV = ["log_초기수신", "log_초기운전", "최우수", "우수", "거래기간"]


def pairs_from_knn(k):
    """1:k 복원 매칭 (3-2와 같은 공통지지 규칙)"""
    p = ps["성향점수"]
    lo, hi = p[~T].min(), p[~T].max()
    t_ids = p[T & p.between(lo, hi)].index
    c_ids = p[~T].index
    dist, pos = NearestNeighbors(n_neighbors=k).fit(p[c_ids].to_numpy()[:, None]).kneighbors(
        p[t_ids].to_numpy()[:, None])
    return pd.DataFrame({"법인ID_노출": np.repeat(t_ids, k), "법인ID_대조": c_ids.to_numpy()[pos].ravel(),
                         "성향점수차": dist.ravel()}), k


def pairs_1to1_no_replace(seed=42):
    """1:1 비복원 탐욕 매칭"""
    p = ps["성향점수"]
    lo, hi = p[~T].min(), p[~T].max()
    t_ids = p[T & p.between(lo, hi)].index.to_numpy()
    c_ids = p[~T].index.to_numpy()
    c_ps = p[c_ids].to_numpy()
    used = np.zeros(len(c_ids), bool)
    rows = []
    for t in np.random.default_rng(seed).permutation(t_ids):
        d = np.abs(c_ps - p[t])
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        rows.append((t, c_ids[j], d[j]))
    return pd.DataFrame(rows, columns=["법인ID_노출", "법인ID_대조", "성향점수차"]), 1


def weights_from_pairs(pairs, k, unweighted=False):
    if unweighted:
        ids = pd.Index(pairs["법인ID_노출"].unique()).append(pd.Index(pairs["법인ID_대조"].unique()))
        return pd.Series(1.0, index=ids)
    # 노출 회사마다 남은 짝 수가 다를 수 있으므로 대조 가중치 = 1/(그 노출 회사의 짝 수)를 합산
    n_per_t = pairs.groupby("법인ID_노출")["법인ID_대조"].transform("size")
    w_ctrl = (1 / n_per_t).groupby(pairs["법인ID_대조"]).sum()
    return pd.concat([pd.Series(1.0, index=pairs["법인ID_노출"].unique()), w_ctrl])


def max_smd(w):
    """가중 매칭 표본의 기준값 SMD 최댓값 (분모는 매칭 전 분산)"""
    out = []
    for c in COV:
        x = ps[c]
        sd = np.sqrt((x[T].var() + x[~T].var()) / 2)
        wt = w.reindex(ps.index).fillna(0)
        mt = np.average(x[T], weights=wt[T]) if wt[T].sum() > 0 else np.nan
        mc = np.average(x[~T], weights=wt[~T])
        out.append(abs(mt - mc) / sd)
    return max(out)


# ── 방식별 매칭 만들기 ───────────────────────────────────────────
A = m3[["법인ID_노출", "법인ID_대조", "성향점수차"]]
cnt = A["법인ID_대조"].value_counts()
heavy = cnt[cnt >= 5].index
specs = {
    "A 기준(1:3 복원, 가중)": (A, 3, False),
    "B 가중치 없음": (A, 3, True),
    "C 캘리퍼 0.01": (A[A["성향점수차"] <= 0.01], 3, False),
    "D 캘리퍼 0.005": (A[A["성향점수차"] <= 0.005], 3, False),
    f"E 5회+ 대조 {len(heavy)}곳 제외": (A[~A["법인ID_대조"].isin(heavy)], 3, False),
    "F 1:1 비복원": (*pairs_1to1_no_replace(), False),
    "G 1:5 복원": (*pairs_from_knn(5), False),
}
assert np.allclose(weights_from_pairs(A, 3).sort_index(), match_weights(m3).sort_index()), "가중치 계산이 3-3과 다름"

print(f"기준 매칭의 대조 회사 뽑힌 횟수: 1회 {int((cnt == 1).sum())}곳, 2~4회 {int(cnt.between(2, 4).sum())}곳, "
      f"5회 이상 {len(heavy)}곳 (이들이 전체 대조 가중치의 {cnt[heavy].sum() / cnt.sum() * 100:.1f}%)")

rows = []
for name, (pairs, k, unw) in specs.items():
    w = weights_from_pairs(pairs, k, unw)
    for h in (3, 6):
        r = fe_reg(df, f"대출증감률_h{h}", w)
        b1, b3 = r["b"][0], r["b"][1]
        rows.append({"방식": name, "h": h,
                     "노출": pairs["법인ID_노출"].nunique(), "대조": pairs["법인ID_대조"].nunique(),
                     "최대SMD": max_smd(w), "β1(대조 반응)": b1, "β1+β3(노출 반응)": b1 + b3, "β3": b3,
                     "p_법인": r["법인ID"][1][1], "p_월": r["ym"][1][1]})
res = pd.DataFrame(rows)
with pd.option_context("display.width", 220, "display.max_columns", 20):
    for h in (6, 3):
        print(f"\n=== h={h} ===")
        print(res[res["h"] == h].drop(columns="h").round(3).to_string(index=False))

b6 = res[res["h"] == 6]["β3"]
print(f"\nh=6 β3 범위: {b6.min():.3f} ~ {b6.max():.3f}, 음수 {int((b6 < 0).sum())}/{len(b6)}개, "
      f"p_법인<0.05 {int((res[res['h'] == 6]['p_법인'] < 0.05).sum())}개, "
      f"p_월<0.05 {int((res[res['h'] == 6]['p_월'] < 0.05).sum())}개")
