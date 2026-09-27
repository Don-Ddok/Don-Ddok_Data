# -*- coding: utf-8 -*-
"""
4단계-3: 이중강건 추정 — "큰 회사일수록 경기에 민감할 뿐"이라는 대안 해석 배제
  주 모형(사전 확정): h=6, 매칭 후(1:3 가중), 법인·월 이중 클러스터
    대출증감률_h6 = 법인FE + 연도더미 + β1·수출YoY + β3·수출YoY×외환노출
                   + Σ γ_c·수출YoY×기준값_c   (기준값 5개: 첫 3개월 수신·운전자금(로그), 최우수, 우수, 거래기간)
    기준값은 노출 회사 평균을 0으로 맞춤(중심화) → β1 = "노출 회사 평균 조건을 가진 비노출 회사"의 반응
  보조 모형: ① 주 모형 + 수출YoY×업종 대분류  ② 매칭 전 전체 표본 + 기준값 교호항 5개(회귀 보정만)
  판정(사전 확정): 이중 클러스터 p<0.05 유의 / 0.05~0.10 약한 증거 / ≥0.10 근거 없음
  이중 확인: 주 모형 β3를 회사 더미 희소행렬 회귀로 다시 계산
입력: 패널(common.py), step3_psm_matched.parquet, step3_propensity.parquet
"""
import sys

import numpy as np
import pandas as pd

from common import (COV, HERE, XCOLS, add_cov_interactions, fe_reg, load_panel, lsdv_check,
                    match_weights, twoway_p)

sys.stdout.reconfigure(encoding="utf-8")

Y = "대출증감률_h6"

df = load_panel()
ps = pd.read_parquet(HERE / "step3_propensity.parquet").set_index("법인ID")
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))

# 기준값 중심화(노출 회사 평균 = 0) 후 수출YoY와 곱하기
df, INT_COV = add_cov_interactions(df, ps)

# 업종 대분류 교호항 (가장 많은 업종을 기준으로 두고 나머지 더미)
top = df["업종_대분류"].value_counts().index[0]
INT_IND = []
for v in sorted(df["업종_대분류"].dropna().unique()):
    if v == top:
        continue
    col = f"충격×업종[{v}]"
    df[col] = df["수출YoY"] * (df["업종_대분류"] == v)
    INT_IND.append(col)


def verdict(p):
    return "유의" if p < 0.05 else "약한 증거" if p < 0.10 else "근거 없음"


models = [
    ("기준(3-3, 교호항 없음)", W_MATCH, XCOLS),
    ("주 모형: 이중강건", W_MATCH, XCOLS + INT_COV),
    ("보조①: + 업종 대분류", W_MATCH, XCOLS + INT_COV + INT_IND),
    ("보조②: 매칭 전 + 회귀보정", None, XCOLS + INT_COV),
]
def usable(xcols, w):
    """그 표본에서 회사 안 변동이 전혀 없는 교호항(해당 업종 회사가 매칭 표본에 없음 등)은 뺀다"""
    d = df[df[Y].notna()]
    if w is not None:
        d = d[d["법인ID"].map(w).fillna(0) > 0]
    dm = d[xcols] - d.groupby("법인ID")[xcols].transform("mean")
    ok = [c for c in xcols if (dm[c] ** 2).sum() > 1e-10 or c in XCOLS]
    dropped = [c for c in xcols if c not in ok]
    return ok, dropped


rows, keep = [], {}
for name, w, xcols in models:
    xcols, dropped = usable(xcols, w)
    if dropped:
        print(f"[{name}] 표본에 없어 뺀 교호항: {dropped}")
    r = fe_reg(df, Y, w, xcols)
    se2, p2 = twoway_p(r, 1)
    rows.append({"모형": name, "교호항 수": len(xcols) - len(XCOLS), "회사": r["firms"],
                 "β1(비노출 반응)": r["b"][0], "β1+β3(노출 반응)": r["b"][0] + r["b"][1],
                 "β3": r["b"][1], "SE(이중)": se2, "p(이중)": p2, "p(법인)": r["법인ID"][1][1],
                 "판정": verdict(p2)})
    keep[name] = (r, xcols)

print(f"=== 4-3 결과 (h=6, 업종 대분류 기준 범주: {top}) ===")
with pd.option_context("display.width", 220, "display.max_columns", 20):
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# 주 모형 교호항: 크기·등급·거래기간이 수출 반응을 설명하는가 (대안 해석 (나) 점검)
r, xcols = keep["주 모형: 이중강건"]
print("\n=== 주 모형의 기준값 교호항 (크기가 반응을 설명하면 여기서 유의하게 나옴) ===")
sd = ps[COV].std()
for j, c in enumerate(xcols):
    if c in INT_COV:
        base = c.replace("충격×", "")
        _, p = twoway_p(r, j)
        print(f"{c:<18} γ={r['b'][j]:+.4f}  (기준값 1 표준편차당 {r['b'][j]*sd[base]:+.4f})  p(이중)={p:.3f}")

chk = lsdv_check(r, Y, xcols)
print(f"\n이중 확인 (주 모형): 평균빼기 β3={r['b'][1]:.5f} / 더미회귀 β3={chk:.5f} / 차이 {abs(r['b'][1]-chk):.1e}")
