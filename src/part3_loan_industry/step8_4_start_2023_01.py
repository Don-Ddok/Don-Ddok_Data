# -*- coding: utf-8 -*-
"""
8단계-4: 2023-01부터 관측된 회사만 — 출발 시점 차이 제거 (설계는 작업순서 8단계에 사전 확정)
  문제: "첫 3개월" 기준값의 달력 시점이 회사마다 다름(2023-01 관측 회사 70.9%)
  절차: 첫 관측월이 2023-01인 운전자금 보유 회사만 → 3-2와 같은 절차로 다시 매칭 → 3-3 식(h=6)
  이중 확인: 같은 함수로 전체 표본을 매칭하면 3-2 결과(step3_psm_matched.parquet)와 정확히 같은지
  판정: 균형(SMD<0.1) 확인 후, β3 부호 같고 크기가 −0.231의 절반 이상이면 "출발 시점 차이로 설명되지 않음"
입력: 패널(common.py), step3_psm_matched.parquet
"""
import sys

import numpy as np
import pandas as pd

from common import (COV, HERE, build_firm_table, fe_reg, load_panel, match_weights, psm_match,
                    twoway_p)

sys.stdout.reconfigure(encoding="utf-8")

Y = "대출증감률_h6"
MATCH_B3 = -0.231
df = load_panel()

# ── 이중 확인: 함수로 3-2 재현 ──────────────────────────────────────
firm_all, ind_all = build_firm_table(df)
pairs_all, _ = psm_match(firm_all, ind_all)
old = pd.read_parquet(HERE / "step3_psm_matched.parquet")
same = (pairs_all[["법인ID_노출", "법인ID_대조"]].reset_index(drop=True)
        .equals(old[["법인ID_노출", "법인ID_대조"]].reset_index(drop=True)))
print(f"이중 확인: 함수로 만든 전체 매칭 = 3-2 매칭 쌍 {len(old):,}개와 {'완전히 같음' if same else '다름'}")
assert same, "매칭 함수가 3-2를 재현하지 못함"

# ── 2023-01 관측 회사만 ───────────────────────────────────────────
keep_ids = firm_all.index[firm_all["첫관측"] == 202301]
sub = df[df["법인ID"].isin(keep_ids)]
firm, ind_cols = build_firm_table(sub)
T = firm["외환노출"] == 1
print(f"\n표본: 2023-01부터 관측된 운전자금 보유 회사 {len(firm):,}개 / {len(firm_all):,}개 "
      f"(노출 {int(T.sum())} / 비노출 {int((~T).sum())}), 업종 그룹 {firm['업종'].nunique()}개")

pairs, ps = psm_match(firm, ind_cols)
w = match_weights(pairs)
print(f"매칭: 노출 {pairs['법인ID_노출'].nunique()}곳 × 3, 대조 {pairs['법인ID_대조'].nunique()}곳, "
      f"공통지지 밖 {int(T.sum()) - pairs['법인ID_노출'].nunique()}곳")

print("\n균형 (SMD, 분모는 매칭 전 분산)")
wt = w.reindex(firm.index).fillna(0)
for c in COV + ind_cols:
    x = firm[c].astype(float)
    sd = np.sqrt((x[T].var() + x[~T].var()) / 2)
    before = (x[T].mean() - x[~T].mean()) / sd
    after = (np.average(x[T], weights=wt[T]) - np.average(x[~T], weights=wt[~T])) / sd
    if c in COV:
        print(f"  {c:<12} 매칭 전 {before:+.3f} → 후 {after:+.3f}")
ia = [abs((np.average(firm[c][T], weights=wt[T]) - np.average(firm[c][~T], weights=wt[~T]))
          / np.sqrt((firm[c][T].var() + firm[c][~T].var()) / 2)) for c in ind_cols]
print(f"  업종 더미 {len(ind_cols)}개: 매칭 후 |SMD| 최대 {max(ia):.3f}")

r = fe_reg(sub, Y, w)
_, p3 = twoway_p(r, 1)
b1, b3 = r["b"][0], r["b"][1]
print(f"\n=== 8-4 결과 (h=6) ===")
print(f"회사 {r['firms']:,}, β1(비노출) {b1:+.3f}, β1+β3(노출) {b1 + b3:+.3f}, "
      f"β3 {b3:+.3f} (매칭 전체 표본 대비 {b3 / MATCH_B3 * 100:.0f}%), p(이중) {p3:.3f}")
ok = (b3 < 0) and (abs(b3) >= abs(MATCH_B3) / 2)
print(f"판정 (사전 기준): {'출발 시점 차이로 설명되지 않음' if ok else '출발 시점 차이와 분리되지 않음'}")
