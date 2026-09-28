# -*- coding: utf-8 -*-
"""
9단계-4: 균형 패널 — 36개월 모두 관측된 회사만으로도 같은가
  (설계는 06_추가분석_설계와_결과.md 1절, 2026-09-23 실행 전 확정)
  표본: 운전자금 보유 회사 중 36개월 모두 관측된 회사
  처치: 외환노출(3-3과 동일, 정의는 바꾸지 않음)
  절차: common.psm_match로 다시 매칭 → 균형 확인 → 3-3 식(h=6)
  주 검정: β3 (Holm 보정은 9-1~9-4가 모두 끝난 뒤 한다)
  9-3·9-4는 강건성 점검이기도 하므로 8단계 기준(부호 같고 크기 ≥ −0.231의 절반 → "유지")도 함께 적는다
  미리 적어 둘 한계: 균형 패널의 노출 회사가 적어(약 190곳) 추정이 부정확할 수 있음
입력: 패널(common.py), step3_psm_matched.parquet(이중 확인용)
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

# ── 36개월 모두 관측된 회사만 ───────────────────────────────────────
n_months = df.groupby("법인ID")["ym"].nunique()
keep_ids = n_months.index[n_months == 36]
n_before = df["법인ID"].nunique()
print(f"\n36개월 모두 관측된 회사: {len(keep_ids):,}곳 / 운전자금 보유 회사 {n_before:,}곳")

sub = df[df["법인ID"].isin(keep_ids)].copy()
firm, ind_cols = build_firm_table(sub)
T = firm["외환노출"] == 1
print(f"표본: {len(firm):,}곳 (노출 {int(T.sum())} / 비노출 {int((~T).sum())}), 업종 그룹 {firm['업종'].nunique()}개")

pairs, ps = psm_match(firm, ind_cols)
w = match_weights(pairs)
print(f"매칭: 노출 {pairs['법인ID_노출'].nunique()}곳 × 3, 대조 {pairs['법인ID_대조'].nunique()}곳, "
      f"공통지지 밖 {int(T.sum()) - pairs['법인ID_노출'].nunique()}곳")

print("\n균형 (SMD, 분모는 매칭 전 분산)")
wt = w.reindex(firm.index).fillna(0)
smd_rows = []
for c in COV + ind_cols:
    x = firm[c].astype(float)
    if x.var() == 0:
        print(f"  {c:<12} 이 표본에서는 값이 전부 같음(분산 0) → SMD 계산 불가, 편향 원천 없음(구조적으로 균형)")
        continue
    sd = np.sqrt((x[T].var() + x[~T].var()) / 2)
    before = (x[T].mean() - x[~T].mean()) / sd
    after = (np.average(x[T], weights=wt[T]) - np.average(x[~T], weights=wt[~T])) / sd
    smd_rows.append((c, before, after))
    if c in COV:
        print(f"  {c:<12} 매칭 전 {before:+.3f} → 후 {after:+.3f}")
ia = [abs(a) for c, b, a in smd_rows if c in ind_cols]
print(f"  업종 더미 {len(ia)}개(값 일정한 것 제외): 매칭 후 |SMD| 최대 {max(ia):.3f}")
after_vals = [abs(a) for _, _, a in smd_rows]  # 분산 0인 변수는 이미 제외됨(NaN 없음)
balance_ok = max(after_vals) < 0.1
print(f"균형 판정(분산이 있는 변수 전부 |SMD|<0.1): {'통과' if balance_ok else '미달'} (최댓값 {max(after_vals):.3f})")

r = fe_reg(sub, Y, w)
_, p3 = twoway_p(r, 1)
b1, b3 = r["b"][0], r["b"][1]
print(f"\n=== 9-4 결과 (h=6) ===")
print(f"회사 {r['firms']:,}, 월 클러스터 {r['ym'][2]}, β1(비노출) {b1:+.3f}, β1+β3(노출) {b1 + b3:+.3f}, "
      f"β3 {b3:+.3f} (매칭 기준 대비 {b3 / MATCH_B3 * 100:.0f}%), p(이중, 보정 전) {p3:.4f}")
print(f"수출 10%p 하락 시 균형 패널 노출 회사의 6개월 대출 추가 변화: {-0.1 * b3 * 100:+.1f}%")

ok = (b3 < 0) and (abs(b3) >= abs(MATCH_B3) / 2)
print(f"\n강건성 판정(사전 기준, 부호 같고 |β3|≥기준의 절반): "
      f"{'균형 패널로도 유지' if ok else '균형 패널에서는 분리되지 않음'}")
print("Holm 보정 판정은 9-1~9-4를 모두 돌린 뒤 별도로 한다.")
