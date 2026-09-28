# -*- coding: utf-8 -*-
"""
9단계-3: 수출 회사만 처치 — 수입만 한 회사를 빼고 봐도 같은가
  (설계는 06_추가분석_설계와_결과.md 1절, 2026-09-23 실행 전 확정)
  표본: 운전자금 보유 회사 중 수입만 한 회사 제외(처치에도 대조에도 넣지 않음)
  처치: 수출노출(외환 수출실적 1회 이상), 비교: 외환 실적이 아예 없는 회사(진짜 비노출)
  절차: common.psm_match로 다시 매칭(같은 공변량, 1:3 복원) → 균형 확인 → 3-3 식(h=6)
  주 검정: β3 (Holm 보정은 9-1~9-4가 모두 끝난 뒤 한다)
  9-3·9-4는 강건성 점검이기도 하므로 8단계 기준(부호 같고 크기 ≥ −0.231의 절반 → "유지")도 함께 적는다
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

# ── 이중 확인: 함수로 3-2(전체 외환노출 기준) 재현 ─────────────────
firm_all, ind_all = build_firm_table(df)
pairs_all, _ = psm_match(firm_all, ind_all)
old = pd.read_parquet(HERE / "step3_psm_matched.parquet")
same = (pairs_all[["법인ID_노출", "법인ID_대조"]].reset_index(drop=True)
        .equals(old[["법인ID_노출", "법인ID_대조"]].reset_index(drop=True)))
print(f"이중 확인: 함수로 만든 전체 매칭 = 3-2 매칭 쌍 {len(old):,}개와 {'완전히 같음' if same else '다름'}")
assert same, "매칭 함수가 3-2를 재현하지 못함"

# ── 수입만 한 회사 제외 (외환노출=1인데 수출노출=0) ────────────────
imp_only = firm_all.index[(firm_all["외환노출"] == 1) & (df.groupby("법인ID")["수출노출"].first().reindex(firm_all.index) == 0)]
n_before = len(firm_all)
print(f"\n수입만 한 회사(외환노출 1, 수출노출 0): {len(imp_only):,}곳 제외")

sub = df[~df["법인ID"].isin(imp_only)].copy()
sub["외환노출"] = sub["수출노출"]           # 이 표본에서는 "노출" = "수출노출"로 재정의
sub["교호"] = sub["수출YoY"] * sub["외환노출"]

firm, ind_cols = build_firm_table(sub)
T = firm["외환노출"] == 1
print(f"표본: {len(firm):,}곳 / 원래 {n_before:,}곳 (수출노출 {int(T.sum())} / 진짜 비노출 {int((~T).sum())}), "
      f"업종 그룹 {firm['업종'].nunique()}개")
print(f"참고: 전체 법인(운전자금 보유 여부 무관) 기준 수출노출은 523곳(CLAUDE.md) — "
      f"여기서는 운전자금 보유 회사로 좁혀서 {int(T.sum())}곳")

pairs, ps = psm_match(firm, ind_cols)
w = match_weights(pairs)
print(f"매칭: 수출노출 {pairs['법인ID_노출'].nunique()}곳 × 3, 대조 {pairs['법인ID_대조'].nunique()}곳, "
      f"공통지지 밖 {int(T.sum()) - pairs['법인ID_노출'].nunique()}곳")

print("\n균형 (SMD, 분모는 매칭 전 분산)")
wt = w.reindex(firm.index).fillna(0)
smd_rows = []
for c in COV + ind_cols:
    x = firm[c].astype(float)
    sd = np.sqrt((x[T].var() + x[~T].var()) / 2)
    before = (x[T].mean() - x[~T].mean()) / sd
    after = (np.average(x[T], weights=wt[T]) - np.average(x[~T], weights=wt[~T])) / sd
    smd_rows.append((c, before, after))
    if c in COV:
        print(f"  {c:<12} 매칭 전 {before:+.3f} → 후 {after:+.3f}")
ia = [abs(a) for c, b, a in smd_rows if c in ind_cols]
print(f"  업종 더미 {len(ind_cols)}개: 매칭 후 |SMD| 최대 {max(ia):.3f}")
balance_ok = max(abs(a) for _, _, a in smd_rows) < 0.1
print(f"균형 판정(모든 변수 |SMD|<0.1): {'통과' if balance_ok else '미달'}")

r = fe_reg(sub, Y, w)
_, p3 = twoway_p(r, 1)
b1, b3 = r["b"][0], r["b"][1]
print(f"\n=== 9-3 결과 (h=6) ===")
print(f"회사 {r['firms']:,}, 월 클러스터 {r['ym'][2]}, β1(비노출) {b1:+.3f}, β1+β3(노출) {b1 + b3:+.3f}, "
      f"β3 {b3:+.3f} (매칭 기준 대비 {b3 / MATCH_B3 * 100:.0f}%), p(이중, 보정 전) {p3:.4f}")
print(f"수출 10%p 하락 시 수출노출 회사의 6개월 대출 추가 변화: {-0.1 * b3 * 100:+.1f}%")

ok = (b3 < 0) and (abs(b3) >= abs(MATCH_B3) / 2)
print(f"\n강건성 판정(사전 기준, 부호 같고 |β3|≥기준의 절반): "
      f"{'수출 회사만으로도 유지' if ok else '수출 회사만으로는 분리되지 않음'}")
print("Holm 보정 판정은 9-1~9-4를 모두 돌린 뒤 별도로 한다.")
