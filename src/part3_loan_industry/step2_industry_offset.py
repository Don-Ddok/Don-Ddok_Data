# -*- coding: utf-8 -*-
"""
2단계: 업종별 부호 상쇄의 원인 — 법인 특성 비교
- 중간보고서 7-4절의 업종 분류(자금압박형 / 수요소멸형 / 자동차 비유의)를 그대로 가져와
  그 위에 법인 특성(규모·거래기간·고객등급)을 얹어 본다. 회귀를 다시 돌리지 않는다.
- 표본: 운전자금을 한 번이라도 받은 회사(운전자금_보유이력=1) — 팀 표7과 같은 기준
입력: step1_loan_industry_panel.parquet
출력: step2_firm_industry_groups.parquet  (법인ID·집단·특성만, 금액 없음 — 그래도 GitHub에 올리지 않는다)
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")

# 중간보고서 7-4절 업종 분류 (회귀는 팀이 수행, 여기서는 분류만 가져다 씀)
GROUP_KEYWORDS = {
    "자금압박형": ["전자 부품", "섬유제품", "기타 기계 및 장비", "금속 가공제품", "화학 물질", "식료품"],
    "수요소멸형": ["1차 금속"],
    "자동차(비유의)": ["자동차 및 트레일러"],
}
GROUPS = list(GROUP_KEYWORDS)


def classify(industry):
    if pd.isna(industry):
        return None
    for grp, kws in GROUP_KEYWORDS.items():
        if any(kw in industry for kw in kws):
            return grp
    return "기타 제조업"


# ── ① 고객등급이 회사 안에서 바뀌는가 (바뀌면 'first'가 대표값이 아님) ──
n_grade = df.groupby("법인ID")["법인_고객등급"].nunique()
print("=== ① 고객등급 변동 점검 ===")
print(f"36개월 동안 등급이 바뀐 회사: {(n_grade > 1).sum():,}개 / {len(n_grade):,}개 "
      f"({(n_grade > 1).mean()*100:.1f}%)")

# 대표 등급 = 가장 많이 관측된 등급(최빈값). 동률이면 더 높은 등급
RANK = {"최우수": 3, "우수": 2, "일반": 1}
g = (df.groupby(["법인ID", "법인_고객등급"]).size().rename("n").reset_index()
     .assign(r=lambda x: x["법인_고객등급"].map(RANK))
     .sort_values(["법인ID", "n", "r"], ascending=[True, False, False])
     .drop_duplicates("법인ID").set_index("법인ID")["법인_고객등급"])

# ── 회사 단위로 줄이기 ──────────────────────────────────────────
firm = (df.groupby("법인ID")
        .agg(KOSIS업종=("KOSIS업종", "first"), 규모분위=("규모분위", "first"),
             거래기간=("거래기간", "first"), 외환노출=("외환노출", "first"),
             보유이력=("운전자금_보유이력", "first"))
        .join(g.rename("고객등급")).reset_index())
firm["규모분위"] = firm["규모분위"].astype(int)
firm["집단"] = firm["KOSIS업종"].map(classify)

# ── ② 표본 한정 효과: 팀 표7 법인 수와 맞는지 ─────────────────────
print("\n=== ② 팀 표7 법인 수 대조 ===")
print(f"{'업종':<12}{'전체 회사':>10}{'보유이력 회사':>14}{'팀 표7':>8}")
for short, full, team in [("전자부품", "전자 부품", 165), ("화학물질", "화학 물질", 219),
                          ("자동차", "자동차 및 트레일러", 373)]:
    m = firm["KOSIS업종"].fillna("").str.contains(full)
    print(f"{short:<12}{m.sum():>10}{(m & (firm['보유이력'] == 1)).sum():>14}{team:>8}")

firm = firm[firm["보유이력"] == 1].copy()
print(f"\n분석 표본: 운전자금 보유이력 회사 {len(firm):,}개 "
      f"(그중 제조업 {firm['KOSIS업종'].notna().sum():,}개)")
print(firm["집단"].value_counts().reindex(GROUPS + ["기타 제조업"]).to_string())

# ── ③ 집단별 특성 ─────────────────────────────────────────────
target = firm[firm["집단"].isin(GROUPS)].copy()
summary = target.groupby("집단").agg(
    법인수=("법인ID", "count"),
    평균규모분위=("규모분위", "mean"),
    평균거래기간=("거래기간", "mean"),
    외환노출비율=("외환노출", "mean"),
    최우수비율=("고객등급", lambda s: (s == "최우수").mean()),
).reindex(GROUPS)
summary["외환노출비율"] *= 100
summary["최우수비율"] *= 100
print("\n=== ③ 집단별 법인 특성 ===")
print(summary.round(2).to_string())

print("\n규모분위 분포 (집단별, 행 합 100%)")
print((pd.crosstab(target["집단"], target["규모분위"], normalize="index") * 100)
      .reindex(GROUPS).round(1).to_string())

# ── ④ 검정: 자금압박형 vs 수요소멸형 ──────────────────────────────
print("\n=== ④ 자금압박형 vs 수요소멸형 검정 ===")
A = target[target["집단"] == "자금압박형"]
B = target[target["집단"] == "수요소멸형"]
for col in ["규모분위", "거래기간"]:
    t, p = stats.ttest_ind(A[col], B[col], equal_var=False)
    u, pu = stats.mannwhitneyu(A[col], B[col], alternative="two-sided")
    print(f"{col}: Welch t={t:.2f} (p={p:.4f}) / 순위검정 p={pu:.4f}")

ct = pd.crosstab(target.loc[target["집단"] != "자동차(비유의)", "집단"],
                 target.loc[target["집단"] != "자동차(비유의)", "고객등급"])
chi2, pc, dof, _ = stats.chi2_contingency(ct)
print(f"고객등급 분포: 카이제곱={chi2:.2f}, 자유도={dof}, p={pc:.4f}")

# ── ⑤ 이전 실행과 비교 ─────────────────────────────────────────
print("\n=== ⑤ 이전 실행(전체 회사, 행 단위 규모분위)과 비교 ===")
old = {"자금압박형": (2043, 3.24, 24.7, 42.6), "수요소멸형": (190, 3.78, 26.8, 60.5),
       "자동차(비유의)": (399, 3.59, 25.6, 55.1)}
print(f"{'집단':<14}{'법인수':>12}{'규모분위':>14}{'거래기간':>14}{'최우수%':>14}")
for grp in GROUPS:
    o, n = old[grp], summary.loc[grp]
    print(f"{grp:<14}{o[0]:>5} → {int(n['법인수']):<5}{o[1]:>6.2f} → {n['평균규모분위']:<5.2f}"
          f"{o[2]:>6.1f} → {n['평균거래기간']:<5.1f}{o[3]:>6.1f} → {n['최우수비율']:<5.1f}")
print("이전 검정: 규모분위 t=-5.50, 거래기간 t=-2.68")

out = HERE / "step2_firm_industry_groups.parquet"
firm[["법인ID", "KOSIS업종", "집단", "규모분위", "거래기간", "고객등급", "외환노출"]].to_parquet(
    out, index=False)
print(f"\n저장: {out.name} ({len(firm):,}개 회사)")
