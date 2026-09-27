# -*- coding: utf-8 -*-
"""
6단계-2: 업종별 불황·호황 차이 (검정 없음, 서술용. 설계는 작업순서 6단계에 사전 확정)
  지표: Δ↑ = P(대출↑ | 불황) − P(대출↑ | 호황),  Δ↓ = P(대출↓ | 불황) − P(대출↓ | 호황)
        0이면 그 업종의 대출 방향은 업황과 무관
  대상: 제조업 업종 중 운전자금 보유 회사 30곳 이상
  2단계 연결: 팀 회귀의 자금압박형 6개 업종 → Δ↑>0 기대 / 1차 금속(수요소멸형) → Δ↓>0 기대
  강건성: 문턱 ±3·5·10%, 관측창 3·6개월에서 업종별 Δ↑ 순위 유지 여부(스피어만)
입력: step1_loan_industry_panel.parquet
출력: step6_2_industry_table.csv (업종 단위 집계표 — 회사 단위 값 없음)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
MIN_FIRMS = 30
GROUP_KEYWORDS = {   # 2단계와 동일 (팀 중간보고서 7-4절)
    "자금압박형": ["전자 부품", "섬유제품", "기타 기계 및 장비", "금속 가공제품", "화학 물질", "식료품"],
    "수요소멸형": ["1차 금속"],
    "자동차(비유의)": ["자동차 및 트레일러"],
}


def team_group(ind):
    for g, kws in GROUP_KEYWORDS.items():
        if any(k in ind for k in kws):
            return g
    return ""


df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
s = df[df["KOSIS업종"].notna() & (df["운전자금_보유이력"] == 1) & df["수요환경"].notna()].copy()
n_firm = s.groupby("KOSIS업종")["법인ID"].nunique()
inds = n_firm[n_firm >= MIN_FIRMS].index
print(f"업종 {s['KOSIS업종'].nunique()}개 중 보유 회사 {MIN_FIRMS}곳 이상: {len(inds)}개")


def direction(lr, thr):
    pct = np.exp(lr) - 1
    return np.where(lr.isna(), None, np.where(pct > thr, "↑", np.where(pct < -thr, "↓", "→")))


def deltas(h, thr):
    d = s[s["KOSIS업종"].isin(inds)].copy()
    d["dir"] = direction(d[f"대출증감률_h{h}"], thr)
    d = d[d["dir"].notna()]
    out = {}
    for ind, g in d.groupby("KOSIS업종"):
        bad, good = g[g["수요환경"] == "↓"], g[g["수요환경"] == "↑"]
        out[ind] = {
            "회사": g["법인ID"].nunique(), "불황 행": len(bad), "호황 행": len(good),
            "무변동%": (g["dir"] == "→").mean() * 100,
            "↑|불황%": (bad["dir"] == "↑").mean() * 100, "↑|호황%": (good["dir"] == "↑").mean() * 100,
            "↓|불황%": (bad["dir"] == "↓").mean() * 100, "↓|호황%": (good["dir"] == "↓").mean() * 100,
        }
    t = pd.DataFrame(out).T
    t["Δ↑"] = t["↑|불황%"] - t["↑|호황%"]
    t["Δ↓"] = t["↓|불황%"] - t["↓|호황%"]
    return t


main = deltas(6, 0.05)
main["팀 분류"] = [team_group(i) for i in main.index]
main.index = [i.replace(" 제조업", "").split(";")[0] for i in main.index]
main = main.sort_values("Δ↑", ascending=False)
cols = ["팀 분류", "회사", "불황 행", "호황 행", "무변동%", "↑|불황%", "↑|호황%", "Δ↑", "↓|불황%", "↓|호황%", "Δ↓"]
print("\n=== 업종별 불황·호황 차이 (h=6, ±5%, 행 기준, Δ 단위 %p) ===")
with pd.option_context("display.width", 240, "display.max_columns", 20):
    print(main[cols].astype({c: float for c in cols if c != "팀 분류"}).round(1).to_string())

# 2단계 연결
print("\n=== 팀 회귀 분류와 대조 (서술, 검정 없음) ===")
for g, key, want in [("자금압박형", "Δ↑", "Δ↑>0"), ("수요소멸형", "Δ↓", "Δ↓>0"), ("자동차(비유의)", "Δ↑", "참고")]:
    sub = main[main["팀 분류"] == g]
    vals = ", ".join(f"{i} {v:+.1f}" for i, v in sub[key].items())
    hit = int((sub[key] > 0).sum())
    print(f"{g} ({want}): {hit}/{len(sub)}개 일치 — {vals}")
others = main[main["팀 분류"] == ""]["Δ↑"]
print(f"팀 분류에 없는 업종 Δ↑: 중앙값 {others.median():+.1f}, 범위 {others.min():+.1f} ~ {others.max():+.1f}")

# 강건성: 문턱·관측창을 바꿔도 Δ↑ 순위가 유지되는가
print("\n=== 강건성: 업종별 Δ↑ 순위 상관 (기준 h=6·±5%) ===")
base = deltas(6, 0.05)["Δ↑"]
for h in (3, 6):
    for thr in (0.03, 0.05, 0.10):
        if (h, thr) == (6, 0.05):
            continue
        alt = deltas(h, thr)["Δ↑"]
        rho = base.rank().corr(alt.reindex(base.index).rank())
        print(f"h={h}, ±{int(thr*100)}%: 스피어만 {rho:+.2f} / Δ↑ 부호가 같은 업종 "
              f"{int((np.sign(alt.reindex(base.index)) == np.sign(base)).sum())}/{len(base)}")

main[cols].to_csv(HERE / "step6_2_industry_table.csv", encoding="utf-8-sig")
print("\n저장: step6_2_industry_table.csv (업종 집계표)")
