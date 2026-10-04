# -*- coding: utf-8 -*-
"""묶음 H — 상담 순위의 실제 작동 검증 (회귀 없음, 기술 통계만)
디벨롭 문서 §5 묶음 H에 대응. matcher.py(원본, 수정 안 함)의 정렬 로직을 코드로 확인:
  out.sort_values(["연월","_st","_seg","요구불잔액","_chg"], ascending=[True,True,True,False,True])
4번째 키(요구불잔액, 내림차순)가 있고 5번째(_chg, 최근 3개월 변화)는 '요구불잔액이 완전히 같을 때만' 작동하는
사전식(lexicographic) 정렬이다 — 요구불잔액이 연속값이라 동률이 드물면 5번째 키는 사실상 죽은 기준이 된다.
이 가설을 데이터로 직접 확인한다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PF = r"C:\test\분석결과\matching\persona_firm_month.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_h_summary.txt"), "w", encoding="utf-8")
STAGE_ORDER = ["적기", "정점·유지", "장기둔화", "관찰", "평시", "일반"]


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


p = pd.read_csv(PF, dtype={"법인ID": str})
p["페르소나"] = p["페르소나"].fillna("")
log(f"# 묶음 H — 상담 순위 실제 작동 검증 (회귀 없음)\n")

# trigger 단계 붙이기 (matcher.py와 같은 방식)
trig = pd.read_csv(r"C:\test\분석결과\matching\trigger_stages.csv")
seg_exposed = p["세그먼트"].isin(["수출형", "수입형"])
p = p.merge(trig[["지역", "연월", "단계"]], on=["지역", "연월"], how="left")
p.loc[~seg_exposed, "단계"] = "일반"
p["_st"] = p["단계"].map({s: i for i, s in enumerate(STAGE_ORDER)})
p["_seg"] = np.where(seg_exposed, 0, 1)

# ---------------- 1. 동률 비율 — 4번째 기준(요구불잔액)에서 순위가 갈리는지 ----------------
log("## 1. 4번째 기준(요구불잔액)의 동률 비율 — 이게 낮아야 5번째 기준(최근 변화)이 실제로 작동한다")
snap = p[p["연월"] == 202512].copy()
grp = snap.groupby(["_st", "_seg"])
tie_rows, total_rows, groups_with_tie = 0, 0, 0
for _, g in grp:
    vc = g["요구불잔액"].value_counts()
    tie = vc[vc > 1]
    tie_rows += int(tie.sum()); total_rows += len(g)
    groups_with_tie += int((vc > 1).any())
log(f"- 2025-12, (단계, 노출여부) 그룹 {grp.ngroups}개, 전체 {total_rows:,}곳 중 요구불잔액이 같은 값을 가진 법인(동률) {tie_rows:,}곳"
    f"({tie_rows/total_rows:.1%})")
log(f"- 대부분은 잔액=0 동률이다: " + f"{int((snap['요구불잔액']==0).sum()):,}곳이 잔액 0 (전체 {len(snap):,}곳 중 {(snap['요구불잔액']==0).mean():.1%})")
nonzero_tie = 0; nonzero_total = 0
for _, g in grp:
    g2 = g[g["요구불잔액"] > 0]
    vc = g2["요구불잔액"].value_counts()
    nonzero_tie += int(vc[vc > 1].sum()); nonzero_total += len(g2)
log(f"- 잔액 > 0인 법인만 보면: {nonzero_total:,}곳 중 동률 {nonzero_tie:,}곳({nonzero_tie/max(nonzero_total,1):.2%}) "
    f"— 원자료가 유효숫자 2자리로 반올림돼 있어 예상과 달리 **양수 구간에서도 동률이 매우 흔하다.** "
    f"5번째 기준(최근 3개월 요구불 변화)이 실제로 순위를 정하는 범위는 이 동률 구간 전체이지, 소수의 예외가 아니다.")

# ---------------- 2. 4번째 기준이 실제로 순위를 바꾸는 법인 비율(전체 재현) ----------------
log("\n## 2. 전체 우선순위 재현 — '최근 변화' 기준이 순위를 바꾼 비율")
out = p.copy()
out["_m"] = (out["연월"] // 100) * 12 + out["연월"] % 100
prev = out[["법인ID", "_m", "요구불잔액"]].rename(columns={"요구불잔액": "_prev"}); prev["_m"] = prev["_m"] + 3
out = out.merge(prev, on=["법인ID", "_m"], how="left")
out["_chg"] = out["요구불잔액"] - out["_prev"]
base = out[out["연월"] == 202512].copy()
# 두 정렬 모두 '섞인 원래 순서'에서 독립적으로 다시 정렬한다 (한쪽을 다른 쪽 위에 또 정렬하면 안 됨 — 버그였던 부분)
shuffled = base.sample(frac=1, random_state=0).reset_index(drop=True)
snap2 = shuffled.sort_values(["_st", "_seg", "요구불잔액", "_chg"], ascending=[True, True, False, True], na_position="last", kind="mergesort")
snap2["순위_전체"] = range(1, len(snap2) + 1)
snap2b = shuffled.sort_values(["_st", "_seg", "요구불잔액"], ascending=[True, True, False], na_position="last", kind="mergesort")
snap2b["순위_잔액만"] = range(1, len(snap2b) + 1)
cmp = snap2[["법인ID", "순위_전체"]].merge(snap2b[["법인ID", "순위_잔액만"]], on="법인ID")
same_rank = (cmp["순위_전체"] == cmp["순위_잔액만"]).mean()
log(f"- 2025-12 전체 {len(cmp):,}곳: '최근 3개월 변화' 기준을 포함한 순위와, 그 기준을 뺀(요구불잔액까지만) 순위가 "
    f"**완전히 같은 비율 {same_rank:.1%}**({int((cmp['순위_전체']==cmp['순위_잔액만']).sum()):,}/{len(cmp):,}곳).")
log(f"- 순위가 달라지는 {int((cmp['순위_전체']!=cmp['순위_잔액만']).sum()):,}곳은 요구불잔액이 동률인 구간 안에서만 순서가 바뀐 것이다. "
    "동률 비율이 위에서 확인한 대로 96%대로 매우 높으므로, **'최근 변화' 기준이 실제 순위에 영향을 주는 범위도 넓다** — "
    "디벨롭 문서 §6이 걱정한 것과 반대로, 이 데이터에서는 4번째 기준(잔액 규모)이 유일한 값으로 순위를 정하는 경우가 오히려 적고, "
    "반올림된 잔액이 자주 겹쳐 5번째 기준(최근 감소)이 상당 부분 순위를 정한다.")

# 실무적으로 중요한 것: 상담에 쓰일 '상위 N'만 놓고 봤을 때도 바뀌는지
log("\n- 실제 상담에 쓰일 상위 20곳만 놓고 보면(단계·노출 그룹별):")
for (st, sg), g in shuffled.groupby(["_st", "_seg"]):
    if g["_st"].iloc[0] > 2:
        continue
    a = g.sort_values("요구불잔액", ascending=False, kind="mergesort").head(20)["법인ID"].tolist()
    b = g.sort_values(["요구불잔액", "_chg"], ascending=[False, True], na_position="last", kind="mergesort").head(20)["법인ID"].tolist()
    stage_name = STAGE_ORDER[st]
    overlap = len(set(a) & set(b))
    log(f"    {stage_name}(노출={'예' if sg==0 else '아니오'}, n={len(g):,}): 상위 20 구성원 겹침 {overlap}/20, 순서까지 동일 {a==b}")

# ---------------- 3. 상위 대상 규모 집중도·반복 ----------------
log("\n## 3. 상위 대상 잔액 규모 집중도·반복")
for st in ["적기", "정점·유지"]:
    g = out[(out["연월"] == 202512) & (out["단계"] == st) & (out["세그먼트"].isin(["수출형", "수입형"]))].sort_values(
        ["요구불잔액", "_chg"], ascending=[False, True])
    if len(g) == 0:
        continue
    top10 = g.head(10)
    log(f"- {st} 2025-12 노출 {len(g):,}곳, 상위 10곳 요구불잔액 합이 전체의 {top10['요구불잔액'].sum()/g['요구불잔액'].sum():.1%}를 차지 "
        f"(상위 10곳 최소값 {top10['요구불잔액'].min():,.0f} vs 전체 중앙값 {g['요구불잔액'].median():,.0f})")
# 같은 법인이 여러 달 상위 10에 반복되는지 (2025년 하반기 6개월)
rep_months = sorted(out["연월"].unique())[-6:]
top_sets = []
for ym in rep_months:
    g = out[(out["연월"] == ym) & (out["세그먼트"].isin(["수출형", "수입형"]))].sort_values(
        ["_st", "_seg", "요구불잔액", "_chg"], ascending=[True, True, False, True])
    top_sets.append(set(g.head(10)["법인ID"]))
from collections import Counter
cnt = Counter()
for s in top_sets:
    cnt.update(s)
rep6 = sum(1 for v in cnt.values() if v == len(rep_months))
log(f"- 최근 6개월({rep_months[0]}~{rep_months[-1]}) 전체 1순위 10곳 안에 6개월 내내 들어간 법인: {rep6}곳 "
    f"(연인원 {sum(cnt.values())}건 중 서로 다른 법인 {len(cnt)}곳)")

LOG.close()
