# -*- coding: utf-8 -*-
"""최종 확정 전 검증: 순위 점수화 + 페르소나 히스테리시스가 규칙대로 동작하는지
recommend_firm_month.csv·persona_firm_month.csv의 실제 우선순위순번·페르소나를,
match_rules.yaml의 공식으로 독립적으로 재계산한 값과 직접 대조한다. 새 회귀 없음.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\test\분석결과\matching"
RULES = yaml.safe_load(open(r"C:\test\code\matching\match_rules.yaml", encoding="utf-8"))
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "step1_verify_summary.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


log("# 최종 확정 전 검증 — 순위 점수화 + 페르소나 히스테리시스\n")

# ============================================================
# A. 순위 점수화 검증
# ============================================================
log("## A. 순위 점수화 (0.7×규모 + 0.3×추세) 독립 재계산 대조")
r = pd.read_csv(os.path.join(BASE, "recommend_firm_month.csv"),
                usecols=["법인ID", "연월", "단계", "세그먼트", "우선순위순번"], dtype={"법인ID": str}, low_memory=False)
p = pd.read_csv(os.path.join(BASE, "persona_firm_month.csv"), usecols=["법인ID", "연월", "요구불잔액"], dtype={"법인ID": str})
d = r.merge(p, on=["법인ID", "연월"], how="left")
d["_m"] = (d["연월"] // 100) * 12 + d["연월"] % 100

pr = RULES["priority"]
n = pr["demand_change_months"]
w = pr["weights"]; miss = pr["missing_momentum_percentile"]
assert pr["method"] == "score", "priority.method이 score가 아님 — 검증 전제가 깨짐"
log(f"- 규칙 확인: method={pr['method']}, weights={w}, demand_change_months={n}, missing_momentum_percentile={miss}")

prev = d[["법인ID", "_m", "요구불잔액"]].rename(columns={"요구불잔액": "_prev"})
prev["_m"] = prev["_m"] + n
d = d.merge(prev, on=["법인ID", "_m"], how="left")
d["_chg"] = d["요구불잔액"] - d["_prev"]
stage_order = pr["stage_order"]
d["_st"] = d["단계"].map({s: i for i, s in enumerate(stage_order)})
d["_seg"] = np.where(d["세그먼트"].isin(RULES["segments"]["exposed"]), 0, 1)

grp = d.groupby(["연월", "_st", "_seg"])
d["_size_pct"] = grp["요구불잔액"].transform(lambda s: s.rank(pct=True, ascending=True))
d["_mom_pct"] = grp["_chg"].transform(lambda s: (-s).rank(pct=True, ascending=True)).fillna(miss)
d["_score_recalc"] = w["size"] * d["_size_pct"] + w["momentum"] * d["_mom_pct"]

# 그룹 안에서 재계산한 점수로 다시 매긴 순위 vs 파일의 우선순위순번(그룹 안에서의 상대 순서로 환산)
d["_rank_recalc_in_group"] = grp["_score_recalc"].rank(ascending=False, method="first")
d["_rank_file_in_group"] = d.groupby(["연월", "_st", "_seg"])["우선순위순번"].rank(method="first")
mismatch = (d["_rank_recalc_in_group"] != d["_rank_file_in_group"])
log(f"- 전체 {len(d):,}행 중 그룹(연월×단계×노출) 안에서 재계산 순위와 파일 순위가 다른 행: {int(mismatch.sum()):,}행 ({mismatch.mean():.4%})")
if mismatch.any():
    ex = d[mismatch].head(5)[["법인ID", "연월", "단계", "요구불잔액", "_chg", "_score_recalc", "_rank_recalc_in_group", "_rank_file_in_group"]]
    log("  불일치 예시:\n" + ex.to_string(index=False))
    log("  (동점 처리 방식 차이일 수 있음 — 아래에서 원인 구분)")
else:
    log("  → **완전히 일치한다. 0.7/0.3 가중치가 정확히 반영돼 있다.**")

# 가중치가 실제로 둘 다 쓰이는지: size만 썼을 때/momentum만 썼을 때와 실제 순위의 차이 비교
d["_rank_size_only"] = grp["_size_pct"].transform(lambda s: s.rank(ascending=False, method="first"))
d["_rank_mom_only"] = grp["_mom_pct"].transform(lambda s: s.rank(ascending=False, method="first"))
same_as_size = (d["_rank_file_in_group"] == d["_rank_size_only"]).mean()
same_as_mom = (d["_rank_file_in_group"] == d["_rank_mom_only"]).mean()
log(f"- 참고: 파일 순위가 '규모만'으로 정렬한 것과 같은 비율 {same_as_size:.1%}, '추세만'으로 정렬한 것과 같은 비율 {same_as_mom:.1%} "
    f"(둘 다 100%가 아니어야 두 요소가 실제로 섞여 반영된 것)")

# 2025-12 정점·유지 상위 10 실사례로 '규모가 크지만 추세가 나빠 순위가 밀린' 경우가 있는지 확인
snap = d[(d["연월"] == 202512) & (d["단계"] == "정점·유지") & (d["_seg"] == 0)].sort_values("우선순위순번")
log("\n- 2025-12 정점·유지 상위 12곳 (요구불잔액 / 최근3개월변화 / 규모백분위 / 추세백분위 / 점수 / 순위):")
log(snap.head(12)[["요구불잔액", "_chg", "_size_pct", "_mom_pct", "_score_recalc", "우선순위순번"]].round(3).to_string(index=False))

# ============================================================
# B. 페르소나 히스테리시스 검증
# ============================================================
log("\n## B. 페르소나 히스테리시스 (2개월 연속 조건) 구조적 검증")
pp = pd.read_csv(os.path.join(BASE, "persona_firm_month.csv"),
                 usecols=["법인ID", "연월", "페르소나", "페르소나_원본"], dtype={"법인ID": str})
for c in ["페르소나", "페르소나_원본"]:
    pp[c] = pp[c].fillna("")
pp["_m"] = (pp["연월"] // 100) * 12 + pp["연월"] % 100
pp = pp.sort_values(["법인ID", "_m"]).reset_index(drop=True)

fid = pp["법인ID"].to_numpy(); m = pp["_m"].to_numpy()
stable = pp["페르소나"].to_numpy(); raw = pp["페르소나_원본"].to_numpy()
prev_fid = np.roll(fid, 1); prev_m = np.roll(m, 1); prev_stable = np.roll(stable, 1); prev_raw = np.roll(raw, 1)
same_firm_consec = (fid == prev_fid) & (m == prev_m + 1)
# 히스테리시스 2개월 조건은 '노출 상태에서 다른 페르소나로 바뀌는' 경우에만 적용된다.
# 비노출로 빠지는 전환(stable이 ""로 바뀜)과 신규 편입(prev_stable이 "")은 설계상 즉시 반영 예외라 제외한다.
changed = same_firm_consec & (stable != prev_stable) & (prev_stable != "") & (stable != "")
# 규칙상 바뀐 시점 t는: raw[t] == raw[t-1] (2개월 연속 원본이 새 값과 같아야 함), 그리고 raw[t]==stable[t](바뀐 결과와 원본이 같아야 함)
ok = (raw == prev_raw) & (raw == stable)
violate = changed & ~ok
log(f"- 노출 상태에서 주 페르소나가 바뀐 사례(신규 편입 제외) {int(changed.sum()):,}건 중, "
    f"'원본이 2개월 연속 새 값과 같다'는 조건을 어긴 사례: {int(violate.sum()):,}건 "
    f"({'없음 — 규칙이 코드대로 정확히 지켜졌다' if violate.sum()==0 else '있음 — 확인 필요'})")
if violate.any():
    idx = np.where(violate)[0][:8]
    exdf = pd.DataFrame({"법인ID": fid[idx], "_m": m[idx], "원본_t-1": prev_raw[idx], "원본_t": raw[idx],
                         "안정_t-1": prev_stable[idx], "안정_t": stable[idx]})
    log("  위반 예시:\n" + exdf.to_string(index=False))
    # 각 사례의 앞뒤 2~3행을 더 보여줘서 실제 흐름 확인
    for j in idx[:3]:
        f_ = fid[j]
        ctx = pp[(pp["법인ID"] == f_) & (pp["_m"].between(m[j] - 3, m[j] + 1))]
        log(f"  --- 법인 컨텍스트 (전환 시점 _m={m[j]}) ---\n" + ctx[["_m", "페르소나_원본", "페르소나"]].to_string(index=False))

# 단기 왕복(A→B→A, 2개월 이내) 잔존 여부: 안정화된 '페르소나' 기준으로 재검사
nxt_m = np.roll(m, -1); nxt_fid = np.roll(fid, -1); nxt_stable = np.roll(stable, -1)
n2_m = np.roll(m, -2); n2_fid = np.roll(fid, -2); n2_stable = np.roll(stable, -2)
seq_ok = (fid == nxt_fid) & (nxt_m == m + 1) & (fid == n2_fid) & (n2_m == m + 2)
switched = seq_ok & (stable != nxt_stable) & (stable != "") & (nxt_stable != "")
bounce = switched & (n2_stable == stable)
log(f"- 안정화된 페르소나 기준 단기 왕복(t→t+1 전환, t+2에 원래대로): 전환 {int(switched.sum()):,}건 중 왕복 {int(bounce.sum()):,}건 "
    f"({bounce.sum()/max(switched.sum(),1):.2%}) — 0%에 가까워야 함")

# 원본 판정에는 단기 왕복이 여전히 존재하는지(대조군, 안정화가 이걸 걸러낸다는 증거)
raw_nxt = np.roll(raw, -1); raw_n2 = np.roll(raw, -2)
raw_switched = seq_ok & (raw != raw_nxt) & (raw != "") & (raw_nxt != "")
raw_bounce = raw_switched & (raw_n2 == raw)
log(f"- (대조) 원본 판정(페르소나_원본) 기준 단기 왕복: 전환 {int(raw_switched.sum()):,}건 중 왕복 {int(raw_bounce.sum()):,}건 "
    f"({raw_bounce.sum()/max(raw_switched.sum(),1):.2%}) — 안정화 전에는 이만큼 있었다는 대조값")

LOG.close()
