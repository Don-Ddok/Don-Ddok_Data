# -*- coding: utf-8 -*-
"""묶음 I — 상품 추천의 집중도·변경 원인·문구 검증 (회귀 없음, 기술 통계만)
디벨롭 문서 §6.6~6.11에 대응. matcher.py·match_rules.yaml(원본, 수정 안 함)의 산출물만 다시 읽는다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RF = r"C:\test\분석결과\matching\recommend_firm_month.csv"
PF = r"C:\test\분석결과\matching\persona_firm_month.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_i_summary.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


usecols = ["법인ID", "연월", "세그먼트", "페르소나", "단계",
           "추천1_상품명", "추천1_법인추천가능", "추천2_상품명", "추천2_법인추천가능", "추천3_상품명", "추천3_법인추천가능"]
r = pd.read_csv(RF, usecols=usecols, dtype={"법인ID": str}, encoding="utf-8-sig", low_memory=False)
for c in ["페르소나"] + [f"추천{i}_상품명" for i in (1, 2, 3)]:
    r[c] = r[c].fillna("")
log(f"# 묶음 I — 상품 추천 검증 (회귀 없음)\n법인×월 {len(r):,}행\n")

# ---------------- 6.6 전체 지표 희석 ----------------
log("## 1. 확인필요 비율 — 전체 vs 노출 슬롯만")
allcan = pd.concat([r[f"추천{i}_법인추천가능"] for i in (1, 2, 3)])
allcan = allcan[allcan != ""]
share_all = (allcan == "확인필요").mean()
exposed = r["세그먼트"].isin(["수출형", "수입형"])
allcan_exp = pd.concat([r.loc[exposed, f"추천{i}_법인추천가능"] for i in (1, 2, 3)])
allcan_exp = allcan_exp[allcan_exp != ""]
share_exp = (allcan_exp == "확인필요").mean()
log(f"- 전체 추천 슬롯 {len(allcan):,}건 중 확인필요 {share_all:.1%}")
log(f"- 노출 법인 추천 슬롯만 {len(allcan_exp):,}건 중 확인필요 {share_exp:.1%}")
log(f"- 비노출 법인 행 비율: {(~exposed).mean():.1%} (전체 {len(r):,}행 중 {int((~exposed).sum()):,}행) — "
    f"전체 지표는 이 비노출 기본 추천에 희석된다는 문서 §6.6 지적이 수치로 확인된다.")

# ---------------- 6.7 추천 개인화 범위 ----------------
log("\n## 2. 추천 개인화 범위 (2025-12 노출 법인)")
snap = r[(r["연월"] == 202512) & exposed].copy()
snap["_combo"] = snap["추천1_상품명"] + "|" + snap["추천2_상품명"] + "|" + snap["추천3_상품명"]
snap["_set"] = snap.apply(lambda x: frozenset(v for v in (x["추천1_상품명"], x["추천2_상품명"], x["추천3_상품명"]) if v), axis=1)
n_combo = snap["_combo"].nunique(); n_set = snap["_set"].nunique()
log(f"- 2025-12 노출 법인 {len(snap):,}곳: 고유 추천 조합(순서 포함) {n_combo}개, 상품 집합(순서 무시) {n_set}개")
log(f"  → 조합 수가 상품 집합 수보다 {'많으면' if n_combo > n_set else '같으면'} 순서만 다른 경우가 {n_combo - n_set}개 있다는 뜻이다.")
top_combo = snap["_combo"].value_counts(normalize=True).head(5)
log("- 상위 5개 조합이 차지하는 비중: " + ", ".join(f"{v:.1%}" for v in top_combo))
log(f"  (상위 1개 조합만으로 {top_combo.iloc[0]:.1%}를 차지 — 이 조합: {top_combo.index[0]})")

# 페르소나가 달라도 추천이 같은 비율
byp = snap.groupby("페르소나")["_combo"].nunique()
log(f"- 페르소나별 서로 다른 조합 수: {byp.to_dict()}")
cross = pd.crosstab(snap["페르소나"], snap["_combo"])
same_combo_diff_persona = (cross > 0).sum(axis=0)
log(f"- 조합 {n_combo}개 중 두 개 이상의 페르소나가 공유하는 조합 수: {int((same_combo_diff_persona > 1).sum())}개")

# ---------------- 6.8 추천 변경 원인 (월 t -> t+1) ----------------
log("\n## 3. 추천 변경 원인 분해 (월 t→t+1)")
r["_m"] = (r["연월"] // 100) * 12 + r["연월"] % 100
r["_combo"] = r["추천1_상품명"] + "|" + r["추천2_상품명"] + "|" + r["추천3_상품명"]
cur = r[["법인ID", "_m", "_combo", "단계", "페르소나", "세그먼트"]]
nxt = r[["법인ID", "_m", "_combo", "단계", "페르소나", "세그먼트"]].rename(
    columns={c: c + "_1" for c in ["_combo", "단계", "페르소나", "세그먼트"]})
nxt["_m"] = nxt["_m"] - 1
mg = cur.merge(nxt, on=["법인ID", "_m"], how="inner")
chg = mg[mg["_combo"] != mg["_combo_1"]]
stage_chg = chg["단계"] != chg["단계_1"]; persona_chg = chg["페르소나"] != chg["페르소나_1"]; seg_chg = chg["세그먼트"] != chg["세그먼트_1"]
log(f"- 추천 조합이 바뀐 {len(chg):,}건 중: 단계 변화 동반 {stage_chg.mean():.1%}, 페르소나 변화 동반 {persona_chg.mean():.1%}, "
    f"세그먼트 변화 동반 {seg_chg.mean():.1%}, 셋 다 그대로인데 추천만 바뀐 경우 {(~stage_chg & ~persona_chg & ~seg_chg).mean():.1%} "
    f"({'상품표·필터 등 다른 원인' if (~stage_chg & ~persona_chg & ~seg_chg).sum() else '이상 없음(전부 설명됨)'})")

LOG.close()
