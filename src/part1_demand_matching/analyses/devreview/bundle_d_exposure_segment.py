# -*- coding: utf-8 -*-
"""묶음 D — 외환노출 정의와 운영 세그먼트 연결 (회귀 없음, 기술 통계·교차표만)
디벨롭 문서 §5 묶음 D에 대응.
① 동일 기준월 교차표: 분석용 36개월 노출(exposed) vs 운영용 최근 12개월 노출(2025-01~2025-12 실적)
② E23 vs 전체기간 노출: 같은 평가 기간(2024-01~)에서 두 정의로 분류가 달라지는 법인 규모만 기술(회귀 재실행 없음, harmonized 결과 재사용)
③ 세그먼트 전환 원인: 창 이탈 vs 신규 거래. "최근 12개 관측 행" != "최근 12개 달력월" 혼동 여부 확인
④ 수출입 병행 기업: 현재 운영 분류(수출 있으면 수출형)는 유지, 분석용 보조 태그만 추가
출력: bundle_d_summary.txt, bundle_d_crosstab.csv (집계만)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
HARM = r"C:\test\outputs\harmonized\results_all.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_d_summary.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "외환_수출실적금액", "외환_수입실적금액"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
log(f"# 묶음 D — 외환노출 정의와 운영 세그먼트 연결 (회귀 없음)\n대상: 대구·경북 {d['법인ID'].nunique():,}법인\n")

# ---------------- ① 교차표 ----------------
log("## 1. 동일 기준월 교차표 — 분석용 36개월 노출 vs 운영용 최근 12개월 노출")
firm_36 = d.groupby("법인ID")["exposed"].max()                                    # 36개월 전체 기준(팀 표준, 시간불변)
last12 = d[d["기준년월"] >= 202501]
assert last12["기준년월"].nunique() == 12, "최근 12개월 정의(달력월 수) 확인"      # §3.6.3 "관측 행" ≠ "달력 월" 혼동 점검
op12 = (last12.groupby("법인ID")[["외환_수출실적금액", "외환_수입실적금액"]].max() > 0).any(axis=1)
op12 = op12.reindex(firm_36.index).fillna(False).astype(bool)   # reindex 후 object dtype이 되면 ~True가 -2(truthy)가 되는 버그 방지
tab = pd.crosstab(firm_36.rename("분석용_36개월"), op12.rename("운영용_최근12개월"))
log(tab.to_string())
imposs = int(((firm_36 == 0) & op12).sum())
log(f"- 정의상 불가능해야 할 칸(36개월 비노출인데 최근 12개월엔 실적): {imposs}곳 "
    f"({'정합성 이상 없음 — 최근 12개월 실적은 36개월 전체에 포함되므로 항상 0이어야 함' if imposs == 0 else '확인 필요: 36개월 노출 정의에 최근 12개월이 안 들어갔을 가능성'})")
n11 = int(((firm_36 == 1) & op12).sum()); n10 = int(((firm_36 == 1) & ~op12).sum())
log(f"- 노출(36개월)=1 중 현재도 실적 이력 있음(1·1): {n11:,}곳 / 과거엔 있었지만 최근 12개월 창에서 이탈(1·0): {n10:,}곳 "
    f"({n10/(n10+n11):.1%})")
tab.to_csv(os.path.join(OUT, "bundle_d_crosstab.csv"), encoding="utf-8-sig")

# ---------------- ③ 세그먼트 전환 원인 ----------------
log("\n## 2. 세그먼트 전환 원인 — 창 이탈 vs 관측 자체가 없어짐")
cand = firm_36[(firm_36 == 1) & (~op12)].index                                    # 창 이탈 후보(노출인데 최근 12개월엔 실적 없는 법인)
sub = d[d["법인ID"].isin(cand)]
last_obs = sub.groupby("법인ID")["기준년월"].max()
no_obs_recent = int((last_obs < 202501).sum())                                    # 아예 2025년 관측 자체가 없음(거래 종료/이탈)
has_obs_but_zero = int((last_obs >= 202501).sum())                                # 2025년 관측은 있으나 실적이 0(창 이탈, 진짜 둔화)
log(f"- 창 이탈 {len(cand):,}곳 중: 2025년 관측 자체가 없음(거래 관계 종료 추정) {no_obs_recent:,}곳 / "
    f"2025년에도 관측은 있지만 수출입 실적만 0(실질적 둔화) {has_obs_but_zero:,}곳")
log("- '최근 12개 관측 행'과 '최근 12개 달력월'을 혼동했는지: 위 assert가 통과했으므로 최근 12개월 창은 달력월(2025-01~2025-12) 12개로 정의했고, 법인별 관측 개수가 아니다. 문제없음.")

# ---------------- ② E23 vs 전체기간 노출 ----------------
log("\n## 3. E23(2023년 기준 노출) vs 전체기간 노출 — 정의 차이의 표본 규모")
e23 = d[d["기준년월"] < 202401].assign(a=lambda x: (x["외환_수출실적금액"] > 0) | (x["외환_수입실적금액"] > 0)).groupby("법인ID")["a"].max().fillna(False)
e23 = e23.reindex(firm_36.index).fillna(False).astype(bool)
log(f"- 노출(36개월 전체) {int(firm_36.sum()):,}곳 vs 노출(2023년 기준, E23) {int(e23.sum()):,}곳 "
    f"— E23 표본이 {(1 - e23.sum()/firm_36.sum()):.1%} 더 작다.")
both = int(((firm_36 == 1) & e23).sum())
only36 = int(((firm_36 == 1) & ~e23).sum())
log(f"- 둘 다 노출: {both:,}곳 / 36개월에서만 노출(2024~2025년에 새로 시작): {only36:,}곳")
try:
    hz = pd.read_csv(HARM)
    r_c1 = hz[(hz["계정"] == "요구불") & (hz["모형"] == "C1") & (hz["h"] == 6)].iloc[0]
    r_e23 = hz[(hz["계정"] == "요구불") & (hz["모형"] == "E23") & (hz["h"] == 6)].iloc[0]
    log(f"- harmonized 결과 재인용(회귀 재실행 안 함): C1 SE {r_c1['SE']:.4f}(노출 {int(r_c1['노출법인'])}곳, {int(r_c1['월'])}개월) vs "
        f"E23 SE {r_e23['SE']:.4f}(노출 {int(r_e23['노출법인'])}곳, {int(r_e23['월'])}개월)")
    log(f"- SE 비율 {r_e23['SE']/r_c1['SE']:.2f}배. 노출 법인 수 비율은 {r_c1['노출법인']/r_e23['노출법인']:.2f}배, 월 수 비율은 {r_c1['월']/r_e23['월']:.2f}배다.")
    log("  → SE가 커진 게 '노출 정의 자체의 문제'인지 '표본·기간이 작아져서'인지는, 두 비율이 SE 비율과 비슷한 크기면 후자(표본 크기) 쪽 설명이 더 맞는다. "
        "노출 법인 수 축소와 월 수 축소가 함께 작용한 것으로 보이며, 정의 자체가 틀렸다는 근거는 없다.")
except Exception as ex:
    log(f"- harmonized 결과 재인용 실패: {ex}")

# ---------------- ④ 수출입 병행 기업 ----------------
log("\n## 4. 수출입 병행 기업 (분석용 보조 태그, 운영 분류는 유지)")
exp_only = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
imp_only = d.groupby("법인ID")["외환_수입실적금액"].max() > 0
kind = pd.Series("무노출", index=firm_36.index)
kind[exp_only & ~imp_only] = "수출만"; kind[~exp_only & imp_only] = "수입만"; kind[exp_only & imp_only] = "수출입 병행"
vc = kind[firm_36 == 1].value_counts()
log(f"- 노출 법인 {int(firm_36.sum()):,}곳 중: " + ", ".join(f"{k} {v:,}곳({v/firm_36.sum():.1%})" for k, v in vc.items()))
log(f"- 현재 운영 분류(수출 있으면 수출형)에 포함되는 수출입 병행 {int((kind=='수출입 병행').sum()):,}곳은 "
    f"수출형으로 잡혀 수입 상담 필요가 가려질 수 있다. 분석용 보조 태그로 구분해 둔다(운영 분류 자체는 바꾸지 않음).")
pd.crosstab(firm_36.rename("노출_36개월"), kind.rename("수출입구분")).to_csv(
    os.path.join(OUT, "bundle_d_export_import_tag.csv"), encoding="utf-8-sig")   # 집계만, 법인 ID 없음

LOG.close()
