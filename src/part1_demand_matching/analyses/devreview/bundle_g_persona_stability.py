# -*- coding: utf-8 -*-
"""묶음 G — 페르소나의 안정성과 설명력 (회귀 없음, 기술 통계만)
디벨롭 문서 §5 묶음 G에 대응. persona.py(원본, 수정 안 함)의 산출물(persona_firm_month.csv)만 다시 읽는다.
우선순위 A>C>B>D>E 자체가 실제 자금 필요를 나타낸다는 검증은 아니며, 새 군집 모델을 만들지 않는다.
목표는 기존 페르소나가 얼마나 안정적이고 무엇을 설명하는지 확인하는 것.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PF = r"C:\test\분석결과\matching\persona_firm_month.csv"
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_g_summary.txt"), "w", encoding="utf-8")
PORDER = ["A 결제형", "C 설비투자형", "B 차입운영형", "D 현금비축형", "E 요구불중심형"]


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


p = pd.read_csv(PF, dtype={"법인ID": str})
p["페르소나"] = p["페르소나"].fillna("")               # 빈 칸(비노출)이 NaN으로 읽혀 "!= ''"가 항상 True가 되는 함정 방지
for c in ["보유_A", "보유_B", "보유_C", "보유_D"]:
    p[c] = p[c].fillna("")
p["_m"] = (p["연월"] // 100) * 12 + p["연월"] % 100
log(f"# 묶음 G — 페르소나 안정성과 설명력 (회귀 없음)\n법인×월 {len(p):,}행, 노출 법인×월 {(p['페르소나']!='').sum():,}행\n")

# ---------------- 1. 전환행렬 ----------------
log("## 1. 주 페르소나 전환행렬 (월 t → t+1, 노출 법인)")
cur = p[p["페르소나"] != ""][["법인ID", "_m", "페르소나"]].rename(columns={"페르소나": "p0"})
nxt = p[["법인ID", "_m", "페르소나"]].rename(columns={"_m": "_m1", "페르소나": "p1"})
nxt["_m"] = nxt["_m1"] - 1
tr = cur.merge(nxt[["법인ID", "_m", "p1"]], on=["법인ID", "_m"], how="inner")
tr = tr[tr["p1"] != ""]           # 다음 달에도 노출(비노출로 빠지거나 관측 끊기면 제외 — 별도 집계)
ct = pd.crosstab(tr["p0"], tr["p1"]).reindex(index=PORDER, columns=PORDER, fill_value=0)
log(ct.to_string())
ret = pd.Series({k: (ct.loc[k, k] / ct.loc[k].sum() if ct.loc[k].sum() else np.nan) for k in PORDER})
log(f"\n- 페르소나별 유지율(다음달도 같음): " + ", ".join(f"{k} {v:.1%}" for k, v in ret.items()))
log(f"- 전체 유지율(가중): {np.trace(ct.to_numpy())/ct.to_numpy().sum():.1%} ({int(np.trace(ct.to_numpy())):,}/{int(ct.to_numpy().sum()):,})")
# 노출->비노출 이탈, 관측 끊김 비율
all_next = cur.merge(nxt[["법인ID", "_m", "p1"]], on=["법인ID", "_m"], how="left")
log(f"- 다음달 비노출로 전환: {int((all_next['p1']=='').sum()):,}행 / 다음달 관측 자체 없음: {int(all_next['p1'].isna().sum()):,}행 "
    f"(전체 {len(all_next):,}행 중)")

# 단기 복귀율 A->B->A류 (t, t+1, t+2 모두 노출, p0!=p1, p2==p0)
t2 = p[["법인ID", "_m", "페르소나"]].rename(columns={"_m": "_m2", "페르소나": "p2"}); t2["_m"] = t2["_m2"] - 2
tr3 = tr.merge(t2[["법인ID", "_m", "p2"]], on=["법인ID", "_m"], how="inner")
tr3 = tr3[tr3["p2"] != ""]
switched = tr3[tr3["p0"] != tr3["p1"]]
bounce = switched[switched["p0"] == switched["p2"]]
log(f"- 단기 복귀(t≠t+1, t+2=t): 페르소나가 바뀐 {len(switched):,}건 중 2개월 안에 원래대로 돌아온 경우 {len(bounce):,}건({len(bounce)/max(len(switched),1):.1%})")

# ---------------- 2. 변경 원인 ----------------
log("\n## 2. 변경 원인 분해 — 보유 플래그·세그먼트 변화로 100% 설명되는지")
hold_cur = p[p["페르소나"] != ""][["법인ID", "_m", "보유_A", "보유_B", "보유_C", "보유_D", "세그먼트"]]
merged = tr.merge(hold_cur.rename(columns={c: c + "_0" for c in ["보유_A", "보유_B", "보유_C", "보유_D", "세그먼트"]}),
                   on=["법인ID", "_m"], how="left")
hold_next = hold_cur.copy(); hold_next["_m"] = hold_next["_m"] - 1
merged = merged.merge(hold_next.rename(columns={c: c + "_1" for c in ["보유_A", "보유_B", "보유_C", "보유_D", "세그먼트"]}),
                       on=["법인ID", "_m"], how="left")
chg = merged[merged["p0"] != merged["p1"]]
flag_changed = (chg["보유_A_0"] != chg["보유_A_1"]) | (chg["보유_B_0"] != chg["보유_B_1"]) | \
               (chg["보유_C_0"] != chg["보유_C_1"]) | (chg["보유_D_0"] != chg["보유_D_1"])
seg_changed = chg["세그먼트_0"] != chg["세그먼트_1"]
log(f"- 페르소나가 바뀐 {len(chg):,}건 중 보유 플래그(A~D)도 함께 바뀐 경우 {flag_changed.sum():,}건({flag_changed.mean():.1%}), "
    f"세그먼트도 바뀐 경우 {seg_changed.sum():,}건({seg_changed.mean():.1%})")
log(f"- 둘 다 안 바뀌었는데 페르소나만 바뀐 경우(로직상 있으면 안 됨): {int((~flag_changed & ~seg_changed).sum()):,}건 "
    f"({'이상 없음' if (~flag_changed & ~seg_changed).sum()==0 else '확인 필요 — 결정론적 규칙 위반 가능성'})")

# ---------------- 3. 보조 태그 조합 (전체 패널) ----------------
log("\n## 3. 보조 태그 동시 보유 (전체 패널, 노출 법인×월 기준, 스냅샷 아님)")
tags = p[p["페르소나"] != ""][["보유_A", "보유_B", "보유_C", "보유_D"]]
ncnt = (tags == "Y").sum(axis=1)
log(f"- 보유 태그 개수 분포: " + ", ".join(f"{k}개 {v:,}행({v/len(tags):.1%})" for k, v in ncnt.value_counts().sort_index().items()))
combo = tags.apply(lambda r: "".join(k for k in "ABCD" if r[f"보유_{k}"] == "Y") or "(없음)", axis=1)
log("- 상위 조합: " + ", ".join(f"{k} {v:,}" for k, v in combo.value_counts().head(8).items()))

# ---------------- 4. 주 페르소나가 숨기는 거래 특성 ----------------
log("\n## 4. 주 페르소나가 숨기는 거래 특성 (정의에 안 쓴 변수로 비교, 2025-12 스냅샷)")
snap = p[(p["연월"] == 202512) & (p["페르소나"] != "")][["법인ID", "페르소나"]]
extra = pd.read_csv(DATA, usecols=["법인ID", "기준년월", "요구불입금금액", "요구불출금금액",
                                   "외환_수출실적금액", "외환_수입실적금액", "창구거래건수"])
extra = extra[extra["기준년월"] == 202512]
BR = {b: i for i, b in enumerate(["0건", "1건", "2건", "2건초과 5건이하", "5건초과 10건이하", "10건초과 20건이하",
                                  "20건초과 30건이하", "30건초과 40건이하", "40건초과 50건이하", "50건 초과"])}
extra["창구활동순위"] = extra["창구거래건수"].map(BR)
extra["입출금합"] = extra["요구불입금금액"] + extra["요구불출금금액"]
extra["외환합"] = extra["외환_수출실적금액"] + extra["외환_수입실적금액"]
j = snap.merge(extra, on="법인ID", how="left")
g = j.groupby("페르소나")[["입출금합", "외환합", "창구활동순위"]].median()
log(g.reindex(PORDER).round(2).to_string())
log("- '보유 태그(A~D)'만으로는 이 세 변수(입출금 활동량, 외환 활동량, 창구 활동 순위)의 차이가 다 드러나지 않는다 — "
    "예를 들어 페르소나가 같아도 입출금 활동량 중앙값이 갈릴 수 있다는 것을 표로 확인한다.")

# ---------------- 5. 우선순위 민감도 ----------------
log("\n## 5. 우선순위 민감도 — 사전 고정 대안 하나(규모 우선)와 비교")
log("현재 규칙: 보유 여부 우선순위 A>C>B>D>E (첫 번째 해당 조건). 대안: 네 잔액(A=할인어음+무역금융, C=시설자금, "
    "B=순수운전자금, D=거치식+적립식) 중 **가장 큰 것**을 주 페르소나로 정하는 규모 기준.")
r = pd.read_csv(DATA, usecols=["법인ID", "기준년월", "사업장_시도", "업종_대분류",
                               "운전_할인어음잔액", "운전_무역금융잔액", "여신_시설자금대출잔액",
                               "여신_운전자금대출잔액", "거치식예금잔액", "적립식예금잔액"])
r = r[(r["기준년월"] == 202512) & r["사업장_시도"].astype(str).str.strip().isin(["대구", "경북"]) & (r["업종_대분류"] != "금융 및 보험업")]
r["A"] = r["운전_할인어음잔액"].fillna(0) + r["운전_무역금융잔액"].fillna(0)
r["C"] = r["여신_시설자금대출잔액"].fillna(0)
r["B"] = (r["여신_운전자금대출잔액"].fillna(0) - r["A"]).clip(lower=0)
r["D"] = r["거치식예금잔액"].fillna(0) + r["적립식예금잔액"].fillna(0)
amt = r[["A", "B", "C", "D"]]
NAME = {"A": "A 결제형", "B": "B 차입운영형", "C": "C 설비투자형", "D": "D 현금비축형"}
alt = np.where(amt.max(axis=1) <= 0, "E 요구불중심형", amt.idxmax(axis=1).map(NAME))
r["_alt"] = alt
j2 = snap.merge(r[["법인ID", "_alt"]], on="법인ID", how="inner")
same = (j2["페르소나"] == j2["_alt"]).mean()
log(f"- 2025-12 노출 법인 {len(j2):,}곳: 현재 규칙과 규모 기준이 같은 페르소나를 주는 비율 {same:.1%}")
log(pd.crosstab(j2["페르소나"], j2["_alt"]).reindex(index=PORDER, columns=PORDER, fill_value=0).to_string())
log("- 규모 기준으로 바꾸면 결과가 크게 달라진다면, 현재 '보유 여부 우선순위'는 자금 규모가 아니라 **'어떤 종류를 하나라도 쓰는가'**를 "
    "기준으로 삼는 것임을 재확인한다(v2 변경 이력과 일치 — v1은 잔액 크기 비교였다가 규모가 큰 대출 쪽으로 쏠려 v2로 바뀜).")

LOG.close()
