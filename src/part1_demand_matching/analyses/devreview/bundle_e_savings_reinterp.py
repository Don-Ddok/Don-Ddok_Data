# -*- coding: utf-8 -*-
"""묶음 E — 저축성예금 결과의 재해석 (회귀 없음, 기술 통계만)
디벨롭 문서 §5 묶음 E에 대응. 보유 확률 반응(harmonized HOLD)과 잔액 반응(harmonized C1)의
표본이 실제로 같은지, 0/양수 전환이 좌수와 일치하는지, POS(양수 끝점만) 표본이 얼마나 바뀌는지 확인한다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
HARM = r"C:\test\outputs\harmonized\results_all.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_e_summary.txt"), "w", encoding="utf-8")
BUCKETS = ["0개", "1개", "2개", "2개초과 5개이하", "5개초과 10개이하", "10개초과 20개이하",
           "20개초과 30개이하", "30개초과 40개이하", "40개초과 50개이하", "50개 초과"]


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed",
                               "거치식예금잔액", "적립식예금잔액", "거치식예금좌수", "적립식예금좌수"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
log(f"# 묶음 E — 저축성예금 결과 재해석 (회귀 없음)\n대상: 대구·경북 {d['법인ID'].nunique():,}법인\n")

# ---------------- 1. 보유 이력 표본 vs 전체 법인 ----------------
log("## 1. 보유 이력 표본과 전체 법인 표본의 질문 구분")
for name, col in [("거치식", "거치식예금잔액"), ("적립식", "적립식예금잔액")]:
    ever = d.groupby("법인ID")[col].max() > 0
    log(f"- {name}: 전체 {d['법인ID'].nunique():,}곳 중 보유 이력(한 번이라도 양수) {int(ever.sum()):,}곳({ever.mean():.1%}) — "
        f"harmonized C1·HOLD는 이 {int(ever.sum()):,}곳만 대상이다. 나머지 {int((~ever).sum()):,}곳(한 번도 양수 아님)에는 이 결과를 적용할 수 없다.")
    log(f"  → '{name} 보유 확률 반응'은 '이미 보유 이력 있는 법인 안에서, 양수 유지 여부가 어떻게 바뀌는가'이지, "
        f"'일반 법인이 새로 {name}을 개설할 확률'이 아니다.")

# ---------------- 2. 0/양수 전환을 좌수와 대조 ----------------
log("\n## 2. 0/양수 전환과 좌수의 일치")
for name, bal, jwa in [("거치식", "거치식예금잔액", "거치식예금좌수"), ("적립식", "적립식예금잔액", "적립식예금좌수")]:
    b0 = d[bal] > 0
    j0 = d[jwa] != "0개"
    log(f"- {name}: 잔액>0 & 좌수=0개 {int((b0 & ~j0).sum()):,}행 / 잔액=0 & 좌수>0개 {int((~b0 & j0).sum()):,}행 "
        f"(전체 {len(d):,}행 중) — {'완전 일치' if (b0 & ~j0).sum()==0 and (~b0 & j0).sum()==0 else '불일치 존재, 확인 필요'}")

# ---------------- 3. POS(양수 끝점만) 표본 변경 ----------------
log("\n## 3. POS(잔액 양수 관측만) 강건성이 바꾸는 표본 크기")
try:
    hz = pd.read_csv(HARM)
    for acct in ["거치식", "적립식"]:
        c1 = hz[(hz["계정"] == acct) & (hz["모형"] == "C1") & (hz["h"] == 6) & hz["업종"].isna()]
        pos = hz[(hz["계정"] == acct) & (hz["모형"] == "POS") & (hz["h"] == 6) & hz["업종"].isna()]
        if len(c1) and len(pos):
            c1r, posr = c1.iloc[0], pos.iloc[0]
            log(f"- {acct} h=6: C1 N={int(c1r['N']):,}(노출 {int(c1r['노출법인'])}) → POS(양수 끝점만) N={int(posr['N']):,}(노출 {int(posr['노출법인'])}), "
                f"표본이 {1 - posr['N']/c1r['N']:.1%} 줄어든다. β3도 {c1r['β3']:+.4f} → {posr['β3']:+.4f}로 바뀐다(참고 — 표본이 다르므로 직접 비교는 방향만 본다).")
        else:
            log(f"- {acct}: POS 결과가 results_all.csv에 없음(요구불·운전자금에만 있을 수 있음) — 확인 필요")
except Exception as ex:
    log(f"- harmonized 결과 재인용 실패: {ex}")

# ---------------- 4. 업종별 결과의 다중검정 범위 ----------------
log("\n## 4. 업종별 결과의 다중검정 범위 — 최초 탐색 범위까지 추적")
log("- 원본(팀 예금 노트북 `02_deposit_industry_lp_revised.ipynb` §5)에 이미 명시돼 있다: "
    "\"아래 세 업종은 원본 작업에서 **상위 10개 업종 × 2계정 × 13시차(=260개 가설)**를 본 뒤 **사후에 고른** 것입니다.\"")
log("- harmonized의 Holm(36) = 3업종 × h=1~12(12개)는 이 원본 260개 가설의 일부(선택된 3개 업종)에만 적용한 것이다. "
    "**진짜 다중검정 보정은 260개 기준이어야 더 엄격하다.**")
log("- 다행히 이 방향의 보정은 결과를 더 보수적으로 만들 뿐이다: harmonized REPORT.md가 이미 \"Holm(36) 후 h=6 모두 1.000\"이라고 적어 뒀으므로, "
    "260개 기준으로 다시 봐도 결론(업종별 = 탐색적, 판정에 쓰지 않음)은 바뀌지 않는다. 다만 그 260이라는 숫자 자체가 REPORT.md·SPEC.md에 안 적혀 있었다 — 문구 보강을 권한다.")

LOG.close()
