# -*- coding: utf-8 -*-
"""데이터 명세서용 프로파일링 (회귀 없음, 집계만). df_ready.csv 전체 73개 변수.
분석 실제 사용 모집단(대구·경북, 2023-01~2025-12) 기준으로 분포를 낸다.
출력: datasheet_continuous.csv, datasheet_categorical.csv, datasheet_bucket.csv, datasheet_missing.csv, datasheet_summary.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "datasheet_summary.txt"), "w", encoding="utf-8")

BUCKETS = ["0건", "1건", "2건", "2건초과 5건이하", "5건초과 10건이하", "10건초과 20건이하",
           "20건초과 30건이하", "30건초과 40건이하", "40건초과 50건이하", "50건 초과"]
BUCKET_JWA = [b.replace("건", "개") for b in BUCKETS]

ID_TIME = ["기준년월", "법인ID"]
CAT_NOMINAL = ["업종_대분류", "업종_중분류", "사업장_시도", "사업장_시군구", "법인_고객등급", "전담고객여부", "phase"]
CAT_BINARY = ["exposed"]
CONT_LEVEL = ["요구불예금잔액", "거치식예금잔액", "적립식예금잔액", "수익증권잔액", "신탁잔액", "퇴직연금잔액",
              "여신한도금액", "여신_운전자금대출잔액", "운전_할인어음잔액", "운전_당좌대출잔액", "운전_일반자금대출잔액",
              "운전_무역금융잔액", "운전_주택자금대출잔액", "운전_기업구매자금대출잔액", "운전_외상매출채권담보대출잔액",
              "운전_기타운전자금대출잔액", "여신_시설자금대출잔액", "시설_일반자금대출잔액", "시설_에너지절약시설대출잔액",
              "시설_주택자금대출잔액", "시설_기타시설자금대출잔액"]
CONT_FLOW = ["외환_수출실적금액", "외환_수입실적금액", "신용카드사용금액", "체크카드사용금액", "창구거래금액",
             "인터넷뱅킹거래금액", "스마트뱅킹거래금액", "폰뱅킹거래금액", "ATM거래금액", "자동이체금액",
             "요구불입금금액", "요구불출금금액", "exp_yoy"]
BUCKET_VARS = ["요구불예금좌수", "거치식예금좌수", "적립식예금좌수", "수익증권좌수", "신탁좌수", "퇴직연금좌수",
               "여신_운전자금대출좌수", "운전_할인어음좌수", "운전_당좌대출좌수", "운전_일반자금대출좌수",
               "운전_무역금융좌수", "운전_주택자금대출좌수", "운전_기업구매자금대출좌수", "운전_외상매출채권담보대출좌수",
               "운전_기타운전자금대출좌수", "여신_시설자금대출좌수", "시설_일반자금대출좌수", "시설_에너지절약시설대출좌수",
               "시설_주택자금대출좌수", "시설_기타시설자금대출좌수", "신용카드개수",
               "외환_수출실적거래건수", "외환_수입실적거래건수", "창구거래건수", "인터넷뱅킹거래건수",
               "스마트뱅킹거래건수", "폰뱅킹거래건수", "ATM거래건수", "자동이체거래건수"]

ALL = ID_TIME + CAT_NOMINAL + CAT_BINARY + CONT_LEVEL + CONT_FLOW + BUCKET_VARS
assert len(ALL) == 73, f"컬럼 분류 개수 불일치: {len(ALL)}"


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


d = pd.read_csv(DATA)
log(f"# df_ready.csv 프로파일링 (회귀 없음)\n원본 전체: 법인 {d['법인ID'].nunique():,}곳, 행 {len(d):,}, 기간 {d['기준년월'].min()}~{d['기준년월'].max()}")
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
full_n = len(d)
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
log(f"분석 모집단(대구·경북, 2023-01~2025-12): 법인 {d['법인ID'].nunique():,}곳, 행 {len(d):,} (전체의 {len(d)/full_n:.1%})\n")

# ---------------- 결측 ----------------
miss = d.isna().sum()
miss_df = pd.DataFrame({"변수": miss.index, "결측_수": miss.values, "결측_비율(%)": (miss.values / len(d) * 100).round(3)})
miss_df.to_csv(os.path.join(OUT, "datasheet_missing.csv"), index=False, encoding="utf-8-sig")
has_miss = miss_df[miss_df["결측_수"] > 0]
log("## 결측치 현황"); log(f"결측 있는 변수 {len(has_miss)}개 / 전체 {len(miss_df)}개")
log(has_miss.to_string(index=False) if len(has_miss) else "  (결측 있는 변수 없음)")

# ---------------- 연속형 분포 ----------------
rows = []
for c in CONT_LEVEL + CONT_FLOW:
    x = d[c].astype(float)
    pos = x[x > 0]
    q = x.quantile([0, .25, .5, .75, .95, .99, 1.0])
    rows.append({"변수": c, "N": x.notna().sum(), "평균": x.mean(), "표준편차": x.std(), "최솟값": q[0],
                 "P25": q[.25], "중앙값": q[.5], "P75": q[.75], "P95": q[.95], "P99": q[.99], "최댓값": q[1.0],
                 "0_비율(%)": (x == 0).mean() * 100, "음수_비율(%)": (x < 0).mean() * 100,
                 "양수_중앙값": pos.median() if len(pos) else np.nan, "고유값_수": x.nunique()})
cont = pd.DataFrame(rows).round(4)
cont.to_csv(os.path.join(OUT, "datasheet_continuous.csv"), index=False, encoding="utf-8-sig")
log(f"\n## 연속형 변수 {len(cont)}개 분포 (요약, 전체는 datasheet_continuous.csv)")
log(cont[["변수", "N", "평균", "0_비율(%)", "음수_비율(%)", "최댓값", "고유값_수"]].to_string(index=False))

# 로그 IQR 기반 이상치(참고): 양수값 대상 log(x+1)의 IQR 밖
log("\n## 이상치(참고): 양수 관측 중 log(x+1) 기준 IQR 상단 밖(Q3+1.5·IQR) 비율")
for c in CONT_LEVEL + CONT_FLOW:
    x = d.loc[d[c] > 0, c].astype(float)
    if len(x) < 20:
        continue
    lx = np.log(x + 1)
    q1, q3 = lx.quantile([.25, .75]); iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    share = (lx > hi).mean()
    if share > 0:
        log(f"  {c}: 양수 {len(x):,}건 중 {share:.2%}가 로그-IQR 상단 밖 (원 단위 기준값 {np.exp(hi)-1:,.1f})")

# ---------------- 범주형(명목) 분포 ----------------
cat_rows = []
for c in CAT_NOMINAL + CAT_BINARY:
    vc = d[c].astype(str).value_counts(dropna=False)
    top = vc.index[0]; topn = len(vc)
    for cat, n in vc.head(12).items():
        cat_rows.append({"변수": c, "범주": cat, "빈도": n, "비율(%)": round(n / len(d) * 100, 2)})
    if topn > 12:
        rest = vc.iloc[12:].sum()
        cat_rows.append({"변수": c, "범주": f"(그 외 {topn-12}개 범주)", "빈도": rest, "비율(%)": round(rest / len(d) * 100, 2)})
catdf = pd.DataFrame(cat_rows)
catdf.to_csv(os.path.join(OUT, "datasheet_categorical.csv"), index=False, encoding="utf-8-sig")
log(f"\n## 명목형 변수 {len(CAT_NOMINAL+CAT_BINARY)}개 범주 수·최빈값")
for c in CAT_NOMINAL + CAT_BINARY:
    vc = d[c].astype(str).value_counts()
    rare = (vc / len(d) < 0.001).sum()
    log(f"  {c}: 범주 {vc.size}개, 최빈 '{vc.index[0]}'({vc.iloc[0]/len(d):.1%}), 0.1% 미만 희소 범주 {rare}개")

# ---------------- 구간형(좌수·건수) 분포 ----------------
bk_rows = []
for c in BUCKET_VARS:
    vals = d[c].astype(str)
    zero_label = "0건" if "건" in vals.iloc[0] or vals.str.contains("건").any() else "0개"
    zero_rate = (vals == zero_label).mean()
    vc = vals.value_counts()
    bk_rows.append({"변수": c, "범주_수": vc.size, "0_비율(%)": round(zero_rate * 100, 2),
                    "최빈_범주": vc.index[0], "최빈_비율(%)": round(vc.iloc[0] / len(d) * 100, 2)})
bkdf = pd.DataFrame(bk_rows)
bkdf.to_csv(os.path.join(OUT, "datasheet_bucket.csv"), index=False, encoding="utf-8-sig")
log(f"\n## 구간형(좌수·건수) 변수 {len(BUCKET_VARS)}개 — 공통 10구간, 0 비율")
log(bkdf.to_string(index=False))

# ---------------- 정합성 점검 ----------------
log("\n## 정합성 점검")
WJ = ["운전_할인어음잔액", "운전_당좌대출잔액", "운전_일반자금대출잔액", "운전_무역금융잔액", "운전_주택자금대출잔액",
      "운전_기업구매자금대출잔액", "운전_외상매출채권담보대출잔액", "운전_기타운전자금대출잔액"]
SS = ["시설_일반자금대출잔액", "시설_에너지절약시설대출잔액", "시설_주택자금대출잔액", "시설_기타시설자금대출잔액"]
for name, sub, tot in [("운전자금", WJ, "여신_운전자금대출잔액"), ("시설자금", SS, "여신_시설자금대출잔액")]:
    diff = d[tot] - d[sub].sum(axis=1)
    log(f"  {tot} = Σ{name} 하위계정: 일치(|차|≤0.01) {(diff.abs()<=0.01).mean():.2%}, 최대|차| {diff.abs().max():.1f}")
for name, bal, jwa, zero in [("거치식", "거치식예금잔액", "거치식예금좌수", "0개"), ("적립식", "적립식예금잔액", "적립식예금좌수", "0개")]:
    b0 = d[bal] > 0; j0 = d[jwa].astype(str) != zero
    log(f"  {bal}/{jwa} 정합성: 잔액>0·좌수=0 {(b0 & ~j0).mean():.3%}, 잔액=0·좌수>0 {(~b0 & j0).mean():.3%}")
for name, amt, cnt in [("수출", "외환_수출실적금액", "외환_수출실적거래건수"), ("수입", "외환_수입실적금액", "외환_수입실적거래건수")]:
    a0 = np.isclose(d[amt], 0); c0 = d[cnt].astype(str) == "0건"
    log(f"  {amt}/{cnt} 정합성: 금액0·건수≠0건 {(a0 & ~c0).mean():.4%}, 금액>0·건수=0건 {(~a0 & c0).mean():.4%}")

# ---------------- 관측 구조 ----------------
g = d.groupby("법인ID")["기준년월"]
n = g.size(); first = g.min(); last = g.max()
full = (first == 202301) & (last == 202512) & (n == 36)
log(f"\n## 관측 구조: 36개월 완전 패널 법인 {full.mean():.1%}, 법인당 관측월수 중앙값 {n.median():.0f}")
log(f"중복 행(법인ID×기준년월) 여부: {d.duplicated(['법인ID','기준년월']).sum()}건")

LOG.close()
