# -*- coding: utf-8 -*-
"""묶음 A — 데이터 정의·정합성·반올림 (회귀 없음, 기술 통계만)
디벨롭 요청 문서 §5 묶음 A에 대응. 새 회귀·새 정의 없음. df_ready를 읽기만 한다.
1. 계정 포함 관계: 운전자금/시설자금 합계 vs 하위 계정 합
2. 음수·0·작은 양수 비중 (주요 4계정)
3. 반올림 영향: 규모별 동일 잔액 유지율(1·3·6개월), 반올림 경계 부근 변화
4. 관측 구조: 첫/마지막 관측, 연속 관측 개월 수, 중간 누락
출력: bundle_a_summary.txt (집계만)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_a_summary.txt"), "w", encoding="utf-8")

WJ_SUB = ["운전_할인어음잔액", "운전_당좌대출잔액", "운전_일반자금대출잔액", "운전_무역금융잔액",
          "운전_주택자금대출잔액", "운전_기업구매자금대출잔액", "운전_외상매출채권담보대출잔액", "운전_기타운전자금대출잔액"]
SS_SUB = ["시설_일반자금대출잔액", "시설_에너지절약시설대출잔액", "시설_주택자금대출잔액", "시설_기타시설자금대출잔액"]
MAIN = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "여신_운전자금대출잔액", "여신_시설자금대출잔액"] +
                WJ_SUB + SS_SUB + list(MAIN.values()))
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
log(f"# 묶음 A — 데이터 정의·정합성·반올림 (회귀 없음)\n대상: 대구·경북 {d['법인ID'].nunique():,}법인 x 36개월, {len(d):,}행\n")

# ---------------- 1. 계정 포함 관계 ----------------
log("## 1. 계정 포함 관계")
for name, sub, tot in [("운전자금", WJ_SUB, "여신_운전자금대출잔액"), ("시설자금", SS_SUB, "여신_시설자금대출잔액")]:
    s = d[sub].sum(axis=1)
    diff = d[tot] - s
    log(f"- {name}: 합계열 vs 하위 {len(sub)}개 합 — 일치(반올림 오차 |차|≤0.01) {int((diff.abs() <= 0.01).sum()):,}행 "
        f"({(diff.abs() <= 0.01).mean():.1%}), 불일치 최대 |차| {diff.abs().max():.2f}, 불일치 행 중앙값 |차| {diff[diff.abs() > 0.01].abs().median() if (diff.abs() > 0.01).any() else 0:.3f}")
    mism = diff.abs() > 0.01
    if mism.any():
        pos_tot0 = (d.loc[mism, tot] == 0) & (s[mism] > 0)
        log(f"    불일치 중 합계열=0인데 하위합>0인 행 {int(pos_tot0.sum()):,} / 그 반대(합계열>0, 하위합=0) {int(((d.loc[mism, tot] > 0) & (s[mism] == 0)).sum()):,}")
# 할인어음·무역금융이 운전자금 하위에 이미 포함되는지 (중복 합산 여부는 위 포함관계 검산으로 확인됨). 상호 상관만 참고.
log(f"  참고: 할인어음·무역금융 상관 {d['운전_할인어음잔액'].corr(d['운전_무역금융잔액']):.3f} (포함관계는 위에서 이미 확인)")

# ---------------- 2. 음수·0·작은 양수 비중 ----------------
log("\n## 2. 음수·0·작은 양수 비중 (주요 4계정)")
for name, col in MAIN.items():
    v = d[col]
    small = (v > 0) & (v <= v[v > 0].quantile(0.01)) if (v > 0).any() else pd.Series(False, index=v.index)
    log(f"- {name}: 음수 {int((v < 0).sum()):,}행({(v < 0).mean():.3%}) / 0 {int((v == 0).sum()):,}행({(v == 0).mean():.1%}) / "
        f"양수 {int((v > 0).sum()):,}행({(v > 0).mean():.1%}) / 양수 중 하위 1% 이하(작은 양수) {int(small.sum()):,}행")
    if (v < 0).any():
        log(f"    음수값 분포: 최소 {v[v < 0].min():.2f}, 중앙값 {v[v < 0].median():.2f}, 최대(0에 가장 가까움) {v[v < 0].max():.2f}, 음수 법인 {d.loc[v < 0, '법인ID'].nunique()}곳")

# 순수운전자금 = 운전자금 합계 - 할인어음 - 무역금융 (하드코딩된 팀 정의가 로컬에 없어, 사용자 문서 문구 그대로 재현)
d["_순수운전"] = d["여신_운전자금대출잔액"] - d["운전_할인어음잔액"] - d["운전_무역금융잔액"]
neg_p = d["_순수운전"] < 0
log(f"\n- 순수운전자금(운전자금 − 할인어음 − 무역금융, 자체 재현) 음수 {int(neg_p.sum()):,}행({neg_p.mean():.3%}), "
    f"음수 법인 {d.loc[neg_p, '법인ID'].nunique()}곳, 최솟값 {d['_순수운전'].min():.2f}")
if neg_p.any():
    log(f"    음수인 경우 구성: 운전자금 합계 중앙값 {d.loc[neg_p, '여신_운전자금대출잔액'].median():.2f} vs 할인어음+무역금융 중앙값 {(d.loc[neg_p, '운전_할인어음잔액'] + d.loc[neg_p, '운전_무역금융잔액']).median():.2f}")
log("  (주의: 이 정의는 사용자 요청 문서의 표현을 그대로 재현한 것으로, 팀 공식 정의 문서를 로컬에서 찾지 못했다 — 확인 필요)")

# ---------------- 3. 반올림 영향 ----------------
log("\n## 3. 반올림 영향 (요구불예금잔액 기준)")
col = "요구불예금잔액"
x = d[["법인ID", "기준년월", col]].copy()
midx = (x["기준년월"] // 100) * 12 + x["기준년월"] % 100
fid = pd.factorize(x["법인ID"])[0].astype(np.int64)
key = pd.Index(fid * 10000 + midx.to_numpy())


def at(arr, off):
    pos = key.get_indexer(fid * 10000 + midx.to_numpy() + off)
    out = np.full(len(arr), np.nan); ok = pos >= 0; out[ok] = np.asarray(arr, float)[pos[ok]]
    return out


v = x[col].to_numpy(float)
for k in [1, 3, 6]:
    fwd = at(v, k)
    ok = ~np.isnan(fwd)
    same = np.isclose(v[ok], fwd[ok])
    log(f"- {k}개월 뒤 잔액 완전 동일(반올림 포함) 비율: {same.mean():.1%} ({int(same.sum()):,}/{int(ok.sum()):,})")
    # 규모 구간별
    q = pd.qcut(pd.Series(v[ok]).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    g = pd.Series(same, index=np.where(ok)[0]).groupby(q.to_numpy()).mean()
    log(f"    규모 5분위(1=낮음)별 동일 유지율: " + ", ".join(f"Q{i} {r:.1%}" for i, r in g.items()))
# 반올림 경계 부근 변화: 변화폭이 0.01(최소 단위 추정) 이하인 비율
d1 = at(v, 1) - v
nz = ~np.isnan(d1) & (v > 0)
tiny = nz & (d1 != 0) & (np.abs(d1) <= 0.02)
log(f"- 1개월 변화 중 0은 아니지만 매우 작은(|Δ|≤0.02) 변화 비율: {tiny.sum() / nz.sum():.2%} (측정 정밀도 근처 변화)")

# ---------------- 4. 관측 구조 ----------------
log("\n## 4. 관측 구조 (법인 단위, 36개월 만재 기준 대구·경북 11,036곳)")
g = d.groupby("법인ID")["기준년월"]
first, last, n = g.min(), g.max(), g.size()
log(f"- 첫 관측월 분포: 2023-01 {int((first == 202301).sum()):,}곳, 그 외 {int((first != 202301).sum()):,}곳 (최빈 첫 관측월 {int(first.mode().iloc[0])})")
log(f"- 마지막 관측월 분포: 2025-12 {int((last == 202512).sum()):,}곳, 그 외 {int((last != 202512).sum()):,}곳")
full = (first == 202301) & (last == 202512) & (n == 36)
log(f"- 36개월 완전 균형(첫~끝 연속) 법인: {int(full.sum()):,}곳 ({full.mean():.1%})")
expected_span = ((last // 100) * 12 + last % 100) - ((first // 100) * 12 + first % 100) + 1
gap = expected_span - n
log(f"- 관측 구간(첫~끝) 안에서 중간 누락 있는 법인: {int((gap > 0).sum()):,}곳 (누락 개월 수 중앙값 {gap[gap > 0].median() if (gap > 0).any() else 0:.0f})")
log(f"- 연속 관측 개월 수(n) 분포: p10 {n.quantile(.1):.0f}, 중앙값 {n.median():.0f}, p90 {n.quantile(.9):.0f}")

LOG.close()
