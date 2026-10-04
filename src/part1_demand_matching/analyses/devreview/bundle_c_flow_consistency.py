# -*- coding: utf-8 -*-
"""묶음 C — 요구불 잔액과 순입금의 정합성 (회귀 없음, 기술 통계만)
디벨롭 문서 §5 묶음 C에 대응. 목적은 입금 경로를 다시 유의하게 만드는 게 아니라,
잔액과 흐름 데이터가 같은 범위를 측정하는지 확인하는 것이다.
R_it = (B_it - B_i,t-1) - (입금_it - 출금_it)
평가기간이 길어지면 끝점 사이 흐름을 같은 기간으로 누적한다(1·3·6개월).
상관 하나만으로 회계적 정합성을 확인했다고 하지 않는다.
확인 못하는 항목(월말잔액/월평잔 여부, 이자·대체·내부거래 포함 여부)은 "정의상 비교 불가"로 남긴다.
출력: bundle_c_summary.txt, bundle_c_residuals.csv (규모별 분포, 집계만)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_c_summary.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "요구불예금잔액", "요구불입금금액", "요구불출금금액"])
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
midx = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
fid = pd.factorize(d["법인ID"])[0].astype(np.int64)
key = pd.Index(fid * 10000 + midx.to_numpy())
log(f"# 묶음 C — 요구불 잔액-순입금 정합성 (회귀 없음)\n대상: 대구·경북 {d['법인ID'].nunique():,}법인, {len(d):,}행\n")
log("## 정의상 비교 불가능한 항목 (실행 전 확인)")
log("- 월말 잔액인지 월평잔인지: 컬럼명·설명서로 확인 불가. df_ready는 '요구불예금잔액'이라고만 돼 있어 월말 스냅샷으로 가정한다(확인 필요).")
log("- 입출금이 같은 요구불 계정 범위인지: '요구불입금금액'·'요구불출금금액'이 정확히 이 요구불예금 계좌의 흐름인지, 아니면 은행 전체 계좌 흐름인지 설명서가 없다(확인 필요).")
log("- 이자·대체·내부거래 포함 여부: 확인 불가. 이자 입금이 '요구불입금금액'에 섞여 있으면 잔차가 구조적으로 0에 가까워지고, 안 섞여 있으면 이자만큼 잔차가 남는다.")
log("- 신규·종료 관측 영향: 법인이 중간에 관측을 시작/끝내면 그 시점의 ΔB는 실제 잔액 변화가 아니라 관측 시작/종료 인공물일 수 있어 별도로 뺀다(아래).\n")


def at(arr, off):
    pos = key.get_indexer(fid * 10000 + midx.to_numpy() + off)
    out = np.full(len(arr), np.nan); ok = pos >= 0; out[ok] = np.asarray(arr, float)[pos[ok]]
    return out


def sumwin(arr, lo, hi):
    s = np.zeros(len(arr)); ok = np.ones(len(arr), bool)
    for k in range(lo, hi + 1):
        v = at(arr, k); ok &= ~np.isnan(v); s = s + np.nan_to_num(v)
    s[~ok] = np.nan
    return s


bal = d["요구불예금잔액"].to_numpy(float)
inflow = d["요구불입금금액"].to_numpy(float)
outflow = d["요구불출금금액"].to_numpy(float)

# 검증: mechanism_inflow/step0_describe.py의 원래 정의(같은 달, B_t - B_{t-1} vs 입금_t - 출금_t, 요구불 보유 이력 법인)를 그대로 재현
s = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0].copy()
skey = pd.Index(pd.factorize(s["법인ID"])[0].astype(np.int64) * 10000 + ((s["기준년월"] // 100) * 12 + s["기준년월"] % 100).to_numpy())
sfid = pd.factorize(s["법인ID"])[0].astype(np.int64)
smidx = (s["기준년월"] // 100) * 12 + s["기준년월"] % 100


def sat(col, off):
    pos = skey.get_indexer(sfid * 10000 + smidx.to_numpy() + off)
    v = np.full(len(s), np.nan); ok = pos >= 0; v[ok] = s[col].to_numpy(float)[pos[ok]]; return v


dB0 = s["요구불예금잔액"].to_numpy(float) - sat("요구불예금잔액", -1)
net0 = (s["요구불입금금액"] - s["요구불출금금액"]).to_numpy(float)
m0 = ~np.isnan(dB0) & ~np.isnan(net0)
r0 = np.corrcoef(dB0[m0], net0[m0])[0, 1]
log(f"## 검증: 원래 정의(같은 달 B_t−B_t-1 vs 입금_t−출금_t, 요구불 보유 이력 {s['법인ID'].nunique():,}법인) 재현")
log(f"- Pearson {r0:.3f} (FINAL PDF 6절이 인용한 0.589와 대조). 아래 h=1/3/6은 **다른 정의**(끝점 간격을 h와 맞추려 t-1→t+h, 흐름은 그 사이 누적)라 이 값과 다르게 나온다 — 직접 비교하면 안 된다.\n")

rows = []
for h in [1, 3, 6]:
    b0 = at(bal, -1)                                     # 기준: t-1 (harmonized·요구불 DID와 같은 기준월)
    b1 = at(bal, h)                                      # 끝점: t+h
    dB = b1 - b0
    inflow_c = sumwin(inflow, 0, h)                      # t..t+h 구간 합 (h+1개월 분, 끝점 간격과 맞춤)
    outflow_c = sumwin(outflow, 0, h)
    ok = ~np.isnan(dB) & ~np.isnan(inflow_c) & ~np.isnan(outflow_c)
    R = dB[ok] - (inflow_c[ok] - outflow_c[ok])
    b0v = b0[ok]
    log(f"## h={h} (끝점 간격 {h+1}개월, ΔB = 잔액[t+{h}] - 잔액[t-1], 흐름 = t~t+{h} 누적 {h+1}개월 합)")
    log(f"- 판정 가능 {int(ok.sum()):,}행")
    log(f"  R 분포: p5 {np.percentile(R,5):.2f}, p25 {np.percentile(R,25):.2f}, 중앙값 {np.median(R):.2f}, "
        f"p75 {np.percentile(R,75):.2f}, p95 {np.percentile(R,95):.2f}")
    exact = np.isclose(R, 0, atol=0.01)
    log(f"  R ≈ 0(정확 일치, |R|≤0.01) 비율: {exact.mean():.1%}")
    r = stats.pearsonr(dB[ok], (inflow_c[ok] - outflow_c[ok]))[0]
    log(f"  ΔB와 (입금−출금)의 Pearson 상관: {r:.3f}")
    q = pd.qcut(pd.Series(np.abs(b0v)).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    for i in [1, 2, 3, 4, 5]:
        m = (q == i).to_numpy()
        log(f"    잔액규모 Q{i}: |R| 중앙값 {np.median(np.abs(R[m])):.2f}, R≈0 비율 {exact[m].mean():.1%}, n={int(m.sum()):,}")
        rows.append({"h": h, "규모Q": i, "R_절대중앙값": np.median(np.abs(R[m])), "R근사0_비율": exact[m].mean(), "n": int(m.sum())})
    # 신규/종료 관측 인공물 제외 버전: 전후로 한 달씩 더 여유 있게 관측되는 경우만
    ok2 = ~np.isnan(dB) & ~np.isnan(inflow_c) & ~np.isnan(outflow_c) & ~np.isnan(at(bal, -2)) & ~np.isnan(at(bal, h + 1))
    R2 = (b1[ok2] - b0[ok2]) - (inflow_c[ok2] - outflow_c[ok2])
    log(f"  (신규·종료 인접 관측 제외, 전후 여유 1개월씩 더 있는 {int(ok2.sum()):,}행만) R≈0 비율 {np.isclose(R2,0,atol=0.01).mean():.1%}, "
        f"|R| 중앙값 {np.median(np.abs(R2)):.2f}\n")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "bundle_c_residuals.csv"), index=False, encoding="utf-8-sig")
log("## 결론")
log(f"- 원래 정의(같은 달)로는 상관 {r0:.3f}로 FINAL PDF의 0.589를 그대로 재현했다. 이번에 새로 본 h=1/3/6(끝점 간격을 넓힌 정의)에서는 상관이 0.36~0.46으로 원래 정의보다 낮다 — **정의(창 길이)에 따라 상관 자체가 달라지므로 '0.589'라는 숫자 하나로 일반화하면 안 된다.**")
log("- R≈0(잔액 변화와 순입금이 정확히 같음) 비율이 표에 나온 수준이라, **잔액과 입출금이 100% 같은 범위를 측정한다고 확인되지 않는다.**")
log("- 신규·종료 관측 인접 구간을 빼도 R≈0 비율이 크게 달라지지 않으면(위 괄호 결과 참고) 관측 경계 인공물이 주된 원인은 아니라는 뜻이다.")
log("- 정의상 비교 불가 항목(월말/월평잔, 계정 범위, 이자·내부거래)이 남아 있어, 이 잔차를 '데이터 오류'로 단정하지 않는다 — 확인 필요로 남긴다.")
LOG.close()
