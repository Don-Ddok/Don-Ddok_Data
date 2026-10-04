# -*- coding: utf-8 -*-
"""묶음 B — 동일 기업의 계정·거래활동 동반 변화 (회귀 없음, 기술 통계·집계만)
디벨롭 문서 §5 묶음 B에 대응. "요구불에서 빠진 돈이 저축성으로 이동했다"는 식의 인과 서술은 하지 않는다.
동반 변화는 금융적 설명의 단서일 뿐, 실제 이체·자금이동·자금압박의 입증이 아니다.

설계 (문서 그대로):
① 두 가지 표본 병행: A=전체 관측 표본(계정별로 다름, 기존 결과와 연결) / B=동일 평가창 공통 기업군(요구불+상대 계정 둘 다 t-1·t+h 관측)
② 상태 변화와 잔액 변화 분리: 0→0 / 0→양수 / 양수→0 / 양수→양수 / 판단불가(결측 포함)
③ 동반 변화 유형(사전 고정 5개 조합), 단월(h=1)과 3개월 누적(h=3) 병행. 잔액은 끝점 비교, 흐름(입출금·거래건수)은 기간 합계.
표본: 대구·경북, 노출·비노출 모두. 노출별로도 갈라 보고한다.
출력: bundle_b_summary.txt, bundle_b_*.csv (집계만, 법인 ID 없음)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "bundle_b_summary.txt"), "w", encoding="utf-8")
BUCKETS = ["0건", "1건", "2건", "2건초과 5건이하", "5건초과 10건이하", "10건초과 20건이하",
           "20건초과 30건이하", "30건초과 40건이하", "40건초과 50건이하", "50건 초과"]
BRANK = {b: i for i, b in enumerate(BUCKETS)}
CHAN = ["창구거래건수", "인터넷뱅킹거래건수", "스마트뱅킹거래건수", "폰뱅킹거래건수", "ATM거래건수", "자동이체거래건수"]


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


cols = (["기준년월", "법인ID", "사업장_시도", "exposed", "요구불예금잔액", "여신_운전자금대출잔액",
         "거치식예금잔액", "적립식예금잔액", "요구불입금금액", "요구불출금금액",
         "외환_수출실적금액", "외환_수입실적금액"] + CHAN)
d = pd.read_csv(DATA, usecols=cols)
d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
d = d.rename(columns={"exposed": "노출"})
for c in CHAN:
    d[c] = d[c].map(BRANK)
d["채널활동_합"] = d[CHAN].sum(axis=1)          # 순위 합(활동 강도 대용, 회귀 없음·기술통계용)
d["외환금액_합"] = d["외환_수출실적금액"] + d["외환_수입실적금액"]
d = d.sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
midx = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
fid = pd.factorize(d["법인ID"])[0].astype(np.int64)
key = pd.Index(fid * 10000 + midx.to_numpy())
log(f"# 묶음 B — 동일 기업의 계정·거래활동 동반 변화 (회귀 없음)\n대상: 대구·경북 {d['법인ID'].nunique():,}법인, {len(d):,}행\n")


def at(arr, off):
    pos = key.get_indexer(fid * 10000 + midx.to_numpy() + off)
    out = np.full(len(arr), np.nan); ok = pos >= 0; out[ok] = np.asarray(arr, float)[pos[ok]]
    return out


def sumwin(arr, lo, hi):
    """[t+lo, t+hi] 구간 합 (흐름 변수용, 둘 다 관측될 때만)"""
    s = np.zeros(len(arr)); ok = np.ones(len(arr), bool)
    for k in range(lo, hi + 1):
        v = at(arr, k); ok &= ~np.isnan(v); s = s + np.nan_to_num(v)
    s[~ok] = np.nan
    return s


# ---------------- ② 상태 전환표 ----------------
log("## 2. 상태 변화 (0→0 / 0→양수 / 양수→0 / 양수→양수), h=1과 h=6")
BAL = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}
trans_rows = []
for h in [1, 6]:
    for name, col in BAL.items():
        v = d[col].to_numpy(float)
        v1 = at(v, -1); v2 = at(v, h)
        ok = ~np.isnan(v1) & ~np.isnan(v2)
        s1 = np.where(v1[ok] > 0, "양수", "0"); s2 = np.where(v2[ok] > 0, "양수", "0")
        tab = pd.crosstab(s1, s2)
        n_defined, n_total = int(ok.sum()), len(v)
        log(f"- {name} h={h}: 판단 가능 {n_defined:,}/{n_total:,}행({n_defined/n_total:.1%}), "
            f"0→0 {tab.loc['0','0'] if '0' in tab.index and '0' in tab.columns else 0:,} / "
            f"0→양수 {tab.loc['0','양수'] if '0' in tab.index and '양수' in tab.columns else 0:,} / "
            f"양수→0 {tab.loc['양수','0'] if '양수' in tab.index and '0' in tab.columns else 0:,} / "
            f"양수→양수 {tab.loc['양수','양수'] if '양수' in tab.index and '양수' in tab.columns else 0:,}")
        for a in ["0", "양수"]:
            for b in ["0", "양수"]:
                n = int(tab.loc[a, b]) if a in tab.index and b in tab.columns else 0
                trans_rows.append({"계정": name, "h": h, "이전": a, "이후": b, "행": n})
pd.DataFrame(trans_rows).to_csv(os.path.join(OUT, "bundle_b_transitions.csv"), index=False, encoding="utf-8-sig")

# ---------------- ①+③ 동반 변화 ----------------
log("\n## 3. 동반 변화 (사전 고정 5개 조합)")
dd_ln = np.log(np.clip(d["요구불예금잔액"].to_numpy(float), 0, None) + 1)
sav_ln = np.log(np.clip(d["거치식예금잔액"].to_numpy(float) + d["적립식예금잔액"].to_numpy(float), 0, None) + 1)
loan_ln = np.log(np.clip(d["여신_운전자금대출잔액"].to_numpy(float), 0, None) + 1)
fx_ln = np.log(np.clip(d["외환금액_합"].to_numpy(float), 0, None) + 1)
inout = d["요구불입금금액"].to_numpy(float) + d["요구불출금금액"].to_numpy(float)   # 흐름 총량(활동 규모)
chan = d["채널활동_합"].to_numpy(float)

combos = []
for h, tag in [(1, "단월(h=1)"), (3, "3개월 누적(h=3)")]:
    dd_chg = at(dd_ln, h) - at(dd_ln, -1)
    sav_chg = at(sav_ln, h) - at(sav_ln, -1)
    loan_chg = at(loan_ln, h) - at(loan_ln, -1)
    if h == 1:
        inout_chg = at(inout, h) - at(inout, -1)
        fx_chg = at(fx_ln, h) - at(fx_ln, -1)
        chan_chg = at(chan, h) - at(chan, -1)
    else:
        inout_pre = sumwin(inout, -3, -1); inout_post = sumwin(inout, 1, h)
        inout_chg = inout_post - inout_pre
        fx_pre = sumwin(np.clip(d["외환금액_합"].to_numpy(float), 0, None), -3, -1)
        fx_post = sumwin(np.clip(d["외환금액_합"].to_numpy(float), 0, None), 1, h)
        fx_chg = np.log(fx_post + 1) - np.log(fx_pre + 1)
        chan_pre = sumwin(chan, -3, -1); chan_post = sumwin(chan, 1, h)
        chan_chg = chan_post - chan_pre
    pairs = [("요구불↓·저축성↑", dd_chg, sav_chg, "<", ">"), ("요구불↓·대출유지or↑", dd_chg, loan_chg, "<", ">="),
             ("요구불↓·입출금↓", dd_chg, inout_chg, "<", "<"), ("요구불↓·외환금액↓", dd_chg, fx_chg, "<", "<"),
             ("요구불↓·채널활동↓", dd_chg, chan_chg, "<", "<")]
    for label, x, y, opx, opy in pairs:
        ok = ~np.isnan(x) & ~np.isnan(y)
        down = ok & (x < 0)
        n_ok, n_down = int(ok.sum()), int(down.sum())
        if opy == ">":
            hit = down & (y > 0)
        elif opy == ">=":
            hit = down & (y >= 0)
        else:
            hit = down & (y < 0)
        rate = hit.sum() / n_down if n_down else np.nan
        # 노출별
        by_expo = {}
        for e in [0, 1]:
            m = down & (d["노출"].to_numpy() == e)
            by_expo[e] = (hit[m].sum() / m.sum()) if m.sum() else np.nan
        log(f"- [{tag}] {label}: 요구불↓ {n_down:,}행(판단가능 {n_ok:,}행 중) → 동반조건 충족 {rate:.1%} "
            f"(비노출 {by_expo[0]:.1%}, 노출 {by_expo[1]:.1%})")
        combos.append({"기간": tag, "조합": label, "요구불하락_행": n_down, "판단가능_행": n_ok, "동반충족률": rate,
                       "동반충족률_비노출": by_expo[0], "동반충족률_노출": by_expo[1]})
pd.DataFrame(combos).to_csv(os.path.join(OUT, "bundle_b_combos.csv"), index=False, encoding="utf-8-sig")

# ---------------- ① 동일 평가창 공통 기업군 (h=6, 주요 4계정 전부 정의되는 법인만) ----------------
log("\n## 4. 표본 비교: 전체 관측 표본 vs 동일 평가창 공통 기업군 (h=6)")
need = [~np.isnan(at(d[c].to_numpy(float), -1)) & ~np.isnan(at(d[c].to_numpy(float), 6)) for c in BAL.values()]
common = need[0] & need[1] & need[2] & need[3]
log(f"- 전체 법인×월: {len(d):,}행")
for name, ok in zip(BAL, need):
    log(f"  {name} 단독 t-1·t+6 관측: {int(ok.sum()):,}행 ({ok.mean():.1%})")
log(f"- 네 계정 모두 동시에 t-1·t+6 관측되는 공통 기업×월(동일 평가창 공통 기업군): {int(common.sum()):,}행 "
    f"({common.mean():.1%}), 법인 {d.loc[common, '법인ID'].nunique():,}곳")
log("  → '요구불에서 빠진 돈이 저축성으로 이동했다'는 서술은, 네 계정 모두 겹쳐 관측되는 이 공통 기업군 안에서도 "
    "상관·동반 비율일 뿐 자금 이동의 직접 증거가 아니다 (해석 한계).")

LOG.close()
