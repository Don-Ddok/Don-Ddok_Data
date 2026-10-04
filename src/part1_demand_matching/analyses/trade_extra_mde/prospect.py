# -*- coding: utf-8 -*-
"""무역 추가 분석 3종의 사전 MDE 판단 — 처치 계수(β)는 계산하지 않음
0단계: A 상거래 결제 채널 계정 / B 수출 중단·시작 사건 / C 순노출 × 환율 기술 통계
1단계: h=6 MDE 근사
  SE ≈ k · σ(ỹ) / (sd(X̃) · √N)
    ỹ = h=6 종속변수(윈저 1/99)에서 법인 FE + 지역×연월 FE (+ 통제)를 뺀 잔차 (처치 없는 귀무 모형)
    X̃ = 처치 변수에서 같은 FE (+ 통제)를 뺀 잔차
    k = 군집 설계효과. harmonized 요구불 C1 h=6 SE(0.1135)에 맞춰 한 번 정하고, 다른 harmonized 결과로 검증
  보조: 귀무 잔차 샌드위치 SE (ψ = X̃·ỹ / ΣX̃², 법인·연월 이중 군집, harmonized와 같은 보정)
  MDE(β) = (t.975 + t.80) × SE, 자유도 = min(법인, 월) − 1
출력: step0_*.csv, calibration.csv, mde_prospect.csv, prospect_log.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, partial_out

DATA = r"C:\test\data\processed\df_ready.csv"
FXF = r"C:\test\external\Don-Ddok_Data\data\external\환율_ECOS원자료_202101_202512.csv"
OUT = r"C:\test\outputs\trade_extra_mde"
REF = {("요구불", "C1"): 0.113492621667047, ("운전자금", "C1"): 0.1144132454026595, ("거치식", "C1"): 0.4899053529855707,
       ("적립식", "C1"): 0.3672776203636709, ("요구불", "E23"): 0.2165112755137031}
REF_N = {("요구불", "C1"): 184846, ("운전자금", "C1"): 165674, ("거치식", "C1"): 16971, ("적립식", "C1"): 14674, ("요구불", "E23"): 114712}
ACC = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액",
       "할인어음": "운전_할인어음잔액", "외상매출채권담보": "운전_외상매출채권담보대출잔액", "기업구매자금": "운전_기업구매자금대출잔액"}
LOGF = open(os.path.join(OUT, "prospect_log.txt"), "w", encoding="utf-8")
MULT = lambda dof: stats.t.ppf(0.975, dof) + stats.t.ppf(0.80, dof)


def log(s=""):
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def load():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "외환_수출실적금액", "외환_수입실적금액"] + list(ACC.values()))
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    e23 = d[d["ym"] < 202401].assign(a=lambda x: (x["외환_수출실적금액"] > 0) | (x["외환_수입실적금액"] > 0)).groupby("법인ID")["a"].max()
    d["노출_2023"] = d["법인ID"].map(e23).fillna(False).astype(int)
    g23 = d[d["ym"] < 202401].groupby("법인ID")[["외환_수출실적금액", "외환_수입실적금액"]].sum()
    tot = g23.sum(axis=1)
    net = ((g23["외환_수출실적금액"] - g23["외환_수입실적금액"]) / tot.where(tot > 0))
    d["_keep_net"] = d["법인ID"].map(net)
    d["_keep_exp"] = d["외환_수출실적금액"]
    return d


def components(P, y, X, m, SB=None, winsor=True):
    """처치 없는 귀무 모형의 잔차로 SE 근사 재료 계산 (β 없음)"""
    yy = y[m].copy()
    if winsor:
        lo, hi = np.quantile(yy, [0.01, 0.99]); yy = np.clip(yy, lo, hi)
    fi = np.unique(P.firm[m], return_inverse=True)[1]; ri = np.unique(P.regym[m], return_inverse=True)[1]
    cols = [yy, X[m]] + ([] if SB is None else [SB[m]])
    M = partial_out(np.column_stack(cols), np.ones(int(m.sum())), fi, ri)
    ey, ex = M[:, 0], M[:, 1]
    if SB is not None:                                     # 통제항도 뺀다 (FWL)
        sb = M[:, 2]
        ey = ey - sb * (sb @ ey) / (sb @ sb); ex = ex - sb * (sb @ ex) / (sb @ sb)
    n = len(ey); k = 1 if SB is None else 2
    psi = ex * ey / (ex @ ex)
    V = 0.0
    for g, sign in [(P.firm[m], 1), (P.month[m], 1), (P.obs[m], -1)]:
        gu, gi = np.unique(g, return_inverse=True); G = len(gu)
        V += sign * G / (G - 1) * (n - 1) / (n - k) * float(np.sum(np.bincount(gi, psi) ** 2))
    Gf, Gm = len(np.unique(P.firm[m])), len(np.unique(P.month[m]))
    return {"N": n, "법인": Gf, "월": Gm, "자유도": min(Gf, Gm) - 1, "σ_y": ey.std(), "sd_X": ex.std(),
            "raw": ey.std() / (ex.std() * np.sqrt(n)), "SE_샌드위치": np.sqrt(max(V, 0))}


def dep6(P, hold=False):
    base = P.pos if hold else P.ln
    return P.at(base, 6) - P.at(base, -1)


def main():
    d = load()
    rows_cal, rows = [], []
    # ---------------- 보정(calibration): harmonized C1·E23 재현 ----------------
    log("## 보정: harmonized h=6 SE 재현 (k는 요구불 C1에서 결정)")
    for (acct, model), se_ref in REF.items():
        col = ACC[acct]
        s = d[d.groupby("법인ID")[col].transform("max") > 0]
        expo = "노출_2023" if model == "E23" else "외환노출"
        P = Panel(s, col, expo_col=expo)
        y = dep6(P); X = P.expo * P.X["보정"]; SB = P.expo * P.biz
        m = ~np.isnan(y) & ((P.ymv >= 202401) if model == "E23" else True)
        c = components(P, y, X, m, SB)
        c.update({"계정": acct, "모형": model, "SE_harmonized": se_ref, "N_harmonized": REF_N[(acct, model)]})
        rows_cal.append(c)
    cal = pd.DataFrame(rows_cal)
    k = cal.loc[(cal["계정"] == "요구불") & (cal["모형"] == "C1"), "SE_harmonized"].iloc[0] / cal.loc[(cal["계정"] == "요구불") & (cal["모형"] == "C1"), "raw"].iloc[0]
    cal["SE_근사"] = k * cal["raw"]
    cal["근사/실제"] = cal["SE_근사"] / cal["SE_harmonized"]
    cal["샌드위치/실제"] = cal["SE_샌드위치"] / cal["SE_harmonized"]
    cal.to_csv(os.path.join(OUT, "calibration.csv"), index=False, encoding="utf-8-sig")
    log(f"k = {k:.3f}")
    log(cal[["계정", "모형", "N", "N_harmonized", "월", "σ_y", "sd_X", "SE_harmonized", "SE_근사", "근사/실제", "SE_샌드위치", "샌드위치/실제"]].round(4).to_string(index=False))

    def add(label, acct, c, conv, unit):
        se_k = k * c["raw"]
        for how, se in [("근사(k)", se_k), ("샌드위치", c["SE_샌드위치"])]:
            mde = MULT(c["자유도"]) * se
            lo, hi = conv(mde)
            rows.append({"분석": label, "계정": acct, "방식": how, "N": c["N"], "법인": c["법인"], "월": c["월"], "자유도": c["자유도"],
                         "σ_y": c["σ_y"], "sd_X": c["sd_X"], "SE": se, "MDE_β": mde, "MDE_작은쪽": lo, "MDE_큰쪽": hi, "단위": unit})

    per10 = lambda mde: (abs(np.exp(-0.1 * mde) - 1) * 100, (np.exp(0.1 * mde) - 1) * 100)     # 10%p당
    ev = lambda mde: ((1 - np.exp(-mde)) * 100, (np.exp(mde) - 1) * 100)                        # 사건 1회당

    # ---------------- A. 상거래 결제 채널 ----------------
    log("\n## 0단계 A: 상거래 결제 채널 계정")
    trend = []
    for acct in ["할인어음", "외상매출채권담보", "기업구매자금"]:
        col = ACC[acct]
        f = d.groupby("법인ID").agg(hold=(col, lambda v: (v > 0).any()), expo=("외환노출", "max"))
        s = d[d.groupby("법인ID")[col].transform("max") > 0]
        P = Panel(s, col)
        y = dep6(P); both = ~np.isnan(y)
        zero_ch = np.isclose(P.at(P.bal, 6)[both], P.at(P.bal, -1)[both]).mean()
        pos_share_all = (d[col] > 0).mean()
        log(f"- {acct}: 보유 법인 노출 {int((f.hold & (f.expo == 1)).sum()):,} / 비노출 {int((f.hold & (f.expo == 0)).sum()):,} "
            f"/ 전체 관측 중 0 비율 {1 - pos_share_all:.1%} / 보유 이력 법인 관측 중 0 비율 {(s[col] <= 0).mean():.1%} "
            f"/ t−1→t+6 잔액 변화 0 비율 {zero_ch:.1%} (보유 이력 법인, {int(both.sum()):,}쌍)")
        mt = d.assign(p=d[col] > 0).groupby(["ym", "외환노출"])["p"].sum().unstack()
        mt.columns = [f"{acct}_비노출", f"{acct}_노출"]
        trend.append(mt)
        log(f"    월별 보유 법인(노출/비노출): 2023-01 {int(mt.iloc[0, 1])}/{int(mt.iloc[0, 0])}, 2024-06 {int(mt.loc[202406].iloc[1])}/{int(mt.loc[202406].iloc[0])}, "
            f"2025-12 {int(mt.iloc[-1, 1])}/{int(mt.iloc[-1, 0])} (노출 최소 {int(mt.iloc[:, 1].min())}, 최대 {int(mt.iloc[:, 1].max())})")
        c = components(P, y, P.expo * P.X["보정"], both, P.expo * P.biz)
        add("A 결제채널", acct, c, per10, "% (지역 수출 10%p당)")
    pd.concat(trend, axis=1).to_csv(os.path.join(OUT, "step0_A_holders_by_month.csv"), encoding="utf-8-sig")

    # ---------------- B. 수출 중단·시작 사건 ----------------
    log("\n## 0단계 B: 수출 중단·시작 사건")
    exp_ids = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
    dx = d[d["법인ID"].isin(exp_ids[exp_ids].index)]
    P0 = Panel(dx, "요구불예금잔액")
    e = P0.d["_keep_exp"].to_numpy(float)
    pos = lambda off: P0.at(e, off) > 0
    zero = lambda off: P0.at(e, off) == 0          # nan(관측 없음)은 0으로 보지 않음
    stop = pos(-1) & np.all([zero(o) for o in range(0, 6)], axis=0)
    start = pos(0) & np.all([zero(-o) for o in range(1, 7)], axis=0)
    full = np.all([~np.isnan(P0.at(e, o)) for o in range(-6, 13)], axis=0)
    ev_tab = []
    for name, D in [("중단", stop), ("시작", start)]:
        ids = P0.d.loc[D, "법인ID"]; yms = P0.ymv[D]
        per = ids.value_counts()
        log(f"- {name}: 사건 {int(D.sum()):,}건, 법인 {ids.nunique():,}곳 (수출 법인 {exp_ids.sum():,}곳 중), 2회 이상인 법인 {(per >= 2).mean():.1%}")
        log(f"    사건 시점(연도-분기): " + ", ".join(f"{q} {n}" for q, n in pd.Series(yms // 100 * 10 + ((yms % 100) - 1) // 3 + 1).value_counts().sort_index().items()))
        log(f"    h=−6~12 관측을 모두 갖춘 사건 {int((D & full).sum()):,}건 / 법인 {P0.d.loc[D & full, '법인ID'].nunique():,}곳")
        ev_tab.append({"사건": name, "건": int(D.sum()), "법인": ids.nunique(), "2회이상_비율": (per >= 2).mean(),
                       "h-6_12_완비_건": int((D & full).sum()), "h-6_12_완비_법인": P0.d.loc[D & full, "법인ID"].nunique()})
    pd.DataFrame(ev_tab).to_csv(os.path.join(OUT, "step0_B_events.csv"), index=False, encoding="utf-8-sig")
    log("    정의: 중단 t = 수출 > 0이 t−1에 있고 t~t+5 연속 0 (직전 3개월 중 ≥1회 조건을 만족하는 첫 0인 달로 고정)")
    log("          시작 t = t−6~t−1 연속 0이고 t에 수출 > 0 (중단의 거울)")
    for acct in ["요구불", "운전자금", "적립식"]:
        col = ACC[acct]
        s = dx[dx.groupby("법인ID")[col].transform("max") > 0]
        P = Panel(s, col)
        ee = P.d["_keep_exp"].to_numpy(float)
        ps = lambda off: P.at(ee, off) > 0
        zr = lambda off: P.at(ee, off) == 0
        Ds = {"중단": (ps(-1) & np.all([zr(o) for o in range(0, 6)], axis=0)).astype(float),
              "시작": (ps(0) & np.all([zr(-o) for o in range(1, 7)], axis=0)).astype(float)}
        y = dep6(P)
        m = ~np.isnan(y) & (P.ymv >= 202307)            # 두 사건 모두 정의 가능한 달부터 (시작은 t−6 필요)
        for name, D in Ds.items():
            c = components(P, y, D, m)
            add(f"B 수출{name}", acct, c, ev, "% (사건 1회당)")

    # ---------------- C. 순노출 × 환율 ----------------
    log("\n## 0단계 C: 순노출 (2023년 수출·수입 실적 기준)")
    f = d[d["ym"] < 202401].groupby("법인ID")[["외환_수출실적금액", "외환_수입실적금액"]].sum()
    f = f[f.sum(axis=1) > 0]
    net = (f["외환_수출실적금액"] - f["외환_수입실적금액"]) / f.sum(axis=1)
    kind = np.where(net == 1, "수출만", np.where(net == -1, "수입만", "양쪽"))
    log(f"- 2023년 실적 있는 법인 {len(net):,}곳: 수출만 {np.mean(kind == '수출만'):.1%}, 수입만 {np.mean(kind == '수입만'):.1%}, 양쪽 {np.mean(kind == '양쪽'):.1%}")
    log(f"    순노출 분포: 평균 {net.mean():+.3f}, SD {net.std():.3f}, p10 {net.quantile(.1):+.2f}, p25 {net.quantile(.25):+.2f}, 중앙값 {net.median():+.2f}, "
        f"p75 {net.quantile(.75):+.2f}, p90 {net.quantile(.9):+.2f}")
    both_net = net[kind == "양쪽"]
    log(f"    양쪽 법인만: 평균 {both_net.mean():+.3f}, SD {both_net.std():.3f}, 중앙값 {both_net.median():+.2f}")
    pd.DataFrame({"구분": ["수출만", "수입만", "양쪽"], "법인": [int(np.sum(kind == g)) for g in ["수출만", "수입만", "양쪽"]]}).to_csv(
        os.path.join(OUT, "step0_C_net.csv"), index=False, encoding="utf-8-sig")
    fx = pd.read_csv(FXF, encoding="utf-8-sig").set_index("ym")["전년동월비_%"] / 100
    fx24 = fx.loc[202401:202506]
    log(f"    원/달러 전년동월비 2024-01~2025-06: 평균 {fx24.mean():+.1%}, SD {fx24.std():.1%}, 최소 {fx24.min():+.1%}, 최대 {fx24.max():+.1%}")
    log(f"    (참고) 지역 수출 보정YoY와 비교: 같은 기간 SD는 calibration의 sd_X 참조")
    s = d[(d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0) & d["_keep_net"].notna()]
    P = Panel(s, "요구불예금잔액", expo_col="노출_2023")
    X = P.d["_keep_net"].to_numpy(float) * P.d["ym"].map(fx).to_numpy(float)
    y = dep6(P)
    m = ~np.isnan(y) & (P.ymv >= 202401)
    c = components(P, y, X, m)
    add("C 순노출×환율", "요구불", c, per10, "% (순노출 1 법인, 환율 YoY 10%p당)")

    out = pd.DataFrame(rows)
    out["판단_기준값"] = out["MDE_큰쪽"]
    out["판단"] = np.where(out["판단_기준값"] <= 5, "진행 후보", np.where(out["판단_기준값"] <= 10, "애매(단일 가설만)", "진행 안 함"))
    out.to_csv(os.path.join(OUT, "mde_prospect.csv"), index=False, encoding="utf-8-sig")
    log("\n## 1단계 MDE (h=6)")
    log(out[["분석", "계정", "방식", "N", "법인", "월", "자유도", "SE", "MDE_작은쪽", "MDE_큰쪽", "단위", "판단"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
