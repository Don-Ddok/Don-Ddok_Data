# -*- coding: utf-8 -*-
"""탐색적 분석 (exploratory sweep) — 기존 사전등록 분석과 별개, 기존 판정표·등급을 바꾸지 않음
사양: harmonized C1 그대로 (run_common, 법인 FE + 지역×연월 FE + 노출×영업일수차, 보정YoY, 윈저 1/99, 이중 군집, t(G_min−1))
결과변수·표본만 바꾼다. 판정 시차 h=6, 곡선 h=−6~12는 참고. h=6 검정 전체에 BH-FDR.
모든 회귀(곡선·사전추세 포함)를 tests_log.csv에 기록한다. FDR 묶음은 h=6 검정(`FDR묶음` = True)만.
실행: py -3.11 -u explore.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, joint_test, run_common

DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\exploratory"
HS = [h for h in range(-6, 13) if h != -1]
DEP = ["요구불예금잔액", "거치식예금잔액", "적립식예금잔액", "수익증권잔액", "신탁잔액", "퇴직연금잔액"]
OP6 = ["운전_할인어음잔액", "운전_일반자금대출잔액", "운전_무역금융잔액", "운전_기업구매자금대출잔액",
       "운전_외상매출채권담보대출잔액", "운전_기타운전자금대출잔액"]
CC = ["신용카드사용금액", "체크카드사용금액", "자동이체금액", "창구거래금액", "인터넷뱅킹거래금액", "스마트뱅킹거래금액"]
NOTE = {"여신한도금액": "운전자금 전체 한도가 아니라 한도대출분만일 수 있음 (운전자금잔액÷한도 > 1인 관측 46%)",
        "운전_무역금융잔액": "비노출 법인 47곳뿐", "수익증권잔액": "표본 200곳(노출 47)", "총수신": "요구불을 포함함"}
LOG = []


def log(s):
    print(s, flush=True)


def record(r, 묶음, 결과변수, 표본, 구분, 단위="%", fdr_h=6):
    for _, x in r.iterrows():
        LOG.append({"묶음": 묶음, "결과변수": 결과변수, "표본": 표본, "구분": 구분, "h": x["h"], "β3": x["β3"], "SE": x["SE"],
                    "p": x["p"], "CI_lo": x["CI_lo"], "CI_hi": x["CI_hi"],
                    "환산": (-0.1 * x["β3"] * 100) if 단위 == "%p" else (np.exp(-0.1 * x["β3"]) - 1) * 100, "단위": 단위,
                    "N": x["N"], "노출법인": x["노출법인"], "비노출법인": x["비노출법인"], "월": x["월"],
                    "FDR묶음": bool(구분 == "곡선" and x["h"] == fdr_h), "주석": NOTE.get(결과변수, "")})


def pre_stats(P, r, parts, dep_fn_pre):
    pj, _ = joint_test(r, parts)
    ro, _ = run_common(P, [0], dep=dep_fn_pre, label="원본방식사전", log=lambda s: None)
    return pj, ro


def curve(P, 묶음, 결과변수, 표본, unit="%", ratio=None, keep=None):
    """h=−6~12 곡선 + 사전추세(LP 결합·원본 방식). ratio: 수준 변수 배열(%p)"""
    if ratio is None:
        r, parts = run_common(P, HS, keep=keep, label=결과변수, log=lambda s: None)
        dep_pre = P.at(P.ln, -1) - P.at(P.ln, -7)
    else:
        lag = P.at(ratio, -1)
        r, parts = run_common(P, HS, keep=keep, dep_fn=lambda h: P.at(ratio, h) - lag, label=결과변수, log=lambda s: None)
        dep_pre = P.at(ratio, -1) - P.at(ratio, -7)
    record(r, 묶음, 결과변수, 표본, "곡선", unit)
    pj = run_common(P, [0], dep=dep_pre, keep=keep, label="원본방식사전", log=lambda s: None)[0]
    pre, _ = joint_test(r, parts)
    record(pj.assign(h=np.nan), 묶음, 결과변수, 표본, "사전추세_원본방식", unit)
    x6 = r[r["h"] == 6].iloc[0]
    log(f"  [{묶음}] {결과변수} ({표본}) h=6 β3 {x6['β3']:+.4f} p {x6['p']:.3f} | LP p {pre['p_chi2']:.3f} 검산 {pre['검산_최대상대차']:.0e}")
    return r, parts, {"LP결합p": pre["p_chi2"] if pre["Wald"] >= 0 else np.nan, "LP_Wald": pre["Wald"], "검산": pre["검산_최대상대차"], "원본방식_β3pre": pj["β3"].iloc[0], "원본방식_p": pj["p"].iloc[0]}


def diff_test(pa, pb, h, ba, bb, Gm):
    """두 집단 회귀(같은 Panel, 서로 다른 법인)의 h 시차 β3 차이: 영향 성분을 법인·월·법인월 군집으로 합친 공분산"""
    (psa, da), (psb, db) = pa[h], pb[h]
    V = np.zeros((2, 2))
    for dim in ["법인", "월", "법인월"]:
        ga, ca, sg = da[dim]; gb, cb, _ = db[dim]
        L = int(max(ga.max(), gb.max())) + 1
        sa, sb = np.bincount(ga, psa, minlength=L), np.bincount(gb, psb, minlength=L)
        V += sg * np.array([[ca * sa @ sa, np.sqrt(ca * cb) * sa @ sb], [np.sqrt(ca * cb) * sa @ sb, cb * sb @ sb]])
    se = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
    dof = Gm - 1
    return ba - bb, se, 2 * stats.t.sf(abs((ba - bb) / se), dof), V


def main():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "외환_수출실적금액", "외환_수입실적금액",
                                   "여신한도금액", "여신_운전자금대출잔액", "여신_시설자금대출잔액", "법인_고객등급", "전담고객여부",
                                   "업종_대분류"] + DEP + OP6 + CC)
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    d["총수신"] = d[DEP].sum(axis=1)
    ever = lambda c: d[d.groupby("법인ID")[c].transform("max") > 0].copy()

    # 검증: harmonized 요구불 C1 h=6
    r, _ = run_common(Panel(ever("요구불예금잔액"), "요구불예금잔액"), [6], label="검증", log=lambda s: None)
    ok = round(r["β3"].iloc[0], 4) == 0.3635 and abs(r["SE"].iloc[0] / 0.113493 - 1) <= 0.01
    log(f"[검증] β3 {r['β3'].iloc[0]:.6f} (기준 0.363537), SE {r['SE'].iloc[0]:.6f} (기준 0.113493) → {'통과' if ok else '실패 — 중단'}")
    if not ok:
        return
    info = []

    # A·B·C
    for grp, vars_ in [("A", ["총수신", "요구불비중", "수익증권잔액", "신탁잔액", "퇴직연금잔액"]),
                       ("B", ["여신한도금액", "여신_시설자금대출잔액"] + OP6), ("C", CC)]:
        for v in vars_:
            if v == "요구불비중":
                s = ever("총수신"); P = Panel(s, "총수신")
                al = P.d[["법인ID", "ym"]].merge(s[["법인ID", "ym", "요구불예금잔액", "총수신"]], on=["법인ID", "ym"], how="left")
                tot = al["총수신"].to_numpy(float)
                ratio = np.where(tot > 0, al["요구불예금잔액"].to_numpy(float) / np.where(tot > 0, tot, 1), np.nan)
                smp = "총수신>0 법인"
                rr, parts, pi = curve(P, grp, v, smp, unit="%p", ratio=ratio)
            else:
                s = ever(v); P = Panel(s, v)
                smp = f"{v}>0 법인"
                rr, parts, pi = curve(P, grp, v, smp)
            info.append({"묶음": grp, "결과변수": v, "표본": smp, **pi})

    # D: 요구불 잔액 이질성 (2023-01~03 값)
    s = ever("요구불예금잔액"); P = Panel(s, "요구불예금잔액")
    early = s[s["ym"] <= 202303].sort_values("ym")
    first = early.groupby("법인ID").first()
    size = early.groupby("법인ID")["총수신"].mean()
    med = size.median()
    ind = first["업종_대분류"].where(first["업종_대분류"].isin(["제조업", "도매 및 소매업"]), "기타")
    dims = {"고객등급": (first["법인_고객등급"], "최우수"), "전담고객": (first["전담고객여부"], "Y"),
            "업종": (ind, "제조업"), "규모": (pd.Series(np.where(size >= med, "상위", "하위"), index=size.index), "상위"),
            "시도": (first["사업장_시도"], "대구")}
    firm_ids = P.d["법인ID"].to_numpy()
    log(f"[D] 2023-01~03 관측 법인 {len(first):,} / 규모 기준 중앙값 {med:.3g}")
    for dim, (lab, ref) in dims.items():
        lab_row = pd.Series(firm_ids).map(lab).to_numpy()
        res = {}
        for g in sorted(pd.Series(lab).dropna().unique()):
            keep = lab_row == g
            rr, parts, pi = curve(P, "D", "요구불예금잔액", f"{dim}={g}", keep=keep)
            res[g] = (rr, parts)
            info.append({"묶음": "D", "결과변수": "요구불예금잔액", "표본": f"{dim}={g}", **pi})
        for g in res:
            if g == ref:
                continue
            ra, pa = res[g]; rb, pb = res[ref]
            xa, xb = ra[ra.h == 6].iloc[0], rb[rb.h == 6].iloc[0]
            dif, se, p, _ = diff_test(pa, pb, 6, xa["β3"], xb["β3"], int(min(xa["월"], xb["월"])))
            tq = stats.t.ppf(0.975, int(min(xa["월"], xb["월"])) - 1)
            LOG.append({"묶음": "D", "결과변수": "요구불예금잔액", "표본": f"{dim}: {g} − {ref}", "구분": "곡선", "h": 6, "β3": dif, "SE": se,
                        "p": p, "CI_lo": dif - tq * se, "CI_hi": dif + tq * se, "환산": np.nan, "단위": "β3 차이",
                        "N": xa["N"] + xb["N"], "노출법인": xa["노출법인"] + xb["노출법인"], "비노출법인": xa["비노출법인"] + xb["비노출법인"],
                        "월": min(xa["월"], xb["월"]), "FDR묶음": True, "주석": "집단 간 차이(완전 교차 모형과 같음)"})
            info.append({"묶음": "D", "결과변수": "요구불예금잔액", "표본": f"{dim}: {g} − {ref}", "LP결합p": np.nan, "LP_Wald": np.nan, "검산": np.nan,
                         "원본방식_β3pre": np.nan, "원본방식_p": np.nan})
            log(f"  [D] {dim}: {g} − {ref} 차이 {dif:+.4f} (SE {se:.4f}) p {p:.3f}")

    # E: 노출 쪼개기
    fx = d.groupby("법인ID").agg(x=("외환_수출실적금액", "max"), m=("외환_수입실적금액", "max"), e=("외환노출", "first"))
    exp_ids, imp_ids, non_ids = set(fx.index[fx.x > 0]), set(fx.index[(fx.x <= 0) & (fx.m > 0)]), set(fx.index[fx.e == 0])
    log(f"[E] 수출 실적 있음 {len(exp_ids)}, 수입만 {len(imp_ids)}, 비노출 {len(non_ids)}")
    for v in ["요구불예금잔액", "여신_운전자금대출잔액", "거치식예금잔액", "적립식예금잔액", "총수신"]:
        base = ever(v)
        for nm, ids in [("수출노출(525) vs 비노출", exp_ids), ("수입만(509) vs 비노출", imp_ids)]:
            s = base[base["법인ID"].isin(ids | non_ids)].copy()
            s["노출_E"] = s["법인ID"].isin(ids).astype(int)
            P = Panel(s, v, expo_col="노출_E")
            rr, parts, pi = curve(P, "E", v, nm)
            info.append({"묶음": "E", "결과변수": v, "표본": nm, **pi})

    # 정리: BH-FDR (h=6 검정 전체)
    t = pd.DataFrame(LOG)
    fam = t["FDR묶음"]
    p = t.loc[fam, "p"].to_numpy(); m = len(p); o = np.argsort(p)
    q = np.empty(m); run = 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]; run = min(run, p[i] * m / (rank + 1)); q[i] = run
    t["q_BH"] = np.nan; t.loc[fam, "q_BH"] = q
    inf = pd.DataFrame(info)
    t = t.merge(inf, on=["묶음", "결과변수", "표본"], how="left")
    same = np.sign(t["원본방식_β3pre"]) == np.sign(t["β3"])
    t["사전추세_연장"] = fam & same & (t["원본방식_β3pre"].abs() >= t["β3"].abs() / 2)
    t["사전추세_반대부호유의"] = fam & (~same) & (t["원본방식_p"] < 0.05) & t["원본방식_β3pre"].notna()
    t.to_csv(os.path.join(OUT, "tests_log.csv"), index=False, encoding="utf-8-sig")
    log(f"\n기록된 회귀 행 {len(t):,} / FDR 묶음(h=6) {m}개 / q<0.10 {int((t['q_BH'] < 0.10).sum())}개 / p<0.05 {int((t.loc[fam, 'p'] < 0.05).sum())}개")


if __name__ == "__main__":
    main()
