# -*- coding: utf-8 -*-
"""기업별 외환 실적 충격 (SPEC.md, 커밋 12b17be)
  ln(y_{t+h}+1) − ln(y_{t−1}+1) = 법인 FE + 지역×연월 FE + β·S_{i,t} + ε   (HOLD: 1[y>0] 차)
충격: 주 (a') ln(100·S_t+1) − ln(100·S_{t−12}+1), 강건성 (b) S_t ≤ 0.5×(t−14~t−3 월평균×3)
표본: 외환노출 법인, 충격 정의되는 달(2024-03~), 계정 잔액이 한 번이라도 > 0
h=−6~12 (h=−1 제외), 이중 군집, t(G_min−1), 영업일수 통제 없음(지역×연월 FE에 흡수)
사전추세: LP 결합 h=−6..−3, 원본 방식 ln_{t−3} − ln_{t−9} (참고: harmonized 형태 ln_{t−1} − ln_{t−7})
SPEC 순서대로 추정치는 화면에 내지 않고 파일(run_log.txt, results.csv, pretrend.csv)에만 쓴다 → 다음은 mde.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, joint_test, run_common

DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\firm_shock"
ACC = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}
MODELS = [("요구불", False), ("운전자금", False), ("거치식", False), ("적립식", False), ("거치식", True), ("적립식", True)]
HS = [h for h in range(-6, 13) if h != -1]
LOGF = open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8")


def log(s):
    LOGF.write(s + "\n"); LOGF.flush()


def say(s):
    print(s, flush=True); log(s)


def shocks(P):
    fx = P.d["_keep_FX"].to_numpy(float)
    S_now = fx + P.at(fx, -1) + P.at(fx, -2)
    S_ly = P.at(fx, -12) + P.at(fx, -13) + P.at(fx, -14)
    base12 = sum(P.at(fx, -k) for k in range(3, 15)) / 12 * 3
    a100 = np.log(100 * S_now + 1) - np.log(100 * S_ly + 1)
    b = np.where(np.isnan(S_now) | np.isnan(base12), np.nan,
                 np.where(base12 > 0, (S_now <= 0.5 * base12).astype(float), np.nan))
    return {"a100": a100, "b": b}


def main():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "외환_수출실적금액", "외환_수입실적금액"] + list(ACC.values()))
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512) & (d["exposed"] == 1)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    d["_keep_FX"] = d["외환_수출실적금액"] + d["외환_수입실적금액"]
    say(f"외환노출 법인 {d['법인ID'].nunique():,}곳, 행 {len(d):,}")
    res, pre = [], []
    for acct, hold in MODELS:
        col = ACC[acct]; name = acct + ("_HOLD" if hold else "")
        s = d[d.groupby("법인ID")[col].transform("max") > 0].copy()
        P = Panel(s, col)
        SH = shocks(P)
        for k, v in SH.items():
            P.X[k] = v
        base = P.pos if hold else P.ln
        for sk, slabel in [("a100", "주_a'"), ("b", "강건_b")]:
            keep = ~np.isnan(P.X[sk])
            nf = len(np.unique(P.firm[keep]))
            say(f"[{name} {slabel}] 충격 정의되는 법인 {nf:,}곳, 법인×월 {int(keep.sum()):,}")
            if nf < 20:
                say(f"  → 20곳 미만, 추정 안 함 (SPEC §4)"); continue
            kw = dict(xkey=sk, keep=keep, ctrl="none", need_two=False, hold=hold, winsor=not hold, log=log)
            r, parts = run_common(P, HS, label=f"{name}_{slabel}", **kw)
            r.insert(0, "충격", slabel); r.insert(0, "계정", name)
            res.append(r)
            jt, chk = joint_test(r, parts, hs=(-6, -5, -4, -3))
            log(f"[{name} {slabel}] 결합검정 검산\n{chk.round(6).to_string(index=False)}")
            row = {"계정": name, "충격": slabel, "LP_Wald": jt["Wald"], "LP_p_chi2": jt["p_chi2"] if jt["Wald"] >= 0 else np.nan,
                   "LP_p_F": jt["p_F"] if jt["Wald"] >= 0 else np.nan, "LP_계산불가": jt["Wald"] < 0, "검산_최대상대차": jt["검산_최대상대차"]}
            for tag, (a, bb) in {"원본": (-3, -9), "원본_harmonized형태": (-1, -7)}.items():
                dep = P.at(base, a) - P.at(base, bb)
                ro, _ = run_common(P, [0], dep=dep, label=f"{name}_{slabel}_{tag}", **kw)
                row.update({f"{tag}_β": ro["β3"].iloc[0], f"{tag}_SE": ro["SE"].iloc[0], f"{tag}_p": ro["p"].iloc[0], f"{tag}_N": ro["N"].iloc[0]})
            pre.append(row)
        if name == "요구불":   # pyfixest 대조 (주 충격, h=6): 계수 같음 확인
            import pyfixest as pf
            y = P.at(P.ln, 6) - P.at(P.ln, -1)
            m = ~np.isnan(y) & ~np.isnan(P.X["a100"])
            yy = y[m]; lo, hi = np.quantile(yy, [0.01, 0.99])
            x = pd.DataFrame({"y": np.clip(yy, lo, hi), "S": P.X["a100"][m], "firm": P.firm[m], "regym": P.regym[m], "month": P.month[m]})
            fm = pf.feols("y ~ S | firm + regym", data=x, vcov={"CRV1": "firm + month"}, fixef_maxiter=100000)
            mine = next(r for r in res if r["계정"].iloc[0] == "요구불" and r["충격"].iloc[0] == "주_a'").set_index("h").loc[6]
            chk = pd.DataFrame([{"hlib_β": mine["β3"], "pyfixest_β": float(fm.coef()["S"]), "hlib_SE": mine["SE"], "pyfixest_SE": float(fm.se()["S"])}])
            chk["β_상대차"] = chk["hlib_β"] / chk["pyfixest_β"] - 1
            chk.to_csv(os.path.join(OUT, "pyfixest_check.csv"), index=False, encoding="utf-8-sig")
            say(f"[pyfixest 대조] 요구불 주 충격 h=6 β 상대차 {chk['β_상대차'].iloc[0]:.1e}, SE 비 {chk['hlib_SE'].iloc[0] / chk['pyfixest_SE'].iloc[0]:.3f}")
    out = pd.concat(res, ignore_index=True).drop(columns=["환산", "단위", "γ", "비노출법인", "pf_β3", "pf_SE"])
    out = out.rename(columns={"β3": "β", "노출법인": "법인", "p_holm": "p_holm_계정내h1_12"})
    out.to_csv(os.path.join(OUT, "results.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pre).to_csv(os.path.join(OUT, "pretrend.csv"), index=False, encoding="utf-8-sig")
    say(f"저장: results.csv ({len(out)}행), pretrend.csv ({len(pre)}행). 다음: mde.py (추정치보다 MDE 먼저)")


if __name__ == "__main__":
    main()
