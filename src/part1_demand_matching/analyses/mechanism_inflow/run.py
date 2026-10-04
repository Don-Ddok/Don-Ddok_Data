# -*- coding: utf-8 -*-
"""사후 분석(메커니즘): 요구불 입금·출금 + 충격 변수 타당성 (SPEC.md, 커밋 4e2298c)
검증 → 입금(판정)·출금(보고) C1 사양 LP + 사전추세 + MDE → §3 수출 실적(3개월 합계 주, 월 단위 참고)
실행: py -3.11 -u run.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, holm, joint_test, run_common

DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\mechanism_inflow"
HS = [h for h in range(-6, 13) if h != -1]
PF = {-6, -2, 0, 6, 12}
REF = (0.363537, 0.113493)   # harmonized 요구불 C1 h=6 (커밋 5299669)


def main():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "exposed", "요구불예금잔액", "요구불입금금액",
                                   "요구불출금금액", "외환_수출실적금액"])
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    s = d[d.groupby("법인ID")["요구불예금잔액"].transform("max") > 0].copy()
    print(f"요구불 표본: 법인 {s['법인ID'].nunique():,} (노출 {s.loc[s['외환노출'] == 1, '법인ID'].nunique():,}), 행 {len(s):,}")

    # 검증: 요구불 C1 h=6 재현
    r, _ = run_common(Panel(s, "요구불예금잔액"), [6], label="검증")
    b, se = r["β3"].iloc[0], r["SE"].iloc[0]
    ok = round(b, 4) == round(REF[0], 4) and abs(se / REF[1] - 1) <= 0.01
    print(f"[검증] β3 {b:.6f} (기준 {REF[0]}), SE {se:.6f} (기준 {REF[1]}, 상대차 {se / REF[1] - 1:+.2e}) → {'통과' if ok else '실패 — 중단'}")
    if not ok:
        return

    res, pre_rows, mde_rows = [], [], []
    for acct, col in [("입금", "요구불입금금액"), ("출금", "요구불출금금액")]:
        P = Panel(s, col)
        print(f"\n--- {acct}: 0 이하 행 {(P.bal <= 0).sum():,}")
        rr, parts = run_common(P, HS, pf_hs=PF if acct == "입금" else (), label=acct)
        rr.insert(0, "분석", acct); res.append(rr)
        pj, chk = joint_test(rr, parts)
        print(f"[{acct} 결합검정 검산]\n{chk.round(6).to_string(index=False)}")
        dep = P.at(P.ln, -1) - P.at(P.ln, -7)
        ro, _ = run_common(P, [0], dep=dep, label=f"{acct}_원본방식사전")
        pre_rows.append({"분석": acct, **pj, "원본방식_β3pre": ro["β3"].iloc[0], "원본방식_SE": ro["SE"].iloc[0],
                         "원본방식_p": ro["p"].iloc[0], "원본방식_N": int(ro["N"].iloc[0])})
        x = rr[rr["h"] == 6].iloc[0]
        G = int(min(x["노출법인"] + x["비노출법인"], x["월"])); dof = G - 1
        mde = (stats.t.ppf(0.975, dof) + stats.t.ppf(0.80, dof)) * x["SE"]
        mde_rows.append({"분석": acct, "SE": x["SE"], "dof": dof, "MDE(β)": mde,
                         "MDE_감소방향%": (np.exp(-0.1 * mde) - 1) * 100, "MDE_증가방향%": (np.exp(0.1 * mde) - 1) * 100})

    # §3 충격 변수 타당성: EX 노출 법인만
    ex = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
    e = d[d["법인ID"].isin(ex[ex].index)].copy()
    e["노출_수출"] = 1
    Pe = Panel(e, "외환_수출실적금액", expo_col="노출_수출")
    x = np.clip(Pe.bal, 0, None)
    S3 = x + Pe.at(x, -1) + Pe.at(x, -2)                    # t−2 ~ t 합계 (하나라도 결측이면 nan)
    lnS = np.log(S3 + 1)
    lag3 = Pe.at(lnS, -1)                                     # t−3 ~ t−1 합계
    print(f"\n--- §3 EX 노출 법인 {e['법인ID'].nunique():,}곳, 행 {len(e):,}")
    kw = dict(fe2="month", ctrl="biz", need_two=False)
    r3, _ = run_common(Pe, list(range(0, 13)), dep_fn=lambda h: Pe.at(lnS, h) - lag3, label="수출_3개월합", **kw)
    r1, _ = run_common(Pe, list(range(0, 13)), label="수출_월(참고)", **kw)
    for rr in (r3, r1):
        f03 = rr["h"].between(0, 3)
        rr["p_holm_h0_3"] = np.nan
        rr.loc[f03, "p_holm_h0_3"] = holm(rr.loc[f03, "p"])
    r3.insert(0, "분석", "수출점검_3개월합"); r1.insert(0, "분석", "수출점검_월(참고)")
    res += [r3, r1]

    out = pd.concat(res, ignore_index=True)
    out.to_csv(os.path.join(OUT, "results.csv"), index=False, encoding="utf-8-sig")
    pre = pd.DataFrame(pre_rows); pre.to_csv(os.path.join(OUT, "pretrend.csv"), index=False, encoding="utf-8-sig")
    mdf = pd.DataFrame(mde_rows); mdf.to_csv(os.path.join(OUT, "mde.csv"), index=False, encoding="utf-8-sig")

    # 판정 (SPEC §2)
    x = out[(out["분석"] == "입금") & (out["h"] == 6)].iloc[0]
    pr = pre[pre["분석"] == "입금"].iloc[0]
    ext = np.sign(pr["원본방식_β3pre"]) == np.sign(x["β3"]) and abs(pr["원본방식_β3pre"]) >= abs(x["β3"]) / 2
    rev = np.sign(pr["원본방식_β3pre"]) != np.sign(x["β3"]) and pr["원본방식_p"] < 0.05
    if x["β3"] > 0 and x["p"] < 0.05 and pr["p_chi2"] >= 0.10 and not ext and not rev:
        grade = "지지"
    elif x["β3"] > 0 and x["p"] < 0.10:
        grade = "약한 증거"
    elif x["β3"] < 0 and x["p"] < 0.10:
        grade = "반대 방향"
    else:
        grade = "근거 없음"
    t3 = r3[r3["h"].between(0, 3)]
    link = bool(((t3["β3"] > 0) & (t3["p_holm_h0_3"] < 0.05)).any())
    j = {"등급": grade, "입금_h6_β3": x["β3"], "입금_h6_환산": x["환산"], "입금_h6_p": x["p"], "LP결합p": pr["p_chi2"],
         "원본방식_β3pre": pr["원본방식_β3pre"], "원본방식_p": pr["원본방식_p"], "연장표시": bool(ext), "반대부호유의": bool(rev),
         "§3_연결": link}
    pd.Series(j).to_csv(os.path.join(OUT, "judgment.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n[결과]\n", out[["분석", "h", "β3", "SE", "p", "p_holm", "CI_lo", "CI_hi", "환산", "N", "노출법인", "비노출법인", "월", "pf_β3", "pf_SE"]
                          + (["p_holm_h0_3"] if "p_holm_h0_3" in out else [])].round(4).to_string(index=False))
    print("\n[사전추세]\n", pre.round(4).to_string(index=False))
    print("\n[MDE]\n", mdf.round(4).to_string(index=False))
    print("\n[판정]", j)


if __name__ == "__main__":
    main()
