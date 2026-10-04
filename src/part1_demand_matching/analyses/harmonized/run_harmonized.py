# -*- coding: utf-8 -*-
"""사후 재추정 실행 (SPEC.md, 커밋 58502f3) — 계정·모형을 골라 run_common 하나로 추정

사용: py -3.11 -u run_harmonized.py <계정...> --models C1,C2,E23,EX,POS,HOLD,RAW [--industry]
  C1을 고르면 사전추세(LP 결합 + 검산, 원본 방식)도 함께 계산한다.
출력: parts/res_{계정}_{모형}.csv, parts/pre_{계정}.csv, parts/prechk_{계정}.csv, balance_{계정}.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, holm, joint_test, run_common

P3 = r"C:\test\external\part3_work\통계 프로젝트\여신업종분석\단계별 분석"
sys.path.insert(0, P3)
from common import build_firm_table, match_weights, psm_match   # 수정 없이 import

DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\harmonized"
PARTS = os.path.join(OUT, "parts")
ACC = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}
IND = {"1차 금속": "1차 금속 제조업", "기타 기계": "기타 기계 및 장비 제조업", "도매·상품중개": "도매 및 상품 중개업"}
HS = [h for h in range(-6, 13) if h != -1]
PF_C1 = {-6, -2, 0, 6, 12}
LOG = open(os.path.join(OUT, "run_log.txt"), "a", encoding="utf-8")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def load_df():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "업종_중분류", "exposed",
                                   "외환_수출실적금액", "외환_수입실적금액"] + list(ACC.values()))
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    ex = d.groupby("법인ID")["외환_수출실적금액"].max() > 0
    d["노출_수출"] = d["법인ID"].map(ex).astype(int)
    d["수입만"] = ((d["외환노출"] == 1) & (d["노출_수출"] == 0)).astype(int)
    e23 = d[d["ym"] < 202401].assign(a=lambda x: (x["외환_수출실적금액"] > 0) | (x["외환_수입실적금액"] > 0)).groupby("법인ID")["a"].max()
    d["노출_2023"] = d["법인ID"].map(e23).fillna(False).astype(int)
    last_ind = d.dropna(subset=["업종_중분류"]).sort_values("ym").groupby("법인ID")["업종_중분류"].last()
    d["업종"] = d["법인ID"].map(last_ind)
    log(f"df_ready 대구·경북: 법인 {d['법인ID'].nunique():,}, 노출 {d.loc[d['외환노출'] == 1, '법인ID'].nunique():,}, "
        f"수출 노출 {d.loc[d['노출_수출'] == 1, '법인ID'].nunique():,}, 수입만 {d.loc[d['수입만'] == 1, '법인ID'].nunique():,}, "
        f"2023 노출 {d.loc[d['노출_2023'] == 1, '법인ID'].nunique():,}")
    return d


def c2_weights(sample_ids, acct):
    p = pd.read_parquet(os.path.join(P3, "step1_loan_industry_panel.parquet"))
    p["사업장_시도"] = p["사업장_시도"].astype(str).str.strip()
    p = p[p["사업장_시도"].isin(["대구", "경북"])]
    in_p = set(p["법인ID"])
    miss = len(set(sample_ids) - in_p)
    p = p[p["법인ID"].isin(sample_ids)]
    firm, ind_cols = build_firm_table(p)
    pairs, _ = psm_match(firm, ind_cols, k=3)
    W = match_weights(pairs)
    T = firm["외환노출"].to_numpy() == 1
    outside = int(T.sum()) - pairs["법인ID_노출"].nunique()
    log(f"[{acct} C2] 계정 표본 {len(sample_ids):,} 중 매칭 변수 없음(df_ready에만 있음) {miss:,} / 매칭 모집단 {len(firm):,} "
        f"(노출 {int(T.sum()):,}) / 공통지지 밖 노출 {outside} / 매칭 노출 {pairs['법인ID_노출'].nunique():,}, "
        f"대조 서로 다른 {pairs['법인ID_대조'].nunique():,}곳 / 초기 운전자금 0인 법인 {(firm['log_초기운전'] == 0).sum():,}")
    wv = firm.index.map(W).fillna(0).to_numpy(float)
    mt = firm.index.isin(pairs["법인ID_노출"])
    rows = []
    for c in ["log_초기수신", "log_초기운전", "최우수", "우수", "거래기간"] + ind_cols:
        x = firm[c].to_numpy(float)
        sd = np.sqrt((x[T].var(ddof=1) + x[~T].var(ddof=1)) / 2)
        sd = sd if sd > 0 else np.nan
        rows.append({"변수": c, "SMD_매칭전": (x[T].mean() - x[~T].mean()) / sd,
                     "SMD_매칭후": (x[mt].mean() - np.average(x, weights=np.where(~T, wv, 0))) / sd})
    bal = pd.DataFrame(rows).round(4)
    bal.to_csv(os.path.join(OUT, f"balance_{acct}.csv"), index=False, encoding="utf-8-sig")
    log(f"[{acct} C2] 균형: 매칭후 |SMD| 최대 {bal['SMD_매칭후'].abs().max():.3f}, 0.1 초과 {(bal['SMD_매칭후'].abs() > 0.1).sum()}개")
    return W


def save(r, acct, model, ind=""):
    r = r.copy(); r.insert(0, "업종", ind); r.insert(0, "계정", acct); r["모형"] = model
    r.to_csv(os.path.join(PARTS, f"res_{acct}_{model}{'_' + ind if ind else ''}.csv"), index=False, encoding="utf-8-sig")
    return r


def run_account(d, acct, models, industry):
    col = ACC[acct]
    ever = d.groupby("법인ID")[col].transform("max") > 0
    s = d[ever].copy()
    log(f"\n===== {acct}: 법인 {s['법인ID'].nunique():,} (노출 {s.loc[s['외환노출'] == 1, '법인ID'].nunique():,}), "
        f"행 {len(s):,}, 0 이하 잔액 행 {(s[col] <= 0).sum():,}")
    P = Panel(s, col)
    for model in models:
        log(f"--- {acct} {model}")
        kw = dict(label=model, log=log)
        if model == "C1":
            r, parts = run_common(P, HS, pf_hs=PF_C1, **kw)
            save(r, acct, model)
            pre, chk = joint_test(r, parts)
            log(f"[{acct} 결합검정 검산]\n{chk.round(6).to_string(index=False)}")
            dep = P.at(P.ln, -1) - P.at(P.ln, -7)
            ro, _ = run_common(P, [0], dep=dep, label="원본방식사전", log=log)
            pre.update({"계정": acct, "원본방식_β3pre": ro["β3"].iloc[0], "원본방식_SE": ro["SE"].iloc[0],
                        "원본방식_p": ro["p"].iloc[0], "원본방식_N": ro["N"].iloc[0]})
            pd.DataFrame([pre]).to_csv(os.path.join(PARTS, f"pre_{acct}.csv"), index=False, encoding="utf-8-sig")
            chk.to_csv(os.path.join(PARTS, f"prechk_{acct}.csv"), index=False, encoding="utf-8-sig")
            log(f"[{acct} 사전추세] LP 결합 p(χ²) {pre['p_chi2']:.4f}, F {pre['p_F']:.4f}, 검산 최대 상대차 {pre['검산_최대상대차']:.2e} / "
                f"원본 방식 β3pre {pre['원본방식_β3pre']:+.4f} (p {pre['원본방식_p']:.4f})")
        elif model == "C2":
            W = c2_weights(sorted(s["법인ID"].unique()), acct)
            w = P.d["법인ID"].map(W).fillna(0).to_numpy()
            save(run_common(P, HS, weights=w, **kw)[0], acct, model)
        elif model == "E23":
            P2 = Panel(s, col, expo_col="노출_2023")
            save(run_common(P2, HS, start=202401, **kw)[0], acct, model)
        elif model == "EX":
            P2 = Panel(s[s["수입만"] == 0], col, expo_col="노출_수출")
            save(run_common(P2, HS, **kw)[0], acct, model)
        elif model == "POS":
            save(run_common(P, HS, positive=True, **kw)[0], acct, model)
        elif model == "HOLD":
            save(run_common(P, HS, hold=True, winsor=False, **kw)[0], acct, model)
        elif model == "RAW":
            save(run_common(P, HS, xkey="원", **kw)[0], acct, model)
    if industry:
        firms_ind = P.d["법인ID"].map(s.groupby("법인ID")["업종"].first()).to_numpy()
        for model in ["C1", "HOLD"]:
            got = []
            for short, full in IND.items():
                keep = firms_ind == full
                nf = len(np.unique(P.firm[keep]))
                if nf < 20:
                    log(f"  [{acct} {model} {short}] 법인 {nf}곳 < 20 → 추정 안 함"); continue
                r, _ = run_common(P, HS, keep=keep, hold=(model == "HOLD"), winsor=(model != "HOLD"),
                                  label=f"{model}·{short}", log=log)
                r.insert(0, "업종", short); got.append(r)
            if got:
                g = pd.concat(got, ignore_index=True)
                fam = g["h"].between(1, 12)
                g["p_holm_36"] = np.nan
                g.loc[fam, "p_holm_36"] = holm(g.loc[fam, "p"])
                g.insert(0, "계정", acct); g["모형"] = f"업종_{model}"
                g.to_csv(os.path.join(PARTS, f"res_{acct}_업종_{model}.csv"), index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accounts", nargs="+")
    ap.add_argument("--models", default="C1,C2,E23,EX,POS,HOLD,RAW")
    ap.add_argument("--industry", action="store_true")
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    d = load_df()
    for acct in a.accounts:
        run_account(d, acct, a.models.split(","), a.industry)
    log("완료: " + " ".join(a.accounts) + " / " + a.models + (" + 업종" if a.industry else ""))


if __name__ == "__main__":
    main()
