# -*- coding: utf-8 -*-
"""사후 보조 분석: 요구불예금 × 수출 충격 — 통제를 강화한 이중차분형 국소투영 (사양: SPEC.md, 커밋 c8e48c3)

실행 코드 교체판 (CHANGES.md): 사양은 그대로, 추정은 hlib.run_common(harmonized와 같은 함수, 군집 키 재사용·수렴 1e-10).
  A 주(1/99) · A-원YoY · A-NW(윈저 없음) · A-양수(t−1·t+h 잔액 > 0) · B 매칭 1:3(가중), h=−6~12 (−1 제외)
  이중 군집(법인·연월, CGM), p = t(G_min−1), Holm h=1~12
  사전추세: LP h=−6~−2 결합 Wald(검산 포함) + 원본 방식(ln_{t−1} − ln_{t−7}), A·B
기존 실행 코드는 run_old_code.py로 보존. 원본(기존 주 결과, 파트3 코드, 데이터 폴더)은 수정하지 않는다.
실행: py -3.11 -u run.py [--check-only]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, joint_test, run_common

P3 = r"C:\test\external\part3_work\통계 프로젝트\여신업종분석\단계별 분석"
sys.path.insert(0, P3)
from common import build_firm_table, match_weights, psm_match   # 수정 없이 import

DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\did_demand_deposit"
Y = "요구불예금잔액"
HS = [h for h in range(-6, 13) if h != -1]
PF_A = {-6, -2, 0, 6, 12}


def load():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", Y, "exposed"])
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    P = Panel(d, Y)
    print(f"패널: 대구·경북 법인 {d['법인ID'].nunique():,}, 행 {len(d):,}, 노출 {d.loc[d['외환노출'] == 1, '법인ID'].nunique():,}")
    print(f"  0 이하 잔액 행 {(d[Y] <= 0).sum():,} (빼지 않음, ln(+1)) / 두 지역에 걸친 법인 {(d.groupby('법인ID')['사업장_시도'].nunique() > 1).sum()}")
    return d, P


def matching(d):
    """파트3 함수 그대로: 대구·경북 전체 법인(운전자금 한정 없음)으로 1:3 매칭 — 기존 run_old_code.py와 같은 절차"""
    p = pd.read_parquet(os.path.join(P3, "step1_loan_industry_panel.parquet"))
    p["사업장_시도"] = p["사업장_시도"].astype(str).str.strip()
    p = p[p["사업장_시도"].isin(["대구", "경북"])]
    firm, ind_cols = build_firm_table(p)
    pairs, _ = psm_match(firm, ind_cols, k=3)
    W = match_weights(pairs)
    print(f"매칭: 파트3 패널 대구·경북 법인 {len(firm):,} (노출 {int(firm['외환노출'].sum()):,}) / "
          f"df_ready에만 있는 법인 {len(set(d['법인ID']) - set(firm.index)):,} (매칭 대상 아님) / "
          f"공통지지 밖 노출 {int(firm['외환노출'].sum()) - pairs['법인ID_노출'].nunique()} / "
          f"매칭 노출 {pairs['법인ID_노출'].nunique():,}, 대조 서로 다른 {pairs['법인ID_대조'].nunique():,}곳")
    print(f"  운전자금 초기값 0인 법인 {(firm['log_초기운전'] == 0).sum():,} / {len(firm):,}")
    T = firm["외환노출"].to_numpy() == 1
    wv = firm.index.map(W).fillna(0).to_numpy(float)
    mt = firm.index.isin(pairs["법인ID_노출"])
    rows = []
    for c in ["log_초기수신", "log_초기운전", "최우수", "우수", "거래기간"] + ind_cols:
        x = firm[c].to_numpy(float)
        sd = np.sqrt((x[T].var(ddof=1) + x[~T].var(ddof=1)) / 2)
        rows.append({"변수": c, "SMD_매칭전": round((x[T].mean() - x[~T].mean()) / sd, 4),
                     "SMD_매칭후": round((x[mt].mean() - np.average(x, weights=np.where(~T, wv, 0))) / sd, 4)})
    bal = pd.DataFrame(rows)
    bal.to_csv(os.path.join(OUT, "balance_table.csv"), index=False, encoding="utf-8-sig")
    print(f"  균형: 매칭후 |SMD| 최대 {bal['SMD_매칭후'].abs().max():.3f}, 0.1 초과 {(bal['SMD_매칭후'].abs() > 0.1).sum()}개")
    return W


def check_old(P):
    """교체 검증: 기존 실행이 끝낸 시차(A) 전부를 새 코드와 비교"""
    old = {-6: (0.0078, 0.1373), -5: (-0.0253, 0.1182), -4: (-0.0171, 0.0806), -3: (-0.0292, 0.0771),
           -2: (-0.0563, 0.0462), 0: (0.0150, 0.0437), 1: (0.0938, 0.0810)}   # run_log_old_code.txt (소수 넷째 자리 출력)
    r, _ = run_common(P, list(old), label="A(새 코드)")
    rows = []
    for _, x in r.iterrows():
        ob, ose = old[int(x["h"])]
        rows.append({"h": int(x["h"]), "β3_기존": ob, "β3_새": round(x["β3"], 4), "SE_기존": ose, "SE_새": round(x["SE"], 4),
                     "β3_일치": round(x["β3"], 4) == ob, "SE_상대차": x["SE"] / ose - 1})
    t = pd.DataFrame(rows)
    t["SE_1%이내"] = t["SE_상대차"].abs() <= 0.01
    t.to_csv(os.path.join(OUT, "code_swap_check.csv"), index=False, encoding="utf-8-sig")
    print(t.round(5).to_string(index=False))
    ok = bool(t["β3_일치"].all() and t["SE_1%이내"].all())
    print("교체 검증 통과" if ok else "교체 검증 실패 — 중단")
    return ok


def main():
    d, P = load()
    if not check_old(P) or "--check-only" in sys.argv:
        return
    W = matching(d)
    w = P.d["법인ID"].map(W).fillna(0).to_numpy()
    res, parts_all = [], {}
    for name, kw in [("A", dict(pf_hs=PF_A)), ("A-원YoY", dict(xkey="원")), ("A-NW", dict(winsor=False)),
                     ("A-양수", dict(positive=True)), ("B", dict(weights=w, pf_hs=set(HS)))]:
        r, parts = run_common(P, HS, label=name, **kw)
        res.append(r); parts_all[name] = parts
        print(f"[{name}] 완료, h=6 β3 {r.loc[r.h == 6, 'β3'].item():+.4f}", flush=True)
    out = pd.concat(res, ignore_index=True).rename(columns={"환산": "10%p하락_차이%"})
    out.to_csv(os.path.join(OUT, "results_table.csv"), index=False, encoding="utf-8-sig")

    pre = {}
    for name in ["A", "B"]:
        r = out[out["모형"] == name]
        pj, chk = joint_test(r, parts_all[name])
        print(f"\n[결합검정 검산 {name}]\n{chk.round(6).to_string(index=False)}", flush=True)
        pre[name] = pj
    pd.DataFrame(pre).T.to_csv(os.path.join(OUT, "pretrend_joint.csv"), encoding="utf-8-sig")
    dep = P.at(P.ln, -1) - P.at(P.ln, -7)
    orig = pd.concat([run_common(P, [0], dep=dep, label="A_원본방식사전")[0],
                      run_common(P, [0], dep=dep, weights=w, label="B_원본방식사전")[0]], ignore_index=True)
    orig.to_csv(os.path.join(OUT, "pretrend_original_method.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("\n[h별 결과]\n", out[["모형", "h", "β3", "SE", "p", "p_holm", "CI_lo", "CI_hi", "10%p하락_차이%", "N", "노출법인", "비노출법인", "월", "pf_β3", "pf_SE"]].round(4).to_string(index=False))
    print("\n[사전추세 LP 결합]\n", pd.DataFrame(pre).T.round(4))
    print("\n[사전추세 원본 방식]\n", orig[["모형", "β3", "SE", "p", "CI_lo", "CI_hi", "N"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
