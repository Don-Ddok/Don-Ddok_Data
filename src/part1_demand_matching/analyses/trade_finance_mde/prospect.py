# -*- coding: utf-8 -*-
"""상거래 결제 금융 3종 + 합계의 사전 MDE 판단 — 본 분석 전 판단용, 회귀 없음 (처치 계수 β 계산 안 함)
대상: TF 무역금융, PF 기업구매자금, AR 외상매출채권담보, SUM3 = TF+PF+AR, SUM4 = SUM3 + 할인어음(표시용)
0단계: 기술 통계
1단계: h=6 MDE 근사, 두 구조
  구조 1 harmonized C1: 법인 FE + 지역×연월 FE + 노출×보정YoY + 노출×영업일수차, 2023-02~2025-06 기준월 (29개월)
  구조 2 firm_shock: 노출 법인 안, 충격 (a') ln(100·S_t+1) − ln(100·S_{t−12}+1), 법인 FE + 지역×연월 FE, 2024-03~ (h=6은 16개월)
  SE ≈ k·σ(ỹ)/(sd(X̃)·√N), ỹ·X̃ = 처치 없는 귀무 모형(FE + 통제)의 잔차
    k1: harmonized 요구불 C1 SE에 맞춤, harmonized 운전자금·거치식·적립식 C1로 검증
    k2: firm_shock 요구불 주 충격 SE에 맞춤, firm_shock 나머지 계정(HOLD 포함)으로 검증
  보조: 귀무 잔차 샌드위치 SE (법인·연월 이중 군집)
  MDE(β) = (t.975 + t.80) × SE, 자유도 = min(법인, 월) − 1
출력: step0_*.csv, calibration.csv, mde_prospect.csv, prospect_log.txt (집계만)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, partial_out

DATA = r"C:\test\data\processed\df_ready.csv"
OUT = r"C:\test\outputs\trade_finance_mde"
HARM = r"C:\test\outputs\harmonized\results_all.csv"
FSR = r"C:\test\outputs\firm_shock\results.csv"
COLS = {"TF": "운전_무역금융잔액", "PF": "운전_기업구매자금대출잔액", "AR": "운전_외상매출채권담보대출잔액", "DB": "운전_할인어음잔액",
        "요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}
NAME = {"TF": "무역금융", "PF": "기업구매자금", "AR": "외상매출채권담보", "SUM3": "TF+PF+AR", "SUM4": "SUM3+할인어음(표시용)"}
LOGF = open(os.path.join(OUT, "prospect_log.txt"), "w", encoding="utf-8")
MULT = lambda dof: stats.t.ppf(0.975, dof) + stats.t.ppf(0.80, dof)
L = np.log(0.9)


def log(s=""):
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def load():
    d = pd.read_csv(DATA, usecols=["기준년월", "법인ID", "사업장_시도", "업종_중분류", "exposed", "외환_수출실적금액", "외환_수입실적금액"] + list(COLS.values()))
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d = d.rename(columns={"기준년월": "ym", "exposed": "외환노출"})
    for k, c in COLS.items():
        d[k] = d[c].clip(lower=0)
    d["SUM3"] = d["TF"] + d["PF"] + d["AR"]
    d["SUM4"] = d["SUM3"] + d["DB"]
    d["_keep_FX"] = d["외환_수출실적금액"] + d["외환_수입실적금액"]
    return d


def components(P, y, X, m, SB=None, hold=False):
    yy = y[m].copy()
    if not hold:
        lo, hi = np.quantile(yy, [0.01, 0.99]); yy = np.clip(yy, lo, hi)
    fi = np.unique(P.firm[m], return_inverse=True)[1]; ri = np.unique(P.regym[m], return_inverse=True)[1]
    M = partial_out(np.column_stack([yy, X[m]] + ([] if SB is None else [SB[m]])), np.ones(int(m.sum())), fi, ri)
    ey, ex = M[:, 0], M[:, 1]
    if SB is not None:
        sb = M[:, 2]; ey = ey - sb * (sb @ ey) / (sb @ sb); ex = ex - sb * (sb @ ex) / (sb @ sb)
    n = len(ey); k = 1 if SB is None else 2
    psi = ex * ey / (ex @ ex); V = 0.0
    for g, sign in [(P.firm[m], 1), (P.month[m], 1), (P.obs[m], -1)]:
        gu, gi = np.unique(g, return_inverse=True); G = len(gu)
        V += sign * G / (G - 1) * (n - 1) / (n - k) * float(np.sum(np.bincount(gi, psi) ** 2))
    fe = pd.Series(P.expo[m]).groupby(P.firm[m]).first()
    Gf, Gm = len(np.unique(P.firm[m])), len(np.unique(P.month[m]))
    return {"N": n, "법인": Gf, "노출법인": int((fe == 1).sum()), "비노출법인": int((fe == 0).sum()), "월": Gm,
            "자유도": min(Gf, Gm) - 1, "σ_y": ey.std(), "sd_X": ex.std(), "raw": ey.std() / (ex.std() * np.sqrt(n)),
            "SE_샌드위치": np.sqrt(max(V, 0))}


def s1(d, col, hold=False):
    """구조 1: harmonized C1 (보유 이력 법인, 노출·비노출)"""
    s = d[d.groupby("법인ID")[col].transform("max") > 0]
    P = Panel(s, col)
    base = P.pos if hold else P.ln
    y = P.at(base, 6) - P.at(base, -1)
    return components(P, y, P.expo * P.X["보정"], ~np.isnan(y), P.expo * P.biz, hold)


def s2(d, col, hold=False):
    """구조 2: firm_shock (노출 법인, 충격 (a') 정의되는 달)"""
    e = d[d["외환노출"] == 1]
    s = e[e.groupby("법인ID")[col].transform("max") > 0]
    P = Panel(s, col)
    fx = P.d["_keep_FX"].to_numpy(float)
    S_now = fx + P.at(fx, -1) + P.at(fx, -2); S_ly = P.at(fx, -12) + P.at(fx, -13) + P.at(fx, -14)
    X = np.log(100 * S_now + 1) - np.log(100 * S_ly + 1)
    base = P.pos if hold else P.ln
    y = P.at(base, 6) - P.at(base, -1)
    return components(P, y, np.nan_to_num(X), ~np.isnan(y) & ~np.isnan(X), None, hold)


def main():
    d = load()
    # ---------------- 보정 ----------------
    hz = pd.read_csv(HARM); hz = hz[(hz["h"] == 6) & hz["업종"].isna() & (hz["모형"] == "C1")].set_index("계정")
    fs = pd.read_csv(FSR); fs = fs[(fs["h"] == 6) & (fs["충격"] == "주_a'")].set_index("계정")
    cal = []
    for a in ["요구불", "운전자금", "거치식", "적립식"]:
        c = s1(d, a); c.update({"구조": "1 harmonized C1", "계정": a, "SE_실제": hz.loc[a, "SE"], "N_실제": hz.loc[a, "N"]}); cal.append(c)
    for a in ["요구불", "운전자금", "거치식", "적립식", "거치식_HOLD", "적립식_HOLD"]:
        hold = a.endswith("HOLD")
        c = s2(d, a.replace("_HOLD", ""), hold); c.update({"구조": "2 firm_shock", "계정": a, "SE_실제": fs.loc[a, "SE"], "N_실제": fs.loc[a, "N"]}); cal.append(c)
    cal = pd.DataFrame(cal)
    K = {}
    for st in cal["구조"].unique():
        ref = cal[(cal["구조"] == st) & (cal["계정"] == "요구불")].iloc[0]
        K[st] = ref["SE_실제"] / ref["raw"]
    cal["k"] = cal["구조"].map(K)
    cal["SE_근사"] = cal["k"] * cal["raw"]
    cal["근사/실제"] = cal["SE_근사"] / cal["SE_실제"]; cal["샌드위치/실제"] = cal["SE_샌드위치"] / cal["SE_실제"]
    cal.to_csv(os.path.join(OUT, "calibration.csv"), index=False, encoding="utf-8-sig")
    log("## 보정 (k는 요구불에서 결정, 나머지는 검증)")
    log(cal[["구조", "계정", "N", "N_실제", "월", "k", "SE_실제", "SE_근사", "근사/실제", "SE_샌드위치", "샌드위치/실제"]].round(4).to_string(index=False))

    # ---------------- 0단계 ----------------
    log("\n## 0단계 기술 통계")
    firm = d.groupby("법인ID").agg(expo=("외환노출", "max"), ind=("업종_중분류", lambda v: v.dropna().iloc[-1] if v.notna().any() else "결측"),
                                   **{f"h_{k}": (k, lambda v: (v > 0).any()) for k in ["TF", "PF", "AR", "DB", "SUM3", "SUM4"]})
    desc, trend = [], []
    for code in ["TF", "PF", "AR", "SUM3", "SUM4"]:
        hold_f = firm[f"h_{code}"]
        s = d[d["법인ID"].isin(hold_f[hold_f].index)].sort_values(["법인ID", "ym"])
        P = Panel(s, code)
        both = ~np.isnan(P.at(P.bal, 6)) & ~np.isnan(P.at(P.bal, -1))
        zc = np.isclose(P.at(P.bal, 6)[both], P.at(P.bal, -1)[both]).mean()
        mt = d.assign(p=d[code] > 0).groupby(["ym", "외환노출"])["p"].sum().unstack().rename(columns={0: "비노출", 1: "노출"})
        mt["전체"] = mt.sum(axis=1)
        first, last = mt["전체"].iloc[:6].mean(), mt["전체"].iloc[-6:].mean()
        chg = last / first - 1
        trend.append(mt.add_prefix(code + "_"))
        desc.append({"코드": code, "상품": NAME[code], "보유_노출": int((hold_f & (firm["expo"] == 1)).sum()),
                     "보유_비노출": int((hold_f & (firm["expo"] == 0)).sum()), "법인월": len(s), "잔액0_비율": (s[code] <= 0).mean(),
                     "t-1_t+6_변화0_비율": zc, "변화0_쌍": int(both.sum()), "월보유_첫6개월평균": first, "월보유_끝6개월평균": last,
                     "변화율": chg, "감소추세": chg <= -0.2,
                     "월보유_노출_최소": int(mt["노출"].min()), "월보유_노출_최대": int(mt["노출"].max())})
    desc = pd.DataFrame(desc)
    desc.to_csv(os.path.join(OUT, "step0_desc.csv"), index=False, encoding="utf-8-sig")
    pd.concat(trend, axis=1).to_csv(os.path.join(OUT, "step0_holders_by_month.csv"), encoding="utf-8-sig")
    log(desc.round(3).to_string(index=False))
    log("  감소추세 = 월 보유 법인 수 끝 6개월 평균이 첫 6개월 평균보다 20% 이상 적음 (기술 통계 표시용, 계산 전 정함)")
    # 4. 동시 보유
    any4 = firm[["h_TF", "h_PF", "h_AR", "h_DB"]]
    cnt = any4.sum(axis=1); cnt = cnt[cnt > 0]
    d4 = d.assign(n=(d[["TF", "PF", "AR", "DB"]] > 0).sum(axis=1))
    same = d4[d4["n"] > 0].groupby("법인ID")["n"].max()
    co = pd.DataFrame({"구분": ["1개", "2개", "3개 이상"],
                       "기간 중 보유 이력 기준": [int((cnt == 1).sum()), int((cnt == 2).sum()), int((cnt >= 3).sum())],
                       "같은 달 동시 보유(최대) 기준": [int((same == 1).sum()), int((same == 2).sum()), int((same >= 3).sum())]})
    co.to_csv(os.path.join(OUT, "step0_coholding.csv"), index=False, encoding="utf-8-sig")
    log(f"\n4종 중 하나라도 보유한 법인 {len(cnt):,}곳"); log(co.to_string(index=False))
    pair = {f"{a}&{b}": int((any4[f"h_{a}"] & any4[f"h_{b}"]).sum()) for a, b in [("TF", "PF"), ("TF", "AR"), ("TF", "DB"), ("PF", "AR"), ("PF", "DB"), ("AR", "DB")]}
    log("쌍별 보유 이력 겹침(법인): " + ", ".join(f"{k} {v}" for k, v in pair.items()))
    # 5. TF 노출 비율과 비노출 업종
    tf = firm[firm["h_TF"]]
    log(f"\nTF 보유 법인 {len(tf)}곳 중 노출 {tf['expo'].mean():.1%} ({int(tf['expo'].sum())}곳)")
    ind = tf[tf["expo"] == 0]["ind"].value_counts()
    ind_e = tf[tf["expo"] == 1]["ind"].value_counts()
    it = pd.DataFrame({"비노출_TF": ind, "노출_TF": ind_e}).fillna(0).astype(int).sort_values("비노출_TF", ascending=False)
    it.to_csv(os.path.join(OUT, "step0_TF_industry.csv"), encoding="utf-8-sig")
    log(it.head(15).to_string())
    # 비노출 TF 법인의 외환 실적: 정의상 0 (exposed = 외환 실적 이력)
    fxmax = d[d["법인ID"].isin(tf[tf["expo"] == 0].index)][["외환_수출실적금액", "외환_수입실적금액"]].max().max()
    log(f"비노출 TF 법인의 외환 실적 최대값 {fxmax} (노출 정의상 0이어야 함)")

    # ---------------- 1단계 MDE ----------------
    rows = []
    for code in ["TF", "PF", "AR", "SUM3", "SUM4"]:
        for st, fn in [("1 harmonized C1", s1), ("2 firm_shock", s2)]:
            c = fn(d, code)
            for how, se in [("근사(k)", K[st] * c["raw"]), ("샌드위치", c["SE_샌드위치"])]:
                mde = MULT(c["자유도"]) * se
                if st.startswith("1"):     # 지역 수출 10%p 하락 시 잔액 증가(가설 방향) = exp(0.1·MDE) − 1
                    hyp, other = (np.exp(0.1 * mde) - 1) * 100, (1 - np.exp(-0.1 * mde)) * 100
                else:                      # 외환 실적 10% 감소 시 잔액 증가(가설 방향) = exp(MDE·|ln 0.9|) − 1
                    hyp, other = (np.exp(mde * abs(L)) - 1) * 100, (1 - np.exp(-mde * abs(L))) * 100
                unit = 0.1 if st.startswith("1") else abs(L)          # 기준 단위(10%p / 10% 감소)의 충격 크기
                rows.append({"코드": code, "구조": st, "방식": how, **{k: c[k] for k in ["N", "법인", "노출법인", "비노출법인", "월", "자유도", "σ_y", "sd_X"]},
                             "SE": se, "MDE_β": mde, "MDE_가설방향(증가)": hyp, "MDE_반대방향": other,
                             "기준단위/잔차SD": unit / c["sd_X"], "MDE_잔차1SD당(증가)": (np.exp(mde * c["sd_X"]) - 1) * 100})
    out = pd.DataFrame(rows)
    out["판단"] = np.where(out["MDE_가설방향(증가)"] <= 5, "진행 후보", np.where(out["MDE_가설방향(증가)"] <= 10, "SUM3 단일 가설 검토", "진행 안 함"))
    out.to_csv(os.path.join(OUT, "mde_prospect.csv"), index=False, encoding="utf-8-sig")
    log("\n## 1단계 MDE (h=6)")
    log(out[["코드", "구조", "방식", "N", "노출법인", "비노출법인", "월", "자유도", "SE", "MDE_가설방향(증가)", "MDE_반대방향", "판단"]].round(3).to_string(index=False))
    log("\n## 단위 비교 (잔차 충격 SD 기준)")
    log(out[["코드", "구조", "방식", "sd_X", "기준단위/잔차SD", "MDE_잔차1SD당(증가)"]].round(3).to_string(index=False))
    # 참고 기준 두 개를 같은 척도로
    for st, a in [("1 harmonized C1", "요구불"), ("2 firm_shock", "요구불")]:
        r = cal[(cal["구조"] == st) & (cal["계정"] == a)].iloc[0]
        mde = MULT(r["자유도"]) * r["SE_실제"]
        unit = 0.1 if st.startswith("1") else abs(L)
        log(f"참고 {st} 요구불: 실제 SE 기준 MDE 잔차 1SD당 {(np.exp(mde * r['sd_X']) - 1) * 100:.2f}%, 기준단위/잔차SD {unit / r['sd_X']:.2f}")


if __name__ == "__main__":
    main()
