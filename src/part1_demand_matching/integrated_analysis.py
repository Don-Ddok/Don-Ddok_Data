# -*- coding: utf-8 -*-
"""요구불(민영) × 여신·업종(파트3) 통합 관계 분석 — 집계만 저장

사양은 결과를 보기 전에 기존 분석에서 그대로 가져온다 (탐색적, 새 판정 기준 없음).
 1) 같은 식, 다른 계정: beta_sum_test.py(요구불 β₁+β₃ 재검정)와 같은 식·표본·군집·Holm에
    종속변수만 운전자금대출잔액, 운전_할인어음잔액으로 바꿔 h=0~12 추정 (충격 = 트리거 보정 YoY).
 2) 지역×월 상관 (72개): 원/보정 수출 YoY, 영업일수 전년차, 노출·비노출 법인의 요구불·운전자금·할인어음
    3개월 로그 변화 평균(법인-월 1%/99% 윈저라이즈 후 평균). Pearson.
 3) 법인 단위 동행: 운전자금 보유 이력 법인의 6개월 로그 변화 Δ요구불·Δ운전자금 Spearman 상관을
    노출/비노출 × 수출 감소 달/증가 달(보정 YoY 기준)로.
원본(beta_sum_test.py, lib.py, 데이터 폴더)은 수정하지 않는다.
출력: 분석결과\\통합분석\\*.csv, integrated.json
실행: py -3.11 integrated_analysis.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r"C:\test\code")
import beta_sum_test as B          # est·holm·lead 함수 재사용 (수정하지 않음)

DATA = r"C:\test\data\processed\df_ready.csv"
TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
BIZ = r"C:\test\data\ext_bizday.csv"
OUT = r"C:\test\분석결과\통합분석"
OUTCOMES = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "할인어음": "운전_할인어음잔액"}
H = list(range(0, 13))
WIN = (0.01, 0.99)


def load():
    cols = ["기준년월", "법인ID", "사업장_시도", "exposed", "exp_yoy"] + list(OUTCOMES.values())
    d = pd.read_csv(DATA, usecols=cols)
    d["사업장_시도"] = d["사업장_시도"].astype(str).str.strip()
    d = d[d["사업장_시도"].isin(["대구", "경북"]) & d["기준년월"].between(202301, 202512)].copy()
    d["지역"] = d["사업장_시도"]
    t = pd.read_csv(TRIG)[["지역", "연월", "원YoY", "보정YoY"]].rename(columns={"연월": "기준년월"})
    d = d.merge(t, on=["지역", "기준년월"], how="left").sort_values(["법인ID", "기준년월"]).reset_index(drop=True)
    d["midx"] = (d["기준년월"] // 100) * 12 + d["기준년월"] % 100
    for k, c in OUTCOMES.items():
        d[f"ln_{k}"] = np.log(d[c].clip(lower=0) + 1)
    return d


def lp_all(d):
    rows = []
    for k in OUTCOMES:
        x = d.copy()
        x["ln_y"] = x[f"ln_{k}"]
        if k != "요구불":                        # 계정 보유 이력이 있는 법인만 (0→0 희석 방지, 파트3와 같은 취지)
            ever = x.groupby("법인ID")[OUTCOMES[k]].transform("max") > 0
            x = x[ever]
        res = []
        for h in H:
            r = B.est(x, h, "보정YoY"); r["계정"] = k; res.append(r)
            print(f"[{k}] h={h:2d} β1+β3 {r['b13']:+.4f} (p {r['p13']:.3f}) β3 {r['b3']:+.4f} (p {r['p3']:.3f}) β1 {r['b1']:+.4f}", flush=True)
        res = pd.DataFrame(res)
        for p in ["p13", "p3", "p1"]:
            res[f"{p}_holm"] = B.holm(res[p])
        rows.append(res)
    out = pd.concat(rows)
    out["노출_효과%"] = -out["b13"] * 100          # 수출 YoY 10%p 하락 시 누적 로그 변화 ×100
    out["비노출_효과%"] = -out["b1"] * 100
    out["차이_효과%"] = -out["b3"] * 100
    return out


def chg(d, k, lag):
    key = d[["법인ID", "midx", f"ln_{k}"]]
    b = key.assign(midx=key["midx"] + lag).rename(columns={f"ln_{k}": "lag"})
    x = d[["법인ID", "midx"]].merge(b, on=["법인ID", "midx"], how="left")
    return (d[f"ln_{k}"].values - x["lag"].values)


def region_corr(d):
    for k in OUTCOMES:
        v = pd.Series(chg(d, k, 3))
        lo, hi = v.quantile(list(WIN)); d[f"d3_{k}"] = v.clip(lo, hi).values
    g = d.groupby(["지역", "기준년월", "exposed"])[[f"d3_{k}" for k in OUTCOMES]].mean().unstack("exposed")
    g.columns = [f"{'노출' if e == 1 else '비노출'}_{c[3:]}" for c, e in g.columns]
    m = d.drop_duplicates(["지역", "기준년월"]).set_index(["지역", "기준년월"])[["원YoY", "보정YoY"]]
    biz = pd.read_csv(BIZ)
    bz = dict(zip(biz["ym"], biz["bizdays"]))
    m["영업일수차"] = [bz[ym] - bz[ym - 100] for _, ym in m.index]
    t = m.join(g).dropna()
    corr = t.corr(method="pearson").round(3)
    pv = pd.DataFrame(index=t.columns, columns=t.columns, dtype=float)
    for a in t.columns:
        for b in t.columns:
            pv.loc[a, b] = stats.pearsonr(t[a], t[b])[1] if a != b else 0.0
    return t, corr, pv.round(4)


def firm_comove(d):
    ever = d.groupby("법인ID")["여신_운전자금대출잔액"].transform("max") > 0
    x = d.assign(dd=chg(d, "요구불", 6), dl=chg(d, "운전자금", 6))[ever].dropna(subset=["dd", "dl"])
    x = x[(x["dd"] != 0) | (x["dl"] != 0)]
    rows = []
    for e, eg in [(1, "노출"), (0, "비노출")]:
        for ph, cond in [("수출 감소 달", x["보정YoY"] < 0), ("수출 증가 달", x["보정YoY"] >= 0)]:
            s = x[(x["exposed"] == e) & cond]
            r, p = stats.spearmanr(s["dd"], s["dl"])
            rows.append({"집단": eg, "국면": ph, "법인월": int(len(s)), "법인": int(s["법인ID"].nunique()),
                         "Spearman": round(float(r), 3), "p": float(p),
                         "요구불↓10%+·대출유지 비율(%)": round(float(((s["dd"] < np.log(.9)) & (s["dl"] >= np.log(.95))).mean() * 100), 1)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    print(f"표본: 법인 {d['법인ID'].nunique():,}, 법인×월 {len(d):,}, 노출 {d.loc[d['exposed'] == 1, '법인ID'].nunique():,}")
    lp = lp_all(d)
    lp.to_csv(os.path.join(OUT, "lp_3계정.csv"), index=False, encoding="utf-8-sig")
    t, corr, pv = region_corr(d)
    t.round(4).to_csv(os.path.join(OUT, "지역월_집계.csv"), encoding="utf-8-sig")
    corr.to_csv(os.path.join(OUT, "지역월_상관.csv"), encoding="utf-8-sig")
    pv.to_csv(os.path.join(OUT, "지역월_상관_p.csv"), encoding="utf-8-sig")
    fc = firm_comove(d)
    fc.to_csv(os.path.join(OUT, "법인_동행.csv"), index=False, encoding="utf-8-sig")
    js = {"lp": lp[["계정", "h", "노출_효과%", "비노출_효과%", "차이_효과%", "p13_holm", "p3_holm", "p1", "ci13_lo", "ci13_hi", "N", "법인"]].round(4).to_dict("records"),
          "corr": corr.to_dict(), "p": pv.to_dict(), "firm": fc.to_dict("records"), "n_region_month": int(len(t))}
    with open(os.path.join(OUT, "integrated.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250)
    print("\n[3계정 h=3·6·12]\n", lp[lp["h"].isin([3, 6, 12])][["계정", "h", "노출_효과%", "비노출_효과%", "차이_효과%", "p13_holm", "p3_holm", "p1", "법인"]].round(3).to_string(index=False))
    print("\n[지역×월 상관 — 수출·영업일 행]\n", corr.loc[["원YoY", "보정YoY", "영업일수차"]].to_string())
    print("\n[법인 동행]\n", fc.to_string(index=False))


if __name__ == "__main__":
    main()
