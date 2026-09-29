# -*- coding: utf-8 -*-
"""
조합 신호 검증 — "요구불예금 감소 + 할인어음 증가"가 수출 감소 달에 수출 거래처에서 더 자주 나타나는가
(팀원 결론 '동행 신호: 요구불예금 감소와 할인어음 증가의 조합'을 개별 법인 단위로 점검)

설계(실행 전 고정, 2026-09-27)
  표본: 대구·경북, 금융·보험업 제외 패널. 주 표본은 할인어음을 한 번이라도 보유한 법인(미보유 법인의 0→0이 희석하므로).
        참고 표본은 전체 법인(담당자가 모든 거래처를 보는 상황).
  신호(주): 요구불예금잔액 3개월 변화 < −10%  AND  운전_할인어음잔액 3개월 전보다 증가
  신호(보조): 같은 조건을 1개월 변화로(같은 달)
  계산 가능 조건: 3개월(1개월) 전 관측이 있고, 그때 요구불예금 > 0
  비교: 수출노출(수출 실적 1회 이상) vs 비노출, 지역 수출 전년동월비 감소 달 vs 증가 달. 보조로 외환노출(수출 또는 수입)
  통과 기준(둘 다):
    ① 수출 감소 달에 노출 법인의 신호 비율이 비노출의 1.5배 이상
    ② 노출 초과분(노출 − 비노출)이 감소 달에서 증가 달보다 크고, 법인 단위 재표본 500회 95% 구간이 0을 포함하지 않음
  한계(미리 적음): 수출대금이 요구불계좌로 들어오므로 노출 법인의 잔고 감소는 기계적일 수 있고, 이 검증은 그것을 가르지 못함.

출력은 집계만. 결과 파일 파트3_여신업종분석/내부시연/combo_check.json(로컬 전용).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = HERE / "step1_loan_industry_panel.parquet"
EXPORT = ROOT / "외부데이터" / "export_region.csv"
OUT = HERE.parent / "내부시연" / "combo_check.json"

DEP_DROP = -0.10
RATIO_BAR = 1.5
B = 500
SEED = 20260927


def month_index(ym):
    return (ym // 100 - 2023) * 12 + ym % 100 - 1


def build(window):
    cols = ["법인ID", "ym", "사업장_시도", "수출노출", "외환노출", "요구불예금잔액", "운전_할인어음잔액"]
    d = pd.read_parquet(PANEL, columns=cols)
    d["idx"] = month_index(d["ym"])
    lag = d[["법인ID", "idx", "요구불예금잔액", "운전_할인어음잔액"]].copy()
    lag["idx"] += window
    d = d.merge(lag.rename(columns={"요구불예금잔액": "dep_lag", "운전_할인어음잔액": "bill_lag"}),
                on=["법인ID", "idx"], how="left")
    d = d[d["dep_lag"].notna() & (d["dep_lag"] > 0)].copy()
    d["dep_chg"] = d["요구불예금잔액"] / d["dep_lag"] - 1
    d["signal"] = (d["dep_chg"] < DEP_DROP) & (d["운전_할인어음잔액"] > d["bill_lag"])
    ex = pd.read_csv(EXPORT, encoding="utf-8-sig")[["region", "ym", "yoy_exp_amt"]]
    d = d.merge(ex.rename(columns={"region": "사업장_시도"}), on=["사업장_시도", "ym"], how="left")
    assert d["yoy_exp_amt"].notna().all()
    d["down"] = d["yoy_exp_amt"] < 0
    return d


def rates(d, treat):
    out = {}
    for phase, g in (("down", d[d["down"]]), ("up", d[~d["down"]])):
        for grp, flag in (("exposed", 1), ("other", 0)):
            s = g.loc[g[treat] == flag, "signal"]
            out[f"{phase}_{grp}"] = (float(s.mean() * 100) if len(s) else float("nan"), int(len(s)), int(s.sum()))
    return out


def did(d, treat):
    r = rates(d, treat)
    return (r["down_exposed"][0] - r["down_other"][0]) - (r["up_exposed"][0] - r["up_other"][0])


def boot_did(d, treat, rng):
    """법인 단위 재표본. 법인별 (감소 달·증가 달) 법인-월 수와 신호 수를 먼저 합쳐 두고, 뽑힌 횟수로 가중해 비율을 다시 계산"""
    g = d.groupby(["법인ID", "down"])["signal"].agg(["size", "sum"]).unstack(fill_value=0)
    firm = pd.DataFrame({
        "n_down": g[("size", True)] if ("size", True) in g else 0,
        "h_down": g[("sum", True)] if ("sum", True) in g else 0,
        "n_up": g[("size", False)] if ("size", False) in g else 0,
        "h_up": g[("sum", False)] if ("sum", False) in g else 0,
    }).fillna(0)
    firm["t"] = d.groupby("법인ID")[treat].first().reindex(firm.index).to_numpy()
    arr = firm[["n_down", "h_down", "n_up", "h_up"]].to_numpy(float)
    t = firm["t"].to_numpy() == 1
    k = len(firm)
    vals = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)

        def rate(mask, n, h):
            den = (w[mask] * arr[mask, n]).sum()
            return (w[mask] * arr[mask, h]).sum() / den * 100 if den else np.nan

        v = (rate(t, 0, 1) - rate(~t, 0, 1)) - (rate(t, 2, 3) - rate(~t, 2, 3))
        vals.append(v)
    return np.nanpercentile(vals, [2.5, 97.5])


def evaluate(d, treat, rng, label):
    r = rates(d, treat)
    ratio = r["down_exposed"][0] / r["down_other"][0] if r["down_other"][0] else float("nan")
    dd = did(d, treat)
    lo, hi = boot_did(d, treat, rng)
    pass1 = ratio >= RATIO_BAR
    pass2 = dd > 0 and lo > 0
    firms_exposed = int(d.loc[d[treat] == 1, "법인ID"].nunique())
    firms_other = int(d.loc[d[treat] == 0, "법인ID"].nunique())
    print(f"\n[{label}] 처치={treat} 노출 법인 {firms_exposed} / 비노출 {firms_other}")
    for k, (rate, n, hit) in r.items():
        print(f"  {k:14s} 비율 {rate:5.1f}%  (신호 {hit} / 법인-월 {n})")
    print(f"  ① 감소 달 노출/비노출 = {ratio:.2f}배 → {'통과' if pass1 else '미통과'} (기준 {RATIO_BAR}배)")
    print(f"  ② 초과분 차이(감소−증가) = {dd:+.2f}%p, 95% 구간 [{lo:+.2f}, {hi:+.2f}] → {'통과' if pass2 else '미통과'}")
    return {"label": label, "treat": treat, "firmsExposed": firms_exposed, "firmsOther": firms_other,
            "rates": {k: {"pct": round(v[0], 2), "n": v[1], "hits": v[2]} for k, v in r.items()},
            "ratioDown": round(ratio, 3), "did": round(dd, 3), "didCI": [round(lo, 3), round(hi, 3)],
            "pass1": bool(pass1), "pass2": bool(pass2)}


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for window, wlabel in ((3, "3개월(주)"), (1, "1개월(보조)")):
        d = build(window)
        holders = d.groupby("법인ID")["운전_할인어음잔액"].transform("max") > 0
        # 보유 이력은 전체 기간 기준으로 다시 판정(이 창의 관측만으로 정하지 않음)
        full = pd.read_parquet(PANEL, columns=["법인ID", "운전_할인어음잔액"])
        ever = full.groupby("법인ID")["운전_할인어음잔액"].max() > 0
        hold = d[d["법인ID"].isin(ever[ever].index)]
        results.append(evaluate(hold, "수출노출", rng, f"{wlabel} · 할인어음 보유 법인"))
        if window == 3:
            results.append(evaluate(hold, "외환노출", rng, f"{wlabel} · 할인어음 보유 법인"))
            results.append(evaluate(d, "수출노출", rng, f"{wlabel} · 전체 법인(참고)"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"design": "combo_signal_check.py 머리말", "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
