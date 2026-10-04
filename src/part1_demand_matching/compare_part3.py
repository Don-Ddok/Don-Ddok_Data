# -*- coding: utf-8 -*-
"""파트3(팀원) 조합 신호 결과와 민영 분석의 비교 — 수출 감소 달 정의만 바꿔 같은 판정 기준에 넣는다.

팀원 코드(Don-Ddok_Data/src/part3_loan_industry)는 수정하지 않는다. 같은 폴더 구조의 작업 폴더
(external/part3_work)에서 팀원 파이프라인을 그대로 돌려 만든 패널을 읽고, combo_signal_check.py의
rates·did·boot_did·evaluate 함수를 그대로 불러 쓴다.

비교 축: '지역 수출 감소 달' 정의
  A. 팀원 원본: 원 수출 YoY < 0 (영업일수 보정 없음)
  B. 민영 트리거와 같은 기준: 보정 수출 YoY < 0 (영업일 1일당 +4.46%p 성분 제거)
그 밖의 표본·신호·통과 기준·재표본 횟수·seed는 팀원 설계 그대로다.
대시보드 규칙(internal_demo_summary.py: 수출 감소 + 잔고 3개월 −10% + 운전자금 6개월 ≥ −5%)도 같은 두 기준으로 센다.

출력(집계만): 분석결과\\비교_파트3\\combo_compare.json, dashboard_rule_compare.csv
실행: py -3.13 compare_part3.py   (팀원 환경과 같은 3.13, 팀원 파이프라인 실행 후)
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
WORK = r"C:\test\external\part3_work\통계 프로젝트\여신업종분석\단계별 분석"
TRIG = r"C:\test\분석결과\matching\trigger_stages.csv"
OUT = r"C:\test\분석결과\비교_파트3"
FOCUS_YM = 202501            # 팀원 대시보드 화면의 기준월


def load_team():
    spec = importlib.util.spec_from_file_location("combo", os.path.join(WORK, "combo_signal_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # main()은 실행되지 않음 (__name__ 보호)
    return m


def adj_table():
    t = pd.read_csv(TRIG)[["지역", "연월", "원YoY", "보정YoY"]]
    return t.rename(columns={"지역": "사업장_시도", "연월": "ym"})


def with_down(d, kind, adj):
    d = d.drop(columns=[c for c in ("원YoY", "보정YoY") if c in d.columns]).merge(adj, on=["사업장_시도", "ym"], how="left")
    assert d["보정YoY"].notna().all()
    d["down"] = (d["yoy_exp_amt"] if kind == "원" else d["보정YoY"]) < 0
    return d


def combo(m, adj):
    out = []
    full = pd.read_parquet(m.PANEL, columns=["법인ID", "운전_할인어음잔액"])
    ever = full.groupby("법인ID")["운전_할인어음잔액"].max() > 0
    for window, wl in ((3, "3개월(주)"), (1, "1개월(보조)")):
        base = m.build(window)
        hold = base[base["법인ID"].isin(ever[ever].index)]
        cases = [(hold, "수출노출", "할인어음 보유 법인")]
        if window == 3:
            cases += [(hold, "외환노출", "할인어음 보유 법인"), (base, "수출노출", "전체 법인(참고)")]
        for d0, treat, lab in cases:
            for kind in ("원", "보정"):
                rng = np.random.default_rng(m.SEED)      # 두 기준을 같은 난수로 비교
                r = m.evaluate(with_down(d0, kind, adj), treat, rng, f"{wl} · {lab} · 수출 감소 달={kind} YoY")
                r.update({"window": window, "sample": lab, "down_def": kind})
                out.append(r)
    return out


def dashboard_rule(adj):
    cols = ["법인ID", "ym", "사업장_시도", "수출노출", "요구불예금잔액", "여신_운전자금대출잔액"]
    d = pd.read_parquet(os.path.join(WORK, "step1_loan_industry_panel.parquet"), columns=cols)
    d["idx"] = (d["ym"] // 100 - 2023) * 12 + d["ym"] % 100 - 1
    lag = d[["법인ID", "idx", "요구불예금잔액", "여신_운전자금대출잔액"]]
    d = d.merge(lag.assign(idx=lag["idx"] + 3)[["법인ID", "idx", "요구불예금잔액"]].rename(columns={"요구불예금잔액": "dep3"}), on=["법인ID", "idx"], how="left")
    d = d.merge(lag.assign(idx=lag["idx"] + 6)[["법인ID", "idx", "여신_운전자금대출잔액"]].rename(columns={"여신_운전자금대출잔액": "loan6"}), on=["법인ID", "idx"], how="left")
    d["dep_chg"] = np.where(d["dep3"] > 0, d["요구불예금잔액"] / d["dep3"] - 1, np.nan)
    d["loan_chg"] = np.where(d["loan6"] > 0, d["여신_운전자금대출잔액"] / d["loan6"] - 1, np.nan)
    d = d[d["ym"] >= 202307].merge(adj, on=["사업장_시도", "ym"], how="left")
    judged = d["dep_chg"].notna() & d["loan_chg"].notna()
    shape = judged & (d["dep_chg"] < -0.10) & (d["loan_chg"] >= -0.05)
    rows = []
    for kind, col in (("원", "원YoY"), ("보정", "보정YoY")):
        down = d[col] < 0
        for grp, flag in (("수출 거래처", 1), ("비수출", 0)):
            g = d["수출노출"] == flag
            rows.append({"수출감소_정의": kind, "집단": grp,
                         "감소달_판정가능_법인월": int((judged & down & g).sum()),
                         "감소달_같은모양_비율(%)": round(float(shape[judged & down & g].mean() * 100), 2),
                         "증가달_같은모양_비율(%)": round(float(shape[judged & ~down & g].mean() * 100), 2),
                         "충족_법인월(수출거래처)": int((shape & down & g).sum()) if flag == 1 else None,
                         f"{FOCUS_YM}_충족_법인수": int((shape & down & g & (d["ym"] == FOCUS_YM)).sum()) if flag == 1 else None})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    adj = adj_table()
    m = load_team()
    res = combo(m, adj)
    with open(os.path.join(OUT, "combo_compare.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    tab = dashboard_rule(adj)
    tab.to_csv(os.path.join(OUT, "dashboard_rule_compare.csv"), index=False, encoding="utf-8-sig")
    print("\n[대시보드 규칙, 두 기준]\n" + tab.to_string(index=False))
    diff = adj[(adj["원YoY"] < 0) != (adj["보정YoY"] < 0)]
    print(f"\n[수출 감소 판정이 두 기준에서 다른 달] {len(diff)}개\n" + diff.to_string(index=False))


if __name__ == "__main__":
    main()
