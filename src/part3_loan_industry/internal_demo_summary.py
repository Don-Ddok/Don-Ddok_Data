# -*- coding: utf-8 -*-
"""
내부 시연용 집계 — 대시보드 프로토타입의 '참고 신호' 규칙을 실제 법인 데이터에 적용해 집계만 만든다.

  출력: 파트3_여신업종분석/내부시연/summary.json  (로컬 전용, 저장소·외부 공유 금지)
  - 거래처 단위 값(법인ID, 개별 잔액)은 하나도 내보내지 않는다. 월·지역·업종·그룹별 개수와 비율만.
  - 5곳(건) 미만인 칸은 값을 지우고 '가림'으로 표시한다(소수 칸으로 개별 법인이 드러나는 것을 막는 규칙).

규칙(대시보드 src/data/signals.ts와 같음, 수출 거래처 = 수출노출 1):
  지역 수출 전년동월비 < 0  AND  요구불예금 3개월 변화 < −10%  AND  운전자금대출 6개월 변화 ≥ −5%
  기준 근접: 첫째·셋째 조건 충족 + 요구불예금 3개월 변화 −10%~−5%
  3개월 전·6개월 전 값이 없거나 0이면 판정하지 않는다(판정 불가).
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
OUT = HERE.parent / "내부시연" / "summary.json"

MIN_CELL = 5  # 이 값 미만인 개수는 가린다
MIN_INDUSTRY_FIRMS = 10  # 업종 표는 수출 거래처가 이 수 이상인 업종만
DEP_DROP, DEP_NEAR, LOAN_HOLD = -0.10, -0.05, -0.05
FIRST_YM = 202307  # 대출 6개월 변화를 계산할 수 있는 첫 달


def month_index(ym):
    return (ym // 100 - 2023) * 12 + ym % 100 - 1


def mask(n):
    """5 미만이면 None(가림). 0은 드러나도 개별 법인을 특정하지 않으므로 그대로 둔다."""
    n = int(n)
    return None if 0 < n < MIN_CELL else n


def main():
    cols = ["법인ID", "ym", "사업장_시도", "업종_중분류", "수출노출", "외환노출", "운전자금_보유이력",
            "요구불예금잔액", "여신_운전자금대출잔액"]
    d = pd.read_parquet(PANEL, columns=cols)
    d["idx"] = month_index(d["ym"])

    # 같은 법인의 3개월 전 잔고, 6개월 전 대출(그 달에 관측된 경우만)
    lag = d[["법인ID", "idx", "요구불예금잔액", "여신_운전자금대출잔액"]]
    d = d.merge(lag.assign(idx=lag["idx"] + 3)[["법인ID", "idx", "요구불예금잔액"]]
                .rename(columns={"요구불예금잔액": "잔고_3개월전"}), on=["법인ID", "idx"], how="left")
    d = d.merge(lag.assign(idx=lag["idx"] + 6)[["법인ID", "idx", "여신_운전자금대출잔액"]]
                .rename(columns={"여신_운전자금대출잔액": "대출_6개월전"}), on=["법인ID", "idx"], how="left")
    d["잔고변화"] = np.where(d["잔고_3개월전"] > 0, d["요구불예금잔액"] / d["잔고_3개월전"] - 1, np.nan)
    d["대출변화"] = np.where(d["대출_6개월전"] > 0, d["여신_운전자금대출잔액"] / d["대출_6개월전"] - 1, np.nan)

    ex = pd.read_csv(EXPORT, encoding="utf-8-sig")[["region", "ym", "yoy_exp_amt"]]
    d = d.merge(ex.rename(columns={"region": "사업장_시도", "yoy_exp_amt": "수출YoY"}),
                on=["사업장_시도", "ym"], how="left")
    assert d["수출YoY"].notna().all(), "지역 수출 전년동월비가 빠진 행이 있음"

    d = d[d["ym"] >= FIRST_YM].copy()
    judged = d["잔고변화"].notna() & d["대출변화"].notna()
    region_down = d["수출YoY"] < 0
    loan_hold = d["대출변화"] >= LOAN_HOLD
    met = judged & region_down & (d["잔고변화"] < DEP_DROP) & loan_hold
    near = judged & region_down & loan_hold & (d["잔고변화"] <= DEP_NEAR) & ~met
    d["판정"] = judged
    d["충족"] = met
    d["근접"] = near

    exp = d[d["수출노출"] == 1]

    # 1) 월 × 지역: 수출 거래처 수, 판정 가능, 충족, 기준 근접
    monthly = []
    for (ym, region), g in exp.groupby(["ym", "사업장_시도"]):
        monthly.append({
            "ym": int(ym), "region": region,
            "exporters": mask(g["법인ID"].nunique()),
            "judged": mask(g["판정"].sum()),
            "met": mask(g["충족"].sum()),
            "near": mask(g["근접"].sum()),
        })

    # 2) 업종(중분류) × 전체 기간: 수출 거래처 수, 판정 가능 거래처-월, 충족 거래처-월, 충족 비율
    industries = []
    for ind, g in exp.groupby("업종_중분류"):
        firms = g["법인ID"].nunique()
        if firms < MIN_INDUSTRY_FIRMS:
            continue
        j, m = int(g["판정"].sum()), int(g["충족"].sum())
        industries.append({
            "industry": ind, "exporters": firms, "judgedMonths": j,
            "metMonths": mask(m),
            "metShare": None if mask(m) is None or j == 0 else round(m / j * 100, 1),
        })
    industries.sort(key=lambda r: -(r["metShare"] or -1))
    small = exp.groupby("업종_중분류")["법인ID"].nunique()
    other_firms = int(small[small < MIN_INDUSTRY_FIRMS].sum())

    # 3) 연구 비교의 단순판: 운전자금 보유 이력 법인 중 6개월 대출이 5% 넘게 줄어든 비율, 수출 노출 vs 비노출
    #    (매칭 전 단순 비율. 연구의 근거는 매칭 비교이며 이 선은 참고용)
    lc = d[(d["운전자금_보유이력"] == 1) & d["대출변화"].notna()]
    loan_cut = []
    for ym, g in lc.groupby("ym"):
        row = {"ym": int(ym)}
        for key, grp in (("exposed", g[g["수출노출"] == 1]), ("other", g[g["수출노출"] == 0])):
            n = len(grp)
            row[f"{key}N"] = mask(n)
            row[f"{key}CutShare"] = None if mask(n) is None or n == 0 else \
                round(float((grp["대출변화"] < LOAN_HOLD).mean() * 100), 1)
        loan_cut.append(row)

    # 4) 같은 모양(잔고 10% 넘게 감소 + 대출 유지)이 지역 수출 감소 달에 얼마나 흔한가: 수출 vs 비수출
    #    비수출에서도 비슷하게 흔하면, 이 신호는 수출 거래처만의 특징이 아니다
    down = d[judged & region_down]
    pattern = (down["잔고변화"] < DEP_DROP) & (down["대출변화"] >= LOAN_HOLD)
    pattern_rate = {}
    for key, flag in (("exporters", 1), ("others", 0)):
        g = down["수출노출"] == flag
        n = int(g.sum())
        pattern_rate[key] = {"judgedMonths": n, "share": round(float(pattern[g].mean() * 100), 1)}
    dep_only = {k: round(float((down.loc[down["수출노출"] == f, "잔고변화"] < DEP_DROP).mean() * 100), 1)
                for k, f in (("exporters", 1), ("others", 0))}

    out = {
        "generatedFrom": "iM뱅크 교육용 법인 익명데이터(대구·경북, 금융·보험업 제외) 집계",
        "notice": "내부 시연 전용. 저장소·외부 공유 금지. 5 미만 칸은 가림.",
        "minCell": MIN_CELL,
        "firstYm": FIRST_YM,
        "totals": {
            "firms": int(pd.read_parquet(PANEL, columns=["법인ID"])["법인ID"].nunique()),
            "exporters": int(exp["법인ID"].nunique()),
            "metFirmMonths": mask(exp["충족"].sum()),
            "metFirms": mask(exp.loc[exp["충족"], "법인ID"].nunique()),
            "judgedFirmMonths": int(exp["판정"].sum()),
        },
        "monthly": monthly,
        "industries": industries,
        "industryOtherFirms": other_firms,
        "loanCut": loan_cut,
        "patternRate": pattern_rate,
        "depositDropRate": dep_only,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # 확인용 출력(집계만)
    print("전체 법인", out["totals"]["firms"], "/ 수출 거래처", out["totals"]["exporters"])
    print("판정 가능 거래처-월", out["totals"]["judgedFirmMonths"], "/ 충족 거래처-월", out["totals"]["metFirmMonths"],
          "/ 충족 경험 거래처", out["totals"]["metFirms"])
    m = pd.DataFrame(monthly)
    print("월·지역 칸", len(m), "/ 충족 가림 칸", int(m["met"].isna().sum()), "/ 근접 가림 칸", int(m["near"].isna().sum()))
    print(m.pivot(index="ym", columns="region", values="met").to_string())
    print("업종", len(industries), "개 표시, 작은 업종 수출 거래처 합", other_firms)
    for r in industries:
        print(" ", r)
    print("대출 5% 넘게 감소 비율(노출/비노출) 앞 3달:", loan_cut[:3])
    print("지역 수출 감소 달, 판정 가능 거래처-월 중 같은 모양 비율:", pattern_rate)
    print("  그중 잔고 10% 넘게 감소만:", dep_only)
    all_exp = pd.read_parquet(PANEL, columns=["법인ID", "수출노출"])
    print("수출노출 법인(전체 기간):", all_exp.loc[all_exp["수출노출"] == 1, "법인ID"].nunique())
    print("저장:", OUT)


if __name__ == "__main__":
    main()
