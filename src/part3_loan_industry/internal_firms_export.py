# -*- coding: utf-8 -*-
"""
내부 시연용 거래처 데이터 — 대시보드 internal 모드의 월보·거래처·거래처 상세를 실제 법인 데이터로 채운다.

  출력: 파트3_여신업종분석/내부시연/firms.json  (로컬 전용, 저장소·외부 공유 금지. **/내부시연/은 두 저장소 모두 gitignore)
  - summary.json(집계만)과 달리 법인 단위 월별 잔액이 들어 있다. 개발 서버가 127.0.0.1에서만 넘겨 주고,
    internal 모드는 빌드가 막혀 있어 배포 결과물에는 들어가지 않는다.
  - 범위: 대구·경북, 금융·보험업 제외 패널 중 외환노출 법인(수출 또는 수입 실적 1회 이상, 1,032곳).
    두 신호 조합의 규칙 대상(수출노출)이 모두 이 안에 있고, 수입만 한 법인은 '비수출'로 비교용.
  - 법인ID는 앞 8자리만 쓴다(1,032곳 안에서 겹치지 않음을 확인).

값(대시보드 src/data/synthetic.ts의 MonthPoint와 같은 순서):
  [요구불예금잔액, 여신_운전자금대출잔액, 운전_할인어음잔액, 외환_수출실적금액]  관측 안 된 달은 null
  잔액은 백만 원, 수출 실적은 백만 달러(추정 단위).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PANEL = HERE / "step1_loan_industry_panel.parquet"
OUT = HERE.parent / "내부시연" / "firms.json"
MONTHS = [y * 100 + m for y in (2023, 2024, 2025) for m in range(1, 13)]
VALUES = ["요구불예금잔액", "여신_운전자금대출잔액", "운전_할인어음잔액", "외환_수출실적금액"]


def num(v):
    v = float(v)
    return int(v) if v.is_integer() else v


def main():
    cols = ["법인ID", "ym", "사업장_시도", "업종_중분류", "법인_고객등급", "외환노출", "수출노출", *VALUES]
    d = pd.read_parquet(PANEL, columns=cols)
    d = d[d["외환노출"] == 1].sort_values(["법인ID", "ym"])
    assert d["법인ID"].str[:8].nunique() == d["법인ID"].nunique(), "앞 8자리가 겹침"
    assert set(d["ym"]) <= set(MONTHS)

    firms = []
    for fid, g in d.groupby("법인ID", sort=False):
        by_month = {int(r.ym): [num(getattr(r, c)) for c in VALUES] for r in g[["ym", *VALUES]].itertuples(index=False)}
        last = g.iloc[-1]
        firms.append({
            "id": fid[:8],
            "region": last["사업장_시도"],
            "industry": last["업종_중분류"],
            "grade": last["법인_고객등급"],  # 등급은 달마다 바뀔 수 있어 마지막 관측 달 기준
            "exporter": bool(g["수출노출"].max() == 1),
            "series": [by_month.get(ym) for ym in MONTHS],
        })

    out = {
        "generatedFrom": "파트3_여신업종분석/단계별 분석/internal_firms_export.py",
        "notice": "실제 은행 법인 데이터(법인 단위). 로컬 내부 시연 전용, 외부 공유·캡처 배포 금지",
        "scope": "대구·경북 외환 거래 법인(수출 또는 수입 실적 1회 이상, 금융·보험업 제외)",
        "exportUnit": "백만 달러",
        "months": MONTHS,
        "firms": firms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    exp = sum(f["exporter"] for f in firms)
    print(f"법인 {len(firms)}곳(수출 {exp}곳), 관측 {len(d):,}행 → {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
