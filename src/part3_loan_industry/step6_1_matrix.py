# -*- coding: utf-8 -*-
"""
6단계-1: 분류 매트릭스 — 전체 (검정 없음, 서술용. 설계는 작업순서 2장)
  세로: 업종 수요환경 (회사 소재지 생산지수 YoY<0 → ↓)
  가로: 운전자금 대출 방향 (t−1 → t+6, ±5%, 사이는 → 무변동)
  대출 축소형(수요↓·대출↓)만 한도 방향으로 하위분할
  표본: 제조업(KOSIS 매핑) + 운전자금 보유 회사, 회사×월 행
  비중은 행 기준(주) + 회사 균등 가중(참고: 오래 관측된 회사가 여러 표를 행사하지 않도록)
입력: step1_loan_industry_panel.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
df = df[df["KOSIS업종"].notna() & (df["운전자금_보유이력"] == 1)].copy()
df["_fw"] = 1 / df.groupby("법인ID")["법인ID"].transform("size")   # 회사 균등 가중

TYPES = {("↑", "↑"): "성장형", ("↑", "→"): "관망", ("↑", "↓"): "건전 디레버리징",
         ("↓", "↑"): "자금압박형", ("↓", "→"): "압박 잠복", ("↓", "↓"): "대출 축소형"}
ORDER = ["성장형", "관망", "건전 디레버리징", "자금압박형", "압박 잠복", "대출 축소형"]


def matrix(d, env_col, dir_col, label):
    ok = d[env_col].notna() & d[dir_col].notna()
    s = d[ok]
    typ = pd.Series([TYPES[(e, l)] for e, l in zip(s[env_col], s[dir_col])], index=s.index)
    print(f"\n{'=' * 12} {label} {'=' * 12}")
    print(f"표본: 제조업·운전자금 보유 회사 {d['법인ID'].nunique():,}개, {len(d):,}행 → "
          f"판정 가능 {ok.sum():,}행 ({ok.mean() * 100:.1f}%, 나머지는 앞뒤 달 미관측)")
    flat = (s[dir_col] == "→")
    print(f"** 무변동(→) 비중: 행 기준 {flat.mean() * 100:.1f}% / 회사 균등 "
          f"{(flat * s['_fw']).sum() / s['_fw'].sum() * 100:.1f}% **")
    tab = pd.crosstab(s[env_col], s[dir_col], normalize="all").reindex(
        index=["↑", "↓"], columns=["↑", "→", "↓"]) * 100
    tab.index = ["수요 ↑(호황)", "수요 ↓(불황)"]
    tab.columns = ["대출 ↑", "대출 →", "대출 ↓"]
    print("\n행 기준 비중(%, 전체 합 100)")
    print(tab.round(1).to_string())
    shares = pd.DataFrame({
        "행 기준(%)": typ.value_counts(normalize=True) * 100,
        "회사 균등(%)": s.groupby(typ)["_fw"].sum() / s["_fw"].sum() * 100,
    }).reindex(ORDER)
    print("\n유형별 비중")
    print(shares.round(1).to_string())
    return s, typ


s, typ = matrix(df, "수요환경", "대출방향_h6", "주: 소재지 지수 × 6개월 (h=6)")

# 대출 축소형 하위분할 (한도 방향)
sub = s[typ == "대출 축소형"]
lim = sub["한도방향_h6"].map({"↓": "신용긴축 의심", "↑": "수요감소형", "→": "수요감소형", "구분불가": "구분 불가"})
print(f"\n=== 대출 축소형 하위분할 ({len(sub):,}행, 전체 판정 행의 {len(sub) / len(s) * 100:.1f}%) ===")
print((lim.value_counts(normalize=True) * 100).reindex(["신용긴축 의심", "수요감소형", "구분 불가"]).round(1).to_string())
print(f"→ 전체 판정 행 대비 '신용긴축 의심': {(lim == '신용긴축 의심').sum() / len(s) * 100:.2f}%")

# 노출 / 비노출별 유형 비중 (3~4단계 결과와의 연결, 서술용)
print("\n=== 외환노출 여부별 유형 비중 (행 기준 %, h=6) ===")
by = pd.crosstab(s["외환노출"].map({1: "노출", 0: "비노출"}), typ, normalize="index").reindex(columns=ORDER) * 100
print(by.round(1).to_string())

# 참고 버전들
matrix(df, "수요환경", "대출방향_h3", "참고: 소재지 지수 × 3개월 (h=3)")
matrix(df, "수요환경_대경평균", "대출방향_h6", "참고: 대경 평균 지수(팀 방식) × 6개월")
