# -*- coding: utf-8 -*-
"""
5단계-1: 여신 세부 항목별 데이터 점검 — 분석할 수 있는 항목인가
  항목: 운전자금 하위 7개, 시설자금 하위 3개 (+ 합계 2개 참고)
  항목마다: 보유 회사(한 번이라도 >0), 그중 외환노출, 매칭 표본(3-2) 안의 노출·대조 회사,
            보유 회사 행 중 0인 비율, 월간 무변동 비율(직전 달과 같은 금액)
  표본: 대구·경북 금융 제외 전체 11,018개 (항목마다 그 항목 보유 회사로 한정할 것이므로)
입력: step1_loan_industry_panel.parquet, step3_psm_matched.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
m = pd.read_parquet(HERE / "step3_psm_matched.parquet")
m_t, m_c = set(m["법인ID_노출"]), set(m["법인ID_대조"])

ITEMS = {
    "운전": ["운전_할인어음잔액", "운전_당좌대출잔액", "운전_일반자금대출잔액", "운전_무역금융잔액",
            "운전_기업구매자금대출잔액", "운전_외상매출채권담보대출잔액", "운전_기타운전자금대출잔액"],
    "시설": ["시설_일반자금대출잔액", "시설_에너지절약시설대출잔액", "시설_기타시설자금대출잔액"],
    "합계": ["여신_운전자금대출잔액", "여신_시설자금대출잔액"],
}

df = df.sort_values(["법인ID", "ym"])
df["_m"] = (df["ym"] // 100) * 12 + df["ym"] % 100
exposed = df.groupby("법인ID")["외환노출"].first()

rows = []
for grp, cols in ITEMS.items():
    for c in cols:
        ever = df.groupby("법인ID")[c].max() > 0
        ids = ever[ever].index
        d = df[df["법인ID"].isin(ids)]
        prev = d.groupby("법인ID")[c].shift()
        consec = d.groupby("법인ID")["_m"].diff() == 1          # 바로 직전 달이 관측된 경우만
        flat = (d[c] == prev)[consec].mean() * 100
        rows.append({
            "구분": grp, "항목": c.split("_", 1)[1].replace("잔액", ""),
            "보유 회사": len(ids), "그중 노출": int(exposed[ids].sum()),
            "매칭 노출": len(m_t & set(ids)), "매칭 대조": len(m_c & set(ids)),
            "0인 행(%)": (d[c] == 0).mean() * 100, "월간 무변동(%)": flat,
        })

res = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(res.round(1).to_string(index=False))

# 무역금융: 수출입 실적과 기계적으로 묶여 있는지 (보유 회사 중 외환노출 비율)
tf = res[res["항목"] == "무역금융"].iloc[0]
print(f"\n무역금융 보유 회사 중 외환노출 비율: {tf['그중 노출'] / tf['보유 회사'] * 100:.1f}% "
      f"(전체 평균 {exposed.mean() * 100:.1f}%)")
