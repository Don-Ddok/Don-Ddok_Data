# -*- coding: utf-8 -*-
"""
5단계-2: 여신 세부 항목별 회귀 — 어떤 항목이, 언제 다르게 반응하는가 (설계는 작업순서 5단계에 사전 확정)
  항목 5개: 할인어음, 기업구매자금, 기타운전, 시설 일반, 기타시설
  표본: 3-2 매칭 표본 중 그 항목을 한 번이라도 보유한 회사 (매칭 가중치 유지)
  식: 3-3과 동일, 관심 계수 β3 = 수출YoY × 외환노출
  겹치지 않는 시차 4구간 (합치면 t−4 → t+12 전체 변화):
    선행 ln x(t−1) − ln x(t−4) / 단기 ln x(t+3) − ln x(t−1) / 중기 ln x(t+8) − ln x(t+3) / 장기 ln x(t+12) − ln x(t+8)
  표준오차: 법인·월 이중 클러스터, 다중검정: 20개 Holm 보정
  이중 확인: 한 항목의 β3를 회사 더미 회귀로 다시 계산
입력: 패널(common.py), step3_psm_matched.parquet
출력: step5_2_item_results.csv (집계 결과표 — 회사 단위 값 없음)
"""
import sys

import numpy as np
import pandas as pd

from common import HERE, XCOLS, fe_reg, load_panel, lsdv_check, match_weights, twoway_p

sys.stdout.reconfigure(encoding="utf-8")

ITEMS = {
    "할인어음": "운전_할인어음잔액",
    "기업구매자금": "운전_기업구매자금대출잔액",
    "기타운전": "운전_기타운전자금대출잔액",
    "시설 일반": "시설_일반자금대출잔액",
    "기타시설": "시설_기타시설자금대출잔액",
}
WINDOWS = {"선행(t-4→t-1)": (-4, -1), "단기(t-1→t+3)": (-1, 3),
           "중기(t+3→t+8)": (3, 8), "장기(t+8→t+12)": (8, 12)}

df = load_panel()
W_MATCH = match_weights(pd.read_parquet(HERE / "step3_psm_matched.parquet"))
df = df[df["법인ID"].isin(W_MATCH.index)].copy()
df["_m"] = (df["ym"] // 100) * 12 + (df["ym"] % 100 - 1)


def value_at(col, offset):
    """각 행 기준 offset개월 떨어진 달의 값 (관측 안 됐으면 빈칸)"""
    s = df[["법인ID", "_m", col]].copy()
    s["_m"] -= offset
    return df[["법인ID", "_m"]].merge(s, on=["법인ID", "_m"], how="left")[col].to_numpy()


def verdict(p):
    return "유의" if p < 0.05 else "약한 증거" if p < 0.10 else "근거 없음"


rows, check_done = [], None
for name, col in ITEMS.items():
    holders = df.groupby("법인ID")[col].max() > 0
    w = W_MATCH[W_MATCH.index.isin(holders[holders].index)]
    at = {o: value_at(col, o) for o in sorted({o for pair in WINDOWS.values() for o in pair})}
    for wname, (a, b) in WINDOWS.items():
        y = f"{name}|{wname}"
        df[y] = np.log(at[b] + 1) - np.log(at[a] + 1)
        # 장기 구간은 t+12가 필요해 2025년 기준월이 전부 빠짐 → 그 해 더미는 표본에 없으므로 제외
        yrs = df.loc[df[y].notna() & df["법인ID"].isin(w.index), "ym"] // 100
        xcols = [c for c in XCOLS if not c.startswith("연도") or (yrs == int(c[2:])).any()]
        r = fe_reg(df, y, w, xcols)
        se, p = twoway_p(r, 1)
        d = r["d"]
        rows.append({"항목": name, "구간": wname,
                     "노출": int(d.loc[d["외환노출"] == 1, "법인ID"].nunique()),
                     "대조": int(d.loc[d["외환노출"] == 0, "법인ID"].nunique()),
                     "행": r["n"], "β1(대조 반응)": r["b"][0], "β3": r["b"][1], "SE(이중)": se, "p(이중)": p,
                     "클러스터(적은 쪽)": min(r["법인ID"][2], r["ym"][2])})
        if check_done is None and name == "할인어음" and wname.startswith("단기"):
            check_done = (y, r["b"][1], lsdv_check(r, y, xcols))
        df.drop(columns=y, inplace=True)

res = pd.DataFrame(rows)

# Holm 보정 (20개)
order = np.argsort(res["p(이중)"].to_numpy())
m = len(res)
holm = np.empty(m)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * res["p(이중)"].iloc[i]))
    holm[i] = running
res["p(Holm)"] = holm
res["판정(Holm)"] = res["p(Holm)"].map(verdict)
res["판정(보정 전, 참고)"] = res["p(이중)"].map(verdict)

print(f"=== 5-2 결과 (검정 {m}개, β3 = 수출YoY × 외환노출) ===")
with pd.option_context("display.width", 240, "display.max_columns", 20):
    print(res.round(3).to_string(index=False))

y, b_fe, b_lsdv = check_done
print(f"\n이중 확인 ({y}): 평균빼기 β3={b_fe:.5f} / 더미회귀 β3={b_lsdv:.5f} / 차이 {abs(b_fe-b_lsdv):.1e}")
print(f"보정 전 p<0.05: {int((res['p(이중)'] < 0.05).sum())}개 / {m}개 (우연만으로도 약 {m*0.05:.0f}개 기대)")
print(f"Holm 보정 후 p<0.05: {int((res['p(Holm)'] < 0.05).sum())}개")

res.to_csv(HERE / "step5_2_item_results.csv", index=False, encoding="utf-8-sig")
print("\n저장: step5_2_item_results.csv (집계 결과표)")
