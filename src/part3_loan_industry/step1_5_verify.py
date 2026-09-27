# -*- coding: utf-8 -*-
"""
1단계-5: 최종 파일 검증 (명세서 8절 체크리스트 + 무결성 점검)
입력: step1_loan_industry_panel.parquet, ../production_index_industry.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
firm = df.groupby("법인ID").first()
mfg = df["KOSIS업종"].notna()

results = []


def check(name, actual, expected, ok):
    results.append((name, actual, expected, "통과" if ok else "불일치"))


# ── 명세서 8절 체크리스트 ─────────────────────────────────────
check("1. 필터 후 행 수", f"{len(df):,}", "271,460", len(df) == 271_460)
check("2. 법인 수", f"{df['법인ID'].nunique():,}", "11,018", df["법인ID"].nunique() == 11_018)
check("3. 외환노출 법인", f"{firm['외환노출'].sum():,}", "1,032", firm["외환노출"].sum() == 1_032)
check("4. 수출노출 법인", f"{firm['수출노출'].sum():,}", "523", firm["수출노출"].sum() == 523)
n_mfg = df.loc[mfg, "법인ID"].nunique()
check("5. KOSIS 매핑 법인", f"{n_mfg:,}", "3,678", n_mfg == 3_678)
bad6 = (mfg & df["생산지수"].isna()).sum() + (~mfg & df["생산지수"].notna()).sum()
check("6. 생산지수 결측은 비제조업에서만", f"어긋난 행 {bad6}", "0", bad6 == 0)

# 7. YoY<0 비중 — KOSIS 계열 기준으로 어제 값(56.7%) 재현되는지
k = pd.read_csv(HERE.parent / "production_index_industry.csv")
k = k[(k["level"] == "중분류") & k["region"].isin(["대구광역시", "경상북도"])].copy()
k = k[k["value"] > 0]   # 미공표(0) 계열 제외
prev = k[["region", "industry", "ym", "value"]].copy()
prev["ym"] += 100
k = k.merge(prev, on=["region", "industry", "ym"], suffixes=("", "_작년"))
k = k[(k["ym"] >= 202301) & (k["ym"] <= 202512)]
p_kosis = ((k["value"] / k["value_작년"] - 1) < 0).mean() * 100
p_panel = (df.loc[mfg, "수요환경"] == "↓").mean() * 100
check("7. 불황 비중 (KOSIS 계열 기준)", f"{p_kosis:.1f}%", "약 56.7%", abs(p_kosis - 56.7) < 0.1)
check("7-1. 불황 비중 (패널 행 기준)", f"{p_panel:.1f}%", "70% 미만(쏠림 규칙 미발동)", p_panel < 70)

holder = df["운전자금_보유이력"] == 1
flat6 = (df.loc[holder, "대출방향_h6"] == "→").sum() / df.loc[holder, "대출방향_h6"].notna().sum() * 100
check("8. 대출방향_h6 무변동 비중 (보유이력 회사)", f"{flat6:.1f}%", "기록용 (월간 73.2%보다 낮을 것)", flat6 < 73.2)

# ── 무결성 점검 ───────────────────────────────────────────────
dup = df.duplicated(["법인ID", "기준년월"]).sum()
check("9. 회사×월 중복 행", f"{dup}", "0", dup == 0)

per_firm = df.groupby("법인ID")[["외환노출", "수출노출", "거래기간", "규모분위", "운전자금_보유이력"]].nunique()
varying = (per_firm > 1).sum().sum()
check("10. 회사 단위 값이 회사 안에서 일정", f"흔들리는 경우 {varying}", "0", varying == 0)

q = firm["규모분위"].value_counts()
check("11. 규모분위 균등", f"{q.min()}~{q.max()}개", "각 약 2,204개", q.max() - q.min() <= 1)

env_bad = (mfg & df["수요환경"].isna()).sum() + (~mfg & df["수요환경"].notna()).sum()
check("12. 수요환경은 제조업에만 존재", f"어긋난 행 {env_bad}", "0", env_bad == 0)

neg = (df[["여신_운전자금대출잔액", "여신_시설자금대출잔액", "여신한도금액"]] < 0).sum().sum()
check("13. 음수 잔액", f"{neg}", "0", neg == 0)

zero_idx = (mfg & (df["생산지수"] <= 0)).sum()
check("14. 제조업 행에 생산지수 0(미공표 값)", f"{zero_idx}", "0", zero_idx == 0)

# ── 결과 표 ───────────────────────────────────────────────────
print(f"{'항목':<36}{'결과':<22}{'기대':<30}판정")
print("-" * 100)
for name, a, e, v in results:
    print(f"{name:<36}{a:<22}{e:<30}{v}")
n_fail = sum(r[3] == "불일치" for r in results)
print(f"\n총 {len(results)}개 중 통과 {len(results)-n_fail}개, 불일치 {n_fail}개")

# ── 손으로 한 줄 확인 (대출방향_h6 계산이 맞는지) ────────────────
df["_m"] = (df["ym"] // 100) * 12 + (df["ym"] % 100 - 1)
cand = df[holder & (df["대출방향_h6"] == "↑")]
row = cand.sample(1, random_state=7).iloc[0]
fid, m0 = row["법인ID"], row["_m"]
g = df[df["법인ID"] == fid].set_index("_m")["여신_운전자금대출잔액"]
v_prev, v_fwd = g.get(m0 - 1), g.get(m0 + 6)
pct = (np.exp(np.log(v_fwd + 1) - np.log(v_prev + 1)) - 1) * 100
print("\n=== 손 검산: 무작위 한 줄 ===")
print(f"법인 {fid}, 기준월 {row['기준년월']}")
print(f"  직전 달(t-1) 운전자금: {v_prev:,.0f}  →  6개월 뒤(t+6): {v_fwd:,.0f}")
print(f"  변화율 {pct:+.1f}%  →  문턱 ±5% 기준 판정 '{'↑' if pct > 5 else '↓' if pct < -5 else '→'}' / 파일에 저장된 값 '{row['대출방향_h6']}'")
