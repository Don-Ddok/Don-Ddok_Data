# -*- coding: utf-8 -*-
"""
1단계-4: 새 계산 3개
  ① 생산지수 YoY — 회사 소재지 지수(주) + 대경 평균(팀 결과 대조용), 두 버전
  ② 운전자금대출 변화 — 충격 직전 달(t-1) 기준, 3·6개월 뒤 (팀 국소투영과 같은 기준)
  ③ 여신한도 방향 — 같은 기준, 한도가 있는 회사만
입력: step1_3_industry.parquet, ../production_index_industry.csv
출력: step1_loan_industry_panel.parquet  (1단계 최종 — GitHub에 올리지 않는다)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_3_industry.parquet")
kosis = pd.read_csv(HERE.parent / "production_index_industry.csv")
kosis = kosis[kosis["level"] == "중분류"][["region", "industry", "ym", "value"]]
kosis = kosis[kosis["value"] > 0]   # 미공표(0) 계열 제외 — step1_3과 같은 규칙
print(f"입력: {len(df):,}행, 법인 {df['법인ID'].nunique():,}개")

THRESH = 0.05  # ±5% (명세서 2-3절)


def yoy(frame, keys):
    """작년 같은 달(ym-100) 값과 비교한 증감률"""
    prev = frame.copy()
    prev["ym"] = prev["ym"] + 100
    out = frame.merge(prev, on=keys + ["ym"], how="left", suffixes=("", "_작년"))
    out["yoy"] = out["value"] / out["value_작년"] - 1
    return out[keys + ["ym", "yoy"]]


# ── ① 생산지수 YoY ─────────────────────────────────────────────
# 소재지 버전: 지역 YoY, 없으면 전국 YoY
reg_yoy = yoy(kosis[kosis["region"].isin(["대구광역시", "경상북도"])], ["region", "industry"])
reg_yoy = reg_yoy.rename(columns={"region": "kosis_region", "industry": "KOSIS업종", "yoy": "_yoy_지역"})
nat_yoy = yoy(kosis[kosis["region"] == "전국"], ["industry"])
nat_yoy = nat_yoy.rename(columns={"industry": "KOSIS업종", "yoy": "_yoy_전국"})

# 대경 평균 버전: 대구·경북 지수를 먼저 평균낸 뒤 YoY (한쪽만 있으면 그쪽만)
dk = (kosis[kosis["region"].isin(["대구광역시", "경상북도"])]
      .groupby(["industry", "ym"], as_index=False)["value"].mean())
dk_yoy = yoy(dk, ["industry"]).rename(columns={"industry": "KOSIS업종", "yoy": "_yoy_대경"})

df = df.merge(reg_yoy, on=["kosis_region", "KOSIS업종", "ym"], how="left")
df = df.merge(nat_yoy, on=["KOSIS업종", "ym"], how="left")
df = df.merge(dk_yoy, on=["KOSIS업종", "ym"], how="left")

df["생산지수_YoY"] = df["_yoy_지역"].fillna(df["_yoy_전국"])
df["생산지수_YoY_대경평균"] = df["_yoy_대경"].fillna(df["_yoy_전국"])


def env(s):
    return np.where(s.isna(), None, np.where(s < 0, "↓", "↑"))


df["수요환경"] = env(df["생산지수_YoY"])
df["수요환경_대경평균"] = env(df["생산지수_YoY_대경평균"])
df = df.drop(columns=["_yoy_지역", "_yoy_전국", "_yoy_대경"])

m = df["KOSIS업종"].notna()
print("\n=== ① 생산지수 YoY ===")
print(f"제조업 행 {m.sum():,}개 중 YoY 계산된 행: {df.loc[m, '생산지수_YoY'].notna().sum():,}")
p1 = (df.loc[m, "수요환경"] == "↓").mean() * 100
p2 = (df.loc[m, "수요환경_대경평균"] == "↓").mean() * 100
agree = (df.loc[m, "수요환경"] != df.loc[m, "수요환경_대경평균"]).mean() * 100
print(f"불황(↓) 비중 — 소재지 버전 {p1:.1f}% / 대경평균 버전 {p2:.1f}%")
print(f"두 버전 판정이 엇갈리는 행: {agree:.1f}%")

# ── ② 운전자금대출 변화 (t-1 기준) ──────────────────────────────
df["_m"] = (df["ym"] // 100) * 12 + (df["ym"] % 100 - 1)       # 달력상 몇 번째 달인지
base = df[["법인ID", "_m", "여신_운전자금대출잔액", "여신한도금액"]]


def value_at(offset, col, name):
    """각 행 기준으로 offset개월 떨어진 달의 값. 그 달이 관측 안 됐으면 빈칸."""
    s = base[["법인ID", "_m", col]].copy()
    s["_m"] = s["_m"] - offset
    return s.rename(columns={col: name})


df = df.merge(value_at(-1, "여신_운전자금대출잔액", "_운전_t-1"), on=["법인ID", "_m"], how="left")
df = df.merge(value_at(-1, "여신한도금액", "_한도_t-1"), on=["법인ID", "_m"], how="left")
for h in (3, 6):
    df = df.merge(value_at(h, "여신_운전자금대출잔액", f"_운전_t+{h}"), on=["법인ID", "_m"], how="left")
df = df.merge(value_at(6, "여신한도금액", "_한도_t+6"), on=["법인ID", "_m"], how="left")


def direction(pct):
    return np.where(pct.isna(), None,
                    np.where(pct > THRESH, "↑", np.where(pct < -THRESH, "↓", "→")))


for h in (3, 6):
    lr = np.log(df[f"_운전_t+{h}"] + 1) - np.log(df["_운전_t-1"] + 1)
    df[f"대출증감률_h{h}"] = lr
    df[f"대출방향_h{h}"] = direction(np.exp(lr) - 1)

# 한 번이라도 운전자금 대출을 받은 적 있는 회사인가
ever = df.groupby("법인ID")["여신_운전자금대출잔액"].max().gt(0).rename("운전자금_보유이력").astype(int)
df = df.merge(ever, on="법인ID", how="left")

print("\n=== ② 운전자금대출 변화 (t-1 → t+h) ===")
print(f"운전자금을 한 번이라도 받은 회사: {df.groupby('법인ID')['운전자금_보유이력'].first().sum():,}개 "
      f"/ 전체 {df['법인ID'].nunique():,}개")
for h in (3, 6):
    col = f"대출방향_h{h}"
    ok = df[col].notna()
    print(f"\n[h={h}] 계산 가능한 행: {ok.sum():,} ({ok.mean()*100:.1f}%) — 나머지는 앞뒤 달이 관측되지 않아 빈칸")
    for label, sub in [("전체 회사", df), ("대출 보유이력 있는 회사만", df[df["운전자금_보유이력"] == 1])]:
        vc = sub[col].value_counts(normalize=True).reindex(["↑", "→", "↓"]).fillna(0) * 100
        print(f"  {label}: ↑ {vc['↑']:.1f}% / → {vc['→']:.1f}% / ↓ {vc['↓']:.1f}%")

# ── ③ 한도 방향 (t-1 기준, 한도>0인 경우만) ───────────────────────
pct_lim = df["_한도_t+6"] / df["_한도_t-1"] - 1
lim_dir = direction(pct_lim)
df["한도방향_h6"] = np.where((df["_한도_t-1"].fillna(0) > 0) & df["_한도_t+6"].notna(), lim_dir, "구분불가")

print("\n=== ③ 한도 방향 (t-1 → t+6) ===")
print(df["한도방향_h6"].value_counts(normalize=True).mul(100).round(1).to_dict())

df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
out = HERE / "step1_loan_industry_panel.parquet"
df.to_parquet(out, index=False)
print(f"\n저장: {out.name} ({len(df):,}행 × {df.shape[1]}열)")
