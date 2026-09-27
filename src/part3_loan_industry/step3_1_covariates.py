# -*- coding: utf-8 -*-
"""
3단계-1: 짝짓기 기준값(공변량) 후보 비교 — PSM 돌리기 전 점검
  문제 ① 규모분위에 운전·시설자금 잔액이 들어 있다 → 결과(대출)로 결과를 맞추는 셈
  문제 ② 고객등급이 회사 절반에서 바뀐다 → 충격 이후 등급으로 맞출 위험
  후보: 규모 = (현행) 수신+여신 / 수신만 / 첫 관측 3개월 수신만
        등급 = (현행) 최빈값 / 첫 관측월 등급
표본: 운전자금 보유이력 회사 9,528개
입력: step1_loan_industry_panel.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "step1_loan_industry_panel.parquet")
df = df[df["운전자금_보유이력"] == 1].copy()
DEP = ["요구불예금잔액", "거치식예금잔액", "적립식예금잔액"]
df["수신합"] = df[DEP].sum(axis=1)

# 각 회사의 첫 관측월 기준 순번 (0 = 첫 달)
df = df.sort_values(["법인ID", "ym"])
df["순번"] = df.groupby("법인ID").cumcount()
first = df[df["순번"] == 0].set_index("법인ID")
early = df[df["순번"] < 3].groupby("법인ID")["수신합"].mean()

firm = df.groupby("법인ID").agg(외환노출=("외환노출", "first"), 거래기간=("거래기간", "first"),
                               규모분위_현행=("규모분위", "first"), 수신평균=("수신합", "mean"),
                               첫관측=("ym", "min"))
firm["수신초기"] = early
firm["등급_첫달"] = first["법인_고객등급"]
RANK = {"최우수": 3, "우수": 2, "일반": 1}
firm["등급_최빈"] = (df.groupby(["법인ID", "법인_고객등급"]).size().rename("n").reset_index()
                    .assign(r=lambda x: x["법인_고객등급"].map(RANK))
                    .sort_values(["법인ID", "n", "r"], ascending=[True, False, False])
                    .drop_duplicates("법인ID").set_index("법인ID")["법인_고객등급"])


def quint(s):
    return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


firm["규모분위_현행"] = firm["규모분위_현행"].astype(int)
firm["규모분위_수신"] = quint(firm["수신평균"])
firm["규모분위_초기수신"] = quint(firm["수신초기"])

T = firm["외환노출"] == 1
print(f"표본: 운전자금 보유 회사 {len(firm):,}개 — 외환노출 {T.sum():,}개 / 비노출 {(~T).sum():,}개")
print(f"첫 관측월 분포: 2023-01에 이미 있던 회사 {(firm['첫관측'] == 202301).mean()*100:.1f}%")


def smd(x):
    a, b = x[T].astype(float), x[~T].astype(float)
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


# ── ① 규모 후보 비교 ─────────────────────────────────────────────
print("\n=== ① 규모 기준 후보: 노출/비노출 차이(SMD, 매칭 전) ===")
print(f"{'후보':<22}{'SMD':>8}{'현행과 같은 분위':>18}")
for col, name in [("규모분위_현행", "현행(수신+여신, 전기간)"), ("규모분위_수신", "수신만(전기간)"),
                  ("규모분위_초기수신", "수신만(첫 3개월)")]:
    same = (firm[col] == firm["규모분위_현행"]).mean() * 100
    print(f"{name:<22}{smd(firm[col]):>8.3f}{same:>17.1f}%")

# ── ② 등급 후보 비교 ─────────────────────────────────────────────
print("\n=== ② 등급 기준 후보 ===")
agree = (firm["등급_첫달"] == firm["등급_최빈"]).mean() * 100
print(f"첫 달 등급 = 최빈 등급인 회사: {agree:.1f}%")
for col in ["등급_첫달", "등급_최빈"]:
    top = (firm[col] == "최우수")
    print(f"{col}: 최우수 비율 노출 {top[T].mean()*100:.1f}% / 비노출 {top[~T].mean()*100:.1f}% "
          f"→ 차이 {(top[T].mean()-top[~T].mean())*100:.1f}%p, SMD {smd(top):.3f}")

# 등급이 바뀐 방향 (첫 달 → 마지막 달)
last = df.groupby("법인ID")["법인_고객등급"].last().map(RANK)
move = np.sign(last - firm["등급_첫달"].map(RANK))
print("\n첫 달 → 마지막 달 등급 이동 (행 합 100%)")
tab = pd.crosstab(T.map({True: "외환노출", False: "비노출"}),
                  move.map({1: "상승", 0: "그대로", -1: "하락"}), normalize="index") * 100
print(tab.reindex(columns=["상승", "그대로", "하락"]).round(1).to_string())

# ── ③ 거래기간 ─────────────────────────────────────────────────
print(f"\n=== ③ 거래기간 SMD: {smd(firm['거래기간']):.3f} (관측 개월 수 — 결과 변수와 무관)")
