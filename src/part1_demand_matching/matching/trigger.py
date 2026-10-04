# -*- coding: utf-8 -*-
"""상품 매칭 모델 1단계: 트리거 엔진 — 지역 수출 둔화 시점과 단계(관찰/적기/정점·유지/평시)

입력(읽기 전용): C:\\test\\data\\ext_region.csv (대구·경북 exp_amt), ext_bizday.csv
출력: C:\\test\\분석결과\\matching\\trigger_stages.csv (지역 × 월)
실행: py -3.11 trigger.py

규칙 (사전 고정 — 결과를 보고 바꾸지 않는다)
  - 수출 YoY: exp_amt 단순 % YoY. 영업일수 전년차 성분을 대구·경북 pooled 기울기로 제거
    (financial_pressure_signals_adj.adjust_export_phase를 그대로 호출해 재사용)
  - 둔화 시작월 T0: 보정 YoY가 2개월 연속 음수가 된 첫째 달
  - 둔화 종료월: 보정 YoY가 2개월 연속 양수가 된 둘째 달. 종료월부터 평시다.
  - 둔화 중에는 새 T0를 만들지 않는다. 종료월부터 다시 T0 후보를 찾는다.
  - 단계: k = 이번 달 − T0. 관찰 k=0~2 / 적기 3~6 / 정점·유지 7~12 / 둔화 중 k≥13은 장기둔화 / 둔화 아니면 평시
  - 좌측절단: T0가 관측 시작월(PERIOD[0])인 구간은 Y (실제 시작은 더 이를 수 있음)

변경 이력
  - 2026-09-28: 13개월 상한을 '종료'에서 '장기둔화 단계 표시'로 변경. 상한은 근거(β 경로 h≤12)
    범위 표시용이었고 새 둔화 구간을 시작하려던 게 아니다. 긴 하락이 쪼개져 하락 중간에
    '관찰'이 생기는 문제를 발견해 수정함.
"""
import os
import sys

import numpy as np
import pandas as pd

# ---------------- 설정 ----------------
DATA_DIR = r"C:\test\data"
OUT_DIR = r"C:\test\분석결과\matching"
REGION_FILE = os.path.join(DATA_DIR, "ext_region.csv")
BIZDAY_FILE = os.path.join(DATA_DIR, "ext_bizday.csv")
FPS_ADJ_DIR = r"C:\test\code"              # financial_pressure_signals_adj.py 위치 (수정하지 않음)
REGIONS = ("대구", "경북")
PERIOD = (202301, 202512)                   # 출력 기간 (YoY 계산 가능 기간)
NEG_RUN = 2                                 # 둔화 시작: 연속 음수 개월 수
POS_RUN = 2                                 # 둔화 종료: 연속 양수 개월 수
MAX_K = 12                                  # 근거(β 경로) 범위. 둔화 중 k > MAX_K면 장기둔화 (종료 조건 아님)
STAGES = [("관찰", 0, 2), ("적기", 3, 6), ("정점·유지", 7, 12)]   # (단계, k 시작, k 끝)
STAGE_LONG = "장기둔화"
STAGE_NONE = "평시"
OUT_FILE = os.path.join(OUT_DIR, "trigger_stages.csv")

sys.path.insert(0, FPS_ADJ_DIR)
import financial_pressure_signals_adj as fa   # noqa: E402


def ym_add(ym: int, n: int) -> int:
    idx = (ym // 100) * 12 + (ym % 100 - 1) + n
    return (idx // 12) * 100 + idx % 12 + 1


def ym_diff(a: int, b: int) -> int:
    return ((a // 100) * 12 + a % 100) - ((b // 100) * 12 + b % 100)


def load_yoy() -> pd.DataFrame:
    """지역 × 월 원YoY(%)와 보정YoY(%) — 보정은 adjust_export_phase 재사용."""
    r = pd.read_csv(REGION_FILE, encoding="utf-8-sig")
    r = r[r["region"].isin(REGIONS)].sort_values(["region", "ym"]).copy()
    r["원YoY"] = r.groupby("region")["exp_amt"].pct_change(12) * 100
    r = r[(r["ym"] >= PERIOD[0]) & (r["ym"] <= PERIOD[1])]
    # adjust_export_phase가 기대하는 컬럼 이름으로 맞춰 호출 (phase는 계산에 쓰지 않지만 필수 컬럼)
    tmp = pd.DataFrame({fa.COLS["region"]: r["region"].values, fa.COLS["ym"]: r["ym"].values,
                        "exp_yoy": r["원YoY"].values, "phase": ""})
    adj, info = fa.adjust_export_phase(tmp, fa.load_bizdays(BIZDAY_FILE))
    r["보정YoY"] = adj["exp_yoy"].values
    return r.rename(columns={"region": "지역", "ym": "연월"})[["지역", "연월", "원YoY", "보정YoY"]], info


def stage_of(k):
    if k is None or pd.isna(k):
        return STAGE_NONE
    for name, lo, hi in STAGES:
        if lo <= k <= hi:
            return name
    return STAGE_LONG if k > MAX_K else STAGE_NONE


def run_rules(yoy: pd.Series, months: list[int]) -> tuple[list, list]:
    """한 지역의 월별 YoY에 규칙 적용 → (월별 T0 목록, 둔화 구간 목록[(T0, 종료월 or None)])."""
    v = list(yoy)
    n = len(v)
    t0_of = [None] * n
    episodes = []
    i = 0
    while i < n:
        # T0 후보: i부터 NEG_RUN개월 연속 음수 (마지막 달까지 확인할 자료가 있어야 함)
        if i + NEG_RUN <= n and all(pd.notna(v[j]) and v[j] < 0 for j in range(i, i + NEG_RUN)):
            t0 = i
            end = None
            j = t0 + 1
            while j < n:
                pos_run = j - POS_RUN + 1 > t0 and all(pd.notna(v[x]) and v[x] > 0 for x in range(j - POS_RUN + 1, j + 1))
                if pos_run:
                    end = j
                    break
                j += 1
            last = (end if end is not None else n)
            for x in range(t0, last):
                t0_of[x] = t0
            episodes.append((months[t0], months[end] if end is not None else None))
            i = last if end is not None else n
        else:
            i += 1
    return t0_of, episodes


def build(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list]:
    rows, eps = [], []
    for reg, g in df.groupby("지역", sort=False):
        g = g.sort_values("연월")
        months = g["연월"].tolist()
        t0_of, episodes = run_rules(g[col], months)
        for (m, t0) in zip(months, t0_of):
            k = None if t0 is None else months.index(m) - t0
            rows.append({"지역": reg, "연월": m, "둔화여부": t0 is not None,
                         "T0": months[t0] if t0 is not None else None, "k": k, "단계": stage_of(k),
                         "좌측절단": ("Y" if months[t0] == PERIOD[0] else "N") if t0 is not None else ""})
        eps += [(reg, s, e) for s, e in episodes]
    return pd.DataFrame(rows), eps


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    yoy, info = load_yoy()
    adj, eps_adj = build(yoy, "보정YoY")
    raw, eps_raw = build(yoy, "원YoY")
    out = yoy.merge(adj, on=["지역", "연월"])
    out["T0"] = out["T0"].astype("Int64")
    out["k"] = out["k"].astype("Int64")
    out[["지역", "연월", "원YoY", "보정YoY", "둔화여부", "T0", "k", "단계", "좌측절단"]].round(2).to_csv(
        OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[달력 보정] 수출 % YoY ~ 영업일수 전년차: 1일당 {info['slope']:+.3f}%p, 상관 {info['corr']:+.3f}, 관측 {info['n']}")
    print(f"저장: {OUT_FILE} ({len(out)}행)")
    print("\n[둔화 구간 — 보정 YoY]")
    for reg, s, e in eps_adj:
        dur = ym_diff(e, s) if e else ym_diff(PERIOD[1], s) + 1
        print(f"  {reg}: T0 {s} → 종료 {e if e else '진행 중(2025-12까지)'} | 둔화 {dur}개월"
              f"{' | 좌측절단' if s == PERIOD[0] else ''}")
    print("\n[단계별 월 수 — 보정 YoY]")
    print(pd.crosstab(out["단계"], out["지역"], margins=True, margins_name="합계").to_string())
    print("\n[참고: 보정 전 YoY로 같은 규칙]")
    for reg, s, e in eps_raw:
        print(f"  {reg}: T0 {s} → 종료 {e if e else '진행 중(2025-12까지)'}{' | 좌측절단' if s == PERIOD[0] else ''}")
    cmp = out[["지역", "연월", "원YoY", "보정YoY", "단계"]].merge(
        raw[["지역", "연월", "단계"]].rename(columns={"단계": "단계_보정전"}), on=["지역", "연월"])
    diff = cmp[cmp["단계"] != cmp["단계_보정전"]]
    print(f"\n[보정 전후 단계가 다른 달: {len(diff)}개]")
    print(diff.round(2).to_string(index=False) if len(diff) else "  없음")
    print("\n[월별 표]")
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
