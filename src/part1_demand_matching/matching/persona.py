# -*- coding: utf-8 -*-
"""상품 매칭 모델 2단계: 세그먼트·페르소나 분류 (법인 × 월)

입력(읽기 전용): C:\\test\\data\\processed\\df_ready.csv
대상: 대구·경북 법인, 업종_대분류 '금융 및 보험업' 제외
출력 (C:\\test\\분석결과\\matching):
  persona_firm_month.csv   법인 × 월 분류 결과 (로컬 전용 — 공유·md에 넣지 않음)
  persona_snapshot_202512.csv  2025-12 세그먼트 × 페르소나 × 지역 법인 수·비중
  persona_trend.csv        월별 페르소나 비중 (노출 법인)
  persona_demand_median.csv 페르소나별 요구불 잔액 중앙값 (2025-12 및 전체 기간)
실행: py -3.11 persona.py

규칙 (사전 고정 — 결과를 보고 바꾸지 않는다)
  세그먼트 (운영용): 최근 12개월(당월 포함) 외환 수출실적 > 0 → 수출형,
                     수출 없이 수입실적 > 0 → 수입형, 둘 다 없음 → 비노출.
                     창은 달력 월 기준. 관측 시작(2023-01) 때문에 12개월이 안 차는 달은 창부족플래그=Y.
  페르소나 (노출 법인 = 수출형·수입형만) — 보유 여부 + 우선순위, 위에서부터 처음 해당하는 것:
     1) A 결제형: 할인어음 + 무역금융 잔액 > 0
     2) C 설비투자형: 시설자금대출 잔액 > 0
     3) B 차입운영형: 순수운전자금 > 0 (= 운전자금대출 − 할인어음 − 무역금융)
     4) D 현금비축형: 거치식 + 적립식 잔액 > 0
     5) E 요구불중심형: 나머지
     보조 태그 보유_A~보유_D: 주 페르소나와 별개로 각 조건 충족 여부 (Y/N).
     비노출은 페르소나·보조 태그 빈 칸.
  쏠림 점검: 노출 법인의 60% 이상이 한 페르소나에 몰려도 멈추지 않고 비율만 기록 (재조정은 v2가 마지막).
  페르소나 안정화(히스테리시스, v3): 매달 계산한 원본 판정(페르소나_원본)이 2개월 연속 같은 값일 때만
    '주 페르소나'(페르소나)를 그 값으로 바꾼다. 그전에는 이전 주 페르소나를 유지한다. 비노출로 빠지거나
    관측이 끊기면(달력월이 연속이 아니면) 즉시 초기화한다. 비노출→노출로 새로 편입되는 첫 달은 보호할
    이전 페르소나가 없으므로 지연 없이 원본 값을 바로 쓴다(노출 법인인데 페르소나가 빈 채로 matcher.py에
    넘어가는 경우를 막기 위함). 보조 태그(보유_A~D)는 안정화하지 않고 당월 값을 그대로 쓴다(상품 보유
    자체는 지연 없이 바로 반영해야 하므로).

변경 이력
  - 2026-09-28 v2: 잔액 크기 비교 → 보유 여부 + 우선순위. 여신_운전자금대출잔액이 할인어음·무역금융을
    포함하는 합계 계정이라 A가 B를 이길 수 없었고, 대출과 예금을 잔액 크기로 비교하면 규모 차이로
    대출 쪽이 항상 우세했다 (v1: 노출 법인의 65.1%가 B).
  - 2026-09-30 v3: 디벨롭 검토(outputs\\devreview\\bundle_g_persona_stability.py)에서 페르소나가 바뀐
    경우의 26.8%가 2개월 안에 원래대로 돌아오는 단기 왕복임을 확인 — 2개월 연속 조건(히스테리시스)을
    추가했다. 사용자 승인(2026-09-30) 사항.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 설정 ----------------
SRC = r"C:\test\data\processed\df_ready.csv"
OUT_DIR = r"C:\test\분석결과\matching"
REGIONS = ("대구", "경북")
EXCLUDE_INDUSTRY = ("금융 및 보험업",)
WINDOW = 12                       # 세그먼트 판단 창 (당월 포함 개월 수)
SNAPSHOT_YM = 202512
CONCENTRATION_LIMIT = 0.60        # 재검토 기준

COL = {                            # 실제 컬럼 이름 (업종 컬럼은 실제 이름 그대로 사용)
    "id": "법인ID", "ym": "기준년월", "region": "사업장_시도",
    "ind_l": "업종_대분류", "ind_m": "업종_중분류",
    "exp": "외환_수출실적금액", "imp": "외환_수입실적금액",
    "demand": "요구불예금잔액",
    "wc_loan": "여신_운전자금대출잔액", "fac_loan": "여신_시설자금대출잔액",
    "fixed": "거치식예금잔액", "install": "적립식예금잔액",
    "bill": "운전_할인어음잔액", "trade": "운전_무역금융잔액",
}
SEG_EXPORT, SEG_IMPORT, SEG_NONE = "수출형", "수입형", "비노출"   # 우선 규칙: 값 고정
SEG_COL = "세그먼트"
# 페르소나 판정 순서 (위에서부터 처음 해당하는 것). 조건은 add_persona의 보유 플래그 이름으로 참조.
PERSONA_ORDER = [("A 결제형", "A"), ("C 설비투자형", "C"), ("B 차입운영형", "B"), ("D 현금비축형", "D")]
PERSONA_ZERO = "E 요구불중심형"
HOLD_THRESHOLD = 0                 # 잔액 > 이 값이면 '보유'
CONFIRM_MONTHS = 2                  # 페르소나 안정화: 새 값이 이 개월 수만큼 연속돼야 주 페르소나로 확정


def midx(ym):
    return (ym // 100) * 12 + ym % 100


def load() -> pd.DataFrame:
    head = pd.read_csv(SRC, nrows=0, encoding="utf-8-sig").columns
    missing = [f"{k}:{v}" for k, v in COL.items() if v not in head]
    if missing:
        print("[중단] 없는 컬럼:", missing)
        sys.exit(1)
    df = pd.read_csv(SRC, usecols=list(COL.values()), encoding="utf-8-sig")
    df = df[df[COL["region"]].isin(REGIONS) & ~df[COL["ind_l"]].isin(EXCLUDE_INDUSTRY)].copy()
    df["_m"] = midx(df[COL["ym"]])
    return df


def add_segment(df: pd.DataFrame) -> pd.DataFrame:
    """달력 월 기준 최근 WINDOW개월(당월 포함) 수출·수입 실적 유무."""
    m0, m1 = df["_m"].min(), df["_m"].max()
    months = np.arange(m0, m1 + 1)
    out = []
    for key, flag_col in [("exp", "_has_exp"), ("imp", "_has_imp")]:
        pos = df[df[COL[key]].fillna(0) > 0][[COL["id"], "_m"]].drop_duplicates()
        pos["_v"] = 1
        wide = pos.pivot(index=COL["id"], columns="_m", values="_v").reindex(columns=months).fillna(0)
        roll = wide.T.rolling(WINDOW, min_periods=1).max().T          # 창 안에 실적이 한 번이라도 있으면 1
        s = roll.stack().rename(flag_col).reset_index().rename(columns={"level_1": "_m"})
        out.append(s)
    df = df.merge(out[0], on=[COL["id"], "_m"], how="left").merge(out[1], on=[COL["id"], "_m"], how="left")
    df[["_has_exp", "_has_imp"]] = df[["_has_exp", "_has_imp"]].fillna(0)
    df[SEG_COL] = np.where(df["_has_exp"] > 0, SEG_EXPORT, np.where(df["_has_imp"] > 0, SEG_IMPORT, SEG_NONE))
    df["창부족플래그"] = np.where(df["_m"] - m0 + 1 < WINDOW, "Y", "N")
    return df


def add_persona(df: pd.DataFrame) -> pd.DataFrame:
    v = {k: df[COL[k]].fillna(0) for k in ("bill", "trade", "wc_loan", "fac_loan", "fixed", "install")}
    hold = {
        "A": (v["bill"] + v["trade"]) > HOLD_THRESHOLD,
        "C": v["fac_loan"] > HOLD_THRESHOLD,
        "B": (v["wc_loan"] - v["bill"] - v["trade"]) > HOLD_THRESHOLD,     # 순수운전자금
        "D": (v["fixed"] + v["install"]) > HOLD_THRESHOLD,
    }
    exposed = df[SEG_COL].isin([SEG_EXPORT, SEG_IMPORT])
    persona = pd.Series(PERSONA_ZERO, index=df.index)
    decided = pd.Series(False, index=df.index)
    for name, key in PERSONA_ORDER:
        hit = hold[key] & ~decided
        persona[hit] = name
        decided |= hit
    df["페르소나"] = np.where(exposed, persona, "")
    for key in ("A", "B", "C", "D"):
        df[f"보유_{key}"] = np.where(exposed, np.where(hold[key], "Y", "N"), "")
    return df


def stabilize_persona(df: pd.DataFrame, confirm=CONFIRM_MONTHS) -> pd.DataFrame:
    """페르소나 안정화(히스테리시스, v3). 원본 판정을 '페르소나_원본'으로 보존하고, '페르소나'를
    새 값이 confirm개월 연속일 때만 바뀌는 안정화된 값으로 덮어쓴다. 법인이 바뀌거나 달력월이
    연속이 아니면(관측 끊김·비노출 복귀 등) 그 시점의 원본 값으로 다시 시작한다."""
    df = df.sort_values([COL["id"], "_m"]).reset_index(drop=True)
    raw = df["페르소나"].to_numpy()
    m = df["_m"].to_numpy()
    fid = df[COL["id"]].to_numpy()
    out = np.empty(len(df), dtype=object)
    stable, candidate, streak = "", "", 0
    prev_fid, prev_m = None, None
    for i in range(len(df)):
        new_run = (fid[i] != prev_fid) or (prev_m is not None and m[i] != prev_m + 1)
        if new_run:
            stable, candidate, streak = raw[i], "", 0
        elif raw[i] == "":
            stable, candidate, streak = "", "", 0          # 비노출: 지연 없이 바로 반영
        elif stable == "":
            stable, candidate, streak = raw[i], "", 0       # 비노출→노출로 새로 편입: 보호할 이전 페르소나가 없어 즉시 반영
        elif raw[i] == stable:
            candidate, streak = "", 0
        else:
            if raw[i] == candidate:
                streak += 1
            else:
                candidate, streak = raw[i], 1
            if streak >= confirm:
                stable, candidate, streak = candidate, "", 0
        out[i] = stable
        prev_fid, prev_m = fid[i], m[i]
    df["페르소나_원본"] = raw
    df["페르소나"] = out
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = stabilize_persona(add_persona(add_segment(load())))
    firm = df.rename(columns={COL["id"]: "법인ID", COL["ym"]: "연월", COL["region"]: "지역",
                              COL["demand"]: "요구불잔액"})
    keep = ["법인ID", "연월", "지역", "업종_대분류", "업종_중분류", SEG_COL, "페르소나", "페르소나_원본",
            "보유_A", "보유_B", "보유_C", "보유_D", "요구불잔액", "창부족플래그"]
    chg_rate = (firm["페르소나"] != firm["페르소나_원본"]).mean()
    print(f"[안정화] 원본과 안정화 값이 다른 법인×월: {chg_rate:.1%} (원본이 아직 {CONFIRM_MONTHS}개월 연속을 못 채워 이전 값을 유지 중)")
    firm[keep].sort_values(["법인ID", "연월"]).to_csv(os.path.join(OUT_DIR, "persona_firm_month.csv"),
                                                    index=False, encoding="utf-8-sig")
    print(f"[대상] 법인 {firm['법인ID'].nunique():,}곳, 법인×월 {len(firm):,}행, "
          f"{firm['연월'].min()}~{firm['연월'].max()}")

    snap = firm[firm["연월"] == SNAPSHOT_YM]
    tab = (snap.assign(페르소나=snap["페르소나"].replace("", "(비노출)"))
           .groupby([SEG_COL, "페르소나", "지역"])["법인ID"].nunique().rename("법인수").reset_index())
    tab["비중(%)"] = (tab["법인수"] / tab.groupby("지역")["법인수"].transform("sum") * 100).round(1)
    tab.to_csv(os.path.join(OUT_DIR, "persona_snapshot_202512.csv"), index=False, encoding="utf-8-sig")

    exp_ = firm[firm["페르소나"] != ""]
    trend = (exp_.groupby(["연월", "페르소나"])["법인ID"].nunique().unstack(fill_value=0))
    trend_share = (trend.div(trend.sum(axis=1), axis=0) * 100).round(1)
    trend_share.to_csv(os.path.join(OUT_DIR, "persona_trend.csv"), encoding="utf-8-sig")

    med = pd.DataFrame({
        f"{SNAPSHOT_YM} 중앙값": snap[snap["페르소나"] != ""].groupby("페르소나")["요구불잔액"].median(),
        f"{SNAPSHOT_YM} 법인수": snap[snap["페르소나"] != ""].groupby("페르소나")["법인ID"].nunique(),
        "전체기간 중앙값": exp_.groupby("페르소나")["요구불잔액"].median(),
    })
    med.loc["(비노출)", f"{SNAPSHOT_YM} 중앙값"] = snap[snap[SEG_COL] == SEG_NONE]["요구불잔액"].median()
    med.loc["(비노출)", f"{SNAPSHOT_YM} 법인수"] = snap[snap[SEG_COL] == SEG_NONE]["법인ID"].nunique()
    med.loc["(비노출)", "전체기간 중앙값"] = firm[firm[SEG_COL] == SEG_NONE]["요구불잔액"].median()
    med.to_csv(os.path.join(OUT_DIR, "persona_demand_median.csv"), encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(f"\n[{SNAPSHOT_YM} 세그먼트 × 지역 법인 수]")
    print(pd.crosstab(snap[SEG_COL], snap["지역"], margins=True, margins_name="합계").to_string())
    print(f"\n[{SNAPSHOT_YM} 세그먼트 × 페르소나 × 지역]")
    print(tab.to_string(index=False))
    print("\n[월별 페르소나 비중(%) — 노출 법인]")
    print(trend_share.to_string())
    print("\n[페르소나별 요구불 잔액 중앙값]")
    print(med.round(0).to_string())

    # 재검토 기준
    snap_share = (snap[snap["페르소나"] != ""]["페르소나"].value_counts(normalize=True))
    max_month = trend_share.max(axis=1)
    print(f"\n[재검토 기준 {CONCENTRATION_LIMIT:.0%}] {SNAPSHOT_YM} 최대 페르소나: "
          f"{snap_share.idxmax()} {snap_share.max():.1%} | 기준 초과 월 수: {(max_month >= CONCENTRATION_LIMIT * 100).sum()} / {len(max_month)}")
    if snap_share.max() >= CONCENTRATION_LIMIT or (max_month >= CONCENTRATION_LIMIT * 100).any():
        print("[기록] 노출 법인이 한 페르소나에 60% 이상 몰림 — v2 규칙상 멈추지 않고 기록만 함")

    # 보조 태그 보유율 (2025-12 노출 법인)
    se = snap[snap["페르소나"] != ""]
    print(f"\n[{SNAPSHOT_YM} 보조 태그 보유율 — 노출 법인 {len(se)}곳]")
    print({k: f"{(se[k] == 'Y').mean():.1%}" for k in ["보유_A", "보유_B", "보유_C", "보유_D"]})
    print("보유 태그 개수 분포:", (se[["보유_A", "보유_B", "보유_C", "보유_D"]] == "Y").sum(axis=1)
          .value_counts().sort_index().to_dict())

    # 운영 정의 vs 분석 정의 노출 법인 수
    ever = firm.groupby("법인ID").apply(lambda g: bool((df.loc[g.index, COL["exp"]].fillna(0) > 0).any()
                                                      or (df.loc[g.index, COL["imp"]].fillna(0) > 0).any()))
    print(f"\n[노출 법인 정의 비교] 운영 정의({SNAPSHOT_YM}, 최근 12개월 실적): {len(se)}곳 | "
          f"분석 정의(36개월 중 1회 이상 실적): {int(ever.sum())}곳")


if __name__ == "__main__":
    main()
