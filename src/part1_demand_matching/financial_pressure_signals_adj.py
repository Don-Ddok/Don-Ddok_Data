# -*- coding: utf-8 -*-
"""
자금압박 동행신호 디벨롭 (아이디어 1,2,3).

전제: 은행 패널 df에 아래 컬럼이 이미 있다고 가정.
  법인ID, 기준년월(YYYYMM, int 또는 str), 사업장_시도(대구/경북), 업종_중분류,
  요구불예금잔액, 운전_할인어음잔액, exposed(0/1), exp_yoy(지역 수출 YoY %),
  phase(증가/하락/깊은하락)
exposed/phase가 없을 때의 폴백 빌더도 제공하지만, exp_yoy(지역 수출 YoY)는
외부 K-stat 데이터 병합이 필요해 이 스크립트에서 새로 만들지 않는다
(analyze_im_bank.py의 STEP15/16 과정을 먼저 거쳐야 함).

원본 행/법인ID는 출력하지 않는다. 모든 print는 집계표만 내보낸다.
저장(to_csv 등)은 하지 않는다 — 필요하면 함수가 반환하는 DataFrame을 호출자가 직접 다룰 것.
"""

import re
import calendar
import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

# =====================================================================
# 설정 (임의 선택한 값은 전부 여기 상수로 모아두고 근거를 주석에 남김)
# =====================================================================

COLS = {
    "id": "법인ID",
    "ym": "기준년월",
    "region": "사업장_시도",
    "industry": "업종_중분류",
    "deposit": "요구불예금잔액",
    "bill": "운전_할인어음잔액",
    "working_capital": "여신_운전자금대출잔액",  # 있으면 사용, 없으면 자동 스킵
    "exposed": "exposed",
    "exp_yoy": "exp_yoy",
    "phase": "phase",
}

DOWN_PHASES = {"하락", "깊은하락"}

# 아이디어1: 전월대비 "변화율" 대신 ln(x+1) 차분을 쓴다.
# 이유: 요구불/할인어음 둘 다 0값이 흔해서(프로젝트 초반 STEP3에서 확인) 퍼센트 변화율은
# 분모가 0이 되는 문제가 있음. 이 프로젝트 전체(STEP4, STEP8 등)에서 일관되게 쓴
# ln(x+1) 차분 방식을 그대로 재사용해 정의를 통일한다.
#
# [개선] 실제 데이터 in/out-sample 검증에서 단월(lag=1) 차분이 국면별 방향을 뒤집는
# 현상을 발견 (깊은하락 요구불변화: in=+0.059 vs out=-0.151). 시차 진단 결과 요구불
# 감소가 수출충격과 동시가 아니라 누적된 뒤 나타나는 것으로 보여, 3개월 누적차분
# (lag=3)으로 바꾸니 in/out 둘 다 음수로 방향이 일관됐다. 이 근거로 기본값을 3으로
# 바꾸되, 원래(lag=1) 정의도 비교용으로 그대로 선택할 수 있게 남겨둔다.
DEFAULT_PRESSURE_LAG = 3      # 기본값(개선). 1=단월(원래), 3=3개월 누적(개선)
INSAMPLE_FRACTION = 0.5  # 기간을 반으로 나누는 기준(시간순 중앙값)

# 보완2: 압박지수 상위 법인의 규모 쏠림 진단용. 상위 몇 %를 "상위군"으로 볼지.
TOP_PRESSURE_QUANTILE = 0.9
# 상위군 규모(중앙값)가 전체 대비 이 배율 밖이면 "규모 쏠림" 경고.
SCALE_SKEW_WARN_RATIO = 2.0

# 보완3: in/out 두 구간의 하락(하락+깊은하락) 비중 차이가 이 값(퍼센트포인트) 이상이면 경고.
PHASE_IMBALANCE_WARN_PP = 20.0

# 아이디어2: 공표 시점은 전부 "가정"이다. 실제 공표 일정이 확인되면 이 값만 바꾸면 된다.
BANK_SIGNAL_LAG_DAYS = 5       # 월말 마감 후 은행 내부적으로 관측 가능해지기까지 가정한 일수
EXPORT_STAT_PUBLISH_DAY = 15   # 관세청 수출입 통계: 익월 15일 공표 가정
# 광공업생산지수는 "익월 말" 공표로 가정 (아래 코드에서 해당 월의 마지막 날 계산)
PRESSURE_SIGNAL_THRESHOLD = 0.0  # region-month 평균 압박지수가 이 값을 넘으면 "신호 발생"
# 0.0 = in-sample로 표준화했을 때의 평균(중립점). 더 보수적으로 잡고 싶으면 양수로 올릴 것.
# 보완1: 하락국면 중 은행신호가 실제로 포착한 비율(recall)이 이 미만이면 리드타임 결과에
# 신뢰 주의를 표시. 절반 미만이면 "포착한 경우만 좋다"는 주장의 대표성이 떨어진다고 판단.
CAPTURE_RATE_WARN_THRESHOLD = 0.5

# 아이디어3: 처치 법인(해당 셀에서 down=1 & exposed=1인 법인) 수가 이 값 미만이면
# 신뢰 주의 플래그. STEP20에서 쓴 기준(40)과 통일.
MIN_TREATED_FOR_CONFIDENCE = 40
LP_HORIZONS = range(0, 7)  # h = 0..6

ACCOUNTS_FOR_DICTIONARY = {
    "요구불": COLS["deposit"],
    "할인어음": COLS["bill"],
    "운전자금": COLS["working_capital"],
}


# =====================================================================
# 공용 헬퍼
# =====================================================================

def _normalize_ym(series: pd.Series) -> pd.Series:
    def parse_one(v):
        if pd.isna(v):
            return np.nan
        digits = re.sub(r"\D", "", str(v))
        if len(digits) != 6:
            return np.nan
        y, m = int(digits[:4]), int(digits[4:6])
        return y * 100 + m if 1 <= m <= 12 else np.nan
    return series.map(parse_one)


def _add_midx(df: pd.DataFrame, ym_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_ym_"] = _normalize_ym(df[ym_col])
    df["_midx_"] = (df["_ym_"] // 100) * 12 + (df["_ym_"] % 100)
    return df


def merge_exact_lag(df: pd.DataFrame, id_col: str, midx_col: str,
                     value_col: str, lag: int, new_col: str) -> np.ndarray:
    """각 (id, t) 행에 정확히 t-lag 시점의 value_col 값을 붙인다 (lag>0: 과거)."""
    src = df[[id_col, midx_col, value_col]].copy()
    src[midx_col] = src[midx_col] + lag
    src = src.rename(columns={value_col: new_col})
    merged = df[[id_col, midx_col]].merge(src, on=[id_col, midx_col], how="left")
    return merged[new_col].values


def merge_exact_lead(df: pd.DataFrame, id_col: str, midx_col: str,
                      value_col: str, h: int, new_col: str) -> np.ndarray:
    """각 (id, t) 행에 정확히 t+h 시점의 value_col 값을 붙인다 (h>=0: 미래)."""
    src = df[[id_col, midx_col, value_col]].copy()
    src[midx_col] = src[midx_col] - h
    src = src.rename(columns={value_col: new_col})
    merged = df[[id_col, midx_col]].merge(src, on=[id_col, midx_col], how="left")
    return merged[new_col].values


def ensure_exposed_column(df: pd.DataFrame, id_col: str,
                           export_col: str, import_col: str,
                           exposed_col: str = "exposed") -> pd.DataFrame:
    """exposed 컬럼이 없으면: 법인 단위 (수출실적 or 수입실적) 최댓값>0 으로 생성."""
    if exposed_col in df.columns:
        return df
    firm_level = df.groupby(id_col)[[export_col, import_col]].max()
    exposed_map = ((firm_level[export_col].fillna(0) > 0) |
                    (firm_level[import_col].fillna(0) > 0)).astype(int)
    df = df.copy()
    df[exposed_col] = df[id_col].map(exposed_map)
    print(f"[생성] '{exposed_col}' 컬럼 없어 자동 생성 (법인수 {exposed_map.sum()}/{len(exposed_map)} 노출).")
    return df


def ensure_phase_column(df: pd.DataFrame, exp_yoy_col: str, region_col: str,
                         deep_decline_cutoff: dict, phase_col: str = "phase") -> pd.DataFrame:
    """phase 컬럼이 없으면 exp_yoy 기준으로 생성.
    증가: exp_yoy>=0, 하락: 0>exp_yoy>=지역별 컷오프, 깊은하락: exp_yoy<지역별 컷오프."""
    if phase_col in df.columns:
        return df
    if exp_yoy_col not in df.columns:
        raise KeyError(f"'{exp_yoy_col}'가 없어 phase를 생성할 수 없습니다. "
                        f"지역 수출 YoY(exp_yoy)를 먼저 병합하세요 (analyze_im_bank.py STEP15/16 참고).")

    def classify(row):
        yoy = row[exp_yoy_col]
        cutoff = deep_decline_cutoff.get(row[region_col])
        if pd.isna(yoy) or cutoff is None:
            return np.nan
        if yoy >= 0:
            return "증가"
        return "깊은하락" if yoy < cutoff else "하락"

    df = df.copy()
    df[phase_col] = df.apply(classify, axis=1)
    print(f"[생성] '{phase_col}' 컬럼 없어 exp_yoy 기준으로 자동 생성. "
          f"컷오프={deep_decline_cutoff}")
    return df


# =====================================================================
# 아이디어 1: 자금압박지수
# =====================================================================

def build_pressure_index(df: pd.DataFrame, lag: int = DEFAULT_PRESSURE_LAG,
                          region: "str | None" = None) -> dict:
    """법인별 월별 자금압박지수 생성 + in/out-sample 검증.

    압박지수 = -(요구불 ln(x+1) lag개월 누적차분의 z) + (할인어음 ln(x+1) lag개월 누적차분의 z)
    표준화 평균/표준편차는 전반부(in-sample)로만 계산해 후반부(out-sample)에 적용
    (과적합/데이터누수 방지).

    lag: 차분 시차(개월). 기본값 DEFAULT_PRESSURE_LAG(=3, 시차진단 근거로 채택한 개선값).
         lag=1을 넘기면 원래(단월) 정의로 계산 - 두 정의를 비교하려면
         compare_lag_definitions()를 쓸 것.
    region: 지정하면 해당 지역(사업장_시도)만 필터링해 그 지역 데이터로 표준화까지
            전부 다시 계산한다(대구/경북의 수출 변동성 스케일이 달라 지역별로 따로
            보는 게 근거 있는 개선이라는 진단에 따른 옵션). None이면 전체 통합.

    Returns: {
        "firm_month_index": 법인×월 압박지수 전체 (내부용, 원본 그대로는 출력 안 함),
        "phase_summary": 국면×기간(in/out) 평균 지수 표 (출력됨),
    }
    """
    print("=" * 70)
    print(f"아이디어 1. 자금압박지수 (lag={lag}개월, region={region or '전체통합'})")
    print("=" * 70)

    id_col, ym_col = COLS["id"], COLS["ym"]
    dep_col, bill_col = COLS["deposit"], COLS["bill"]
    phase_col, region_col = COLS["phase"], COLS["region"]

    if region is not None:
        df = df[df[region_col] == region].copy()
        print(f"[진행] region='{region}' 필터링 적용, 행수={len(df):,}")
        if df.empty:
            raise ValueError(f"region='{region}'에 해당하는 행이 없습니다.")

    work = _add_midx(df, ym_col)
    work = work.sort_values([id_col, "_midx_"]).reset_index(drop=True)

    for role, col in [("요구불", dep_col), ("할인어음", bill_col)]:
        if col not in work.columns:
            raise KeyError(f"{role} 컬럼('{col}')이 없습니다.")

    # ln(x+1)의 lag개월 누적차분 (정확히 lag개월 전 관측이 있는 쌍에서만)
    for col, chg_name in [(dep_col, "_dep_chg_"), (bill_col, "_bill_chg_")]:
        log_col = f"_log_{chg_name}"
        work[log_col] = np.log1p(work[col].clip(lower=0))
        prev_log = merge_exact_lag(work, id_col, "_midx_", log_col, lag=lag, new_col=f"_prev_{log_col}")
        work[chg_name] = work[log_col].values - prev_log

    valid = work.dropna(subset=["_dep_chg_", "_bill_chg_"]).copy()
    if valid.empty:
        raise ValueError("월간 유효쌍이 하나도 없습니다. 데이터 기간/정렬을 확인하세요.")

    # 기간을 시간순 중앙값으로 반 분할
    unique_midx = np.sort(valid["_midx_"].unique())
    median_midx = unique_midx[len(unique_midx) // 2]
    valid["_period_"] = np.where(valid["_midx_"] <= median_midx, "in-sample(전반부)", "out-sample(후반부)")
    print(f"[진행] 기간 분할: in-sample <= {median_midx} midx, out-sample > {median_midx} midx "
          f"(관측 유니크 개월수={len(unique_midx)})")

    # --- 보완3: in/out 분할의 국면(phase) 균형 점검 ---
    # 시간순 중앙값으로 자르면, 특정 구간(예: 2024년 하반기)에 하락국면이 몰려있을 경우
    # in/out 두 표본이 애초에 다른 경기국면을 대표하게 되어 아웃샘플 검증이 왜곡될 수 있음.
    # region-month 단위(법인 수에 안 휘둘리도록)로 phase 비중을 비교한다.
    if phase_col in valid.columns:
        region_month_phase = valid.drop_duplicates(subset=[COLS["region"], "_midx_", "_period_"])[
            ["_period_", phase_col]
        ]
        phase_ratio = (pd.crosstab(region_month_phase["_period_"], region_month_phase[phase_col], normalize="index") * 100).round(1)
        print("\n[보완3: in/out 기간별 국면(phase) 비중(%), region-month 단위]")
        print(phase_ratio.to_string())

        down_cols = [c for c in phase_ratio.columns if c in DOWN_PHASES]
        if down_cols and all(p in phase_ratio.index for p in ["in-sample(전반부)", "out-sample(후반부)"]):
            down_share = phase_ratio[down_cols].sum(axis=1)
            gap_pp = abs(down_share["in-sample(전반부)"] - down_share["out-sample(후반부)"])
            print(f"하락계열(하락+깊은하락) 비중 격차: in={down_share['in-sample(전반부)']:.1f}%, "
                  f"out={down_share['out-sample(후반부)']:.1f}%, 격차={gap_pp:.1f}%p")
            if gap_pp >= PHASE_IMBALANCE_WARN_PP:
                print(f"[경고] 격차가 {PHASE_IMBALANCE_WARN_PP}%p 이상 - 시간순 분할이 국면 불균형을 "
                      f"유발했을 수 있음. 아래 out-sample 검증 결과는 '기간 차이'와 '국면 차이'가 "
                      f"섞여 있을 수 있으니 주의해서 해석할 것.")
                print("  (대안 제안, 미적용: 시간순 대신 phase별로 각각 절반씩 in/out에 배분하는 "
                      "층화 분할을 쓰면 기간 효과와 국면 효과를 분리할 수 있음.)")
    # --- 보완3 끝 ---

    insample = valid[valid["_period_"] == "in-sample(전반부)"]
    dep_mean, dep_std = insample["_dep_chg_"].mean(), insample["_dep_chg_"].std()
    bill_mean, bill_std = insample["_bill_chg_"].mean(), insample["_bill_chg_"].std()
    print(f"[진행] in-sample 표준화 기준: 요구불차분(mean={dep_mean:.4f}, std={dep_std:.4f}), "
          f"할인어음차분(mean={bill_mean:.4f}, std={bill_std:.4f})")

    valid["_dep_z_"] = (valid["_dep_chg_"] - dep_mean) / dep_std
    valid["_bill_z_"] = (valid["_bill_chg_"] - bill_mean) / bill_std
    valid["pressure_index"] = -valid["_dep_z_"] + valid["_bill_z_"]

    firm_month_index = valid[[id_col, ym_col, "_midx_", COLS["region"], phase_col, "_period_", "pressure_index"]].copy()

    # --- 보완2: 압박지수 상위 법인의 규모 쏠림 진단 ---
    # ln(x+1) 차분으로 어느 정도 규모 중립화는 됐지만, 전체 풀 표준편차로 z-score를
    # 내면 변동성 자체가 큰 법인(꼭 대기업/소기업이라는 보장은 없음)이 지수를 지배할
    # 수 있다. 법인별 평균 압박지수 상위 10%(TOP_PRESSURE_QUANTILE)의 "규모"(요구불
    # 잔액 평균 레벨로 근사)가 전체와 크게 다른지 점검한다.
    firm_level = valid.groupby(id_col).agg(
        firm_mean_index=("pressure_index", "mean"),
        firm_scale=(dep_col, "mean"),
    )
    top_threshold = firm_level["firm_mean_index"].quantile(TOP_PRESSURE_QUANTILE)
    top_firms = firm_level[firm_level["firm_mean_index"] >= top_threshold]
    rest_firms = firm_level[firm_level["firm_mean_index"] < top_threshold]

    scale_desc = pd.DataFrame({
        f"상위{int((1 - TOP_PRESSURE_QUANTILE) * 100)}%(n={len(top_firms)})": top_firms["firm_scale"].describe(),
        f"나머지(n={len(rest_firms)})": rest_firms["firm_scale"].describe(),
    })
    print(f"\n[보완2: 압박지수 상위 {int((1 - TOP_PRESSURE_QUANTILE) * 100)}% 법인의 요구불잔액 규모(평균레벨) 분포]")
    print(scale_desc.to_string())

    top_median = top_firms["firm_scale"].median()
    rest_median = rest_firms["firm_scale"].median()
    if pd.notna(top_median) and pd.notna(rest_median) and rest_median > 0:
        ratio = top_median / rest_median
        print(f"상위군 중앙값 / 나머지 중앙값 = {ratio:.2f}배")
        if ratio >= SCALE_SKEW_WARN_RATIO or ratio <= 1 / SCALE_SKEW_WARN_RATIO:
            print(f"[경고] 규모 배율이 {SCALE_SKEW_WARN_RATIO}배 기준을 벗어남 - 압박지수 상위군이 "
                  f"특정 규모대(큰 법인 또는 작은 법인)에 쏠려 있을 가능성이 있음. 규모 효과를 "
                  f"의심해볼 것.")
            print("  (대안 제안, 미적용: ln(x+1) 차분을 전체 풀 표준편차 대신 법인별 자기 "
                  "표준편차로 나누면(법인 내 상대적 변동만 봄) 규모 효과를 더 강하게 중화할 수 "
                  "있음 - 다만 관측치 적은 법인은 자기 표준편차 추정이 불안정해지는 trade-off 있음.)")
        else:
            print("규모 쏠림 뚜렷하지 않음.")
    # --- 보완2 끝 ---

    if phase_col in valid.columns:
        summary = (valid.groupby(["_period_", phase_col])["pressure_index"]
                   .agg(["mean", "std", "count"]).reset_index()
                   .rename(columns={"mean": "평균압박지수", "std": "표준편차", "count": "n"}))
        print("\n[국면 × 기간별 평균 압박지수]")
        print(summary.to_string(index=False))

        pivot = summary.pivot(index=phase_col, columns="_period_", values="평균압박지수")
        both_periods_ok = all(p in pivot.columns for p in ["in-sample(전반부)", "out-sample(후반부)"])
        if both_periods_ok and DOWN_PHASES & set(pivot.index):
            down_rows = [p for p in DOWN_PHASES if p in pivot.index]
            up_val_in = pivot.loc["증가", "in-sample(전반부)"] if "증가" in pivot.index else np.nan
            up_val_out = pivot.loc["증가", "out-sample(후반부)"] if "증가" in pivot.index else np.nan
            down_max_in = pivot.loc[down_rows, "in-sample(전반부)"].max()
            down_max_out = pivot.loc[down_rows, "out-sample(후반부)"].max()
            holds_in = pd.notna(down_max_in) and pd.notna(up_val_in) and down_max_in > up_val_in
            holds_out = pd.notna(down_max_out) and pd.notna(up_val_out) and down_max_out > up_val_out
            print(f"\n[해석] 하락국면 평균지수가 증가국면보다 높은가? "
                  f"in-sample={'예' if holds_in else '아니오'}, out-sample={'예' if holds_out else '아니오'}")
            if holds_in and holds_out:
                print("  -> 아웃샘플에서도 패턴이 유지됨: 과적합이 아니라 일반화되는 신호로 보임.")
            else:
                print("  -> 아웃샘플에서 패턴이 약화/소멸: 과적합 가능성 있음, 추가 검증 필요.")

        # --- 지역별 산출 옵션: region 필터 없이 돌렸을 때도 지역축을 바로 볼 수 있게
        # (이 표는 전체 통합 표준화 기준으로 지역만 나눠본 것 - 지역별로 표준화 자체를
        # 새로 하고 싶으면 region= 인자로 함수를 따로 호출할 것)
        if region is None and region_col in valid.columns:
            region_summary = (valid.groupby([region_col, "_period_", phase_col])["pressure_index"]
                              .agg(["mean", "count"]).reset_index()
                              .rename(columns={"mean": "평균압박지수", "count": "n"}))
            print("\n[지역별 상세 (전체 통합 표준화 기준) - region= 인자로 지역 단독 재계산과 비교할 것]")
            print(region_summary.to_string(index=False))
    else:
        summary = pd.DataFrame()
        print("[경고] phase 컬럼이 없어 국면별 요약을 생략합니다.")

    return {"firm_month_index": firm_month_index, "phase_summary": summary, "lag": lag, "region": region}


def compare_lag_definitions(df: pd.DataFrame, region: "str | None" = None,
                             lags: tuple = (1, DEFAULT_PRESSURE_LAG)) -> pd.DataFrame:
    """lag=1(원래, 단월) vs lag=3(개선, 3개월 누적) 압박지수를 국면×기간별로 나란히 비교.
    build_pressure_index를 lag별로 그대로 재사용해서 돌리고 phase_summary만 합친다."""
    print("\n" + "#" * 70)
    print(f"[lag 비교] region={region or '전체통합'}, lags={lags}")
    print("#" * 70)

    phase_col = COLS["phase"]
    merged = None
    for lag in lags:
        result = build_pressure_index(df, lag=lag, region=region)
        s = result["phase_summary"]
        if s.empty:
            continue
        s = s[["_period_", phase_col, "평균압박지수"]].rename(columns={"평균압박지수": f"lag={lag}"})
        merged = s if merged is None else merged.merge(s, on=["_period_", phase_col], how="outer")

    if merged is None or merged.empty:
        print("[경고] 비교할 phase_summary가 없습니다.")
        return pd.DataFrame()

    merged = merged.sort_values(["_period_", phase_col]).reset_index(drop=True)
    print(f"\n[lag={lags[0]} vs lag={lags[-1]} 국면×기간별 평균압박지수 비교, region={region or '전체통합'}]")
    print(merged.to_string(index=False))

    # 방향 일관성(하락계열이 증가보다 높은가) 요약: lag별로 in/out 둘 다 성립하는지
    for lag in lags:
        col = f"lag={lag}"
        if col not in merged.columns:
            continue
        pivot = merged.pivot(index=phase_col, columns="_period_", values=col)
        if "증가" not in pivot.index or not (DOWN_PHASES & set(pivot.index)):
            continue
        down_rows = [p for p in DOWN_PHASES if p in pivot.index]
        holds = {}
        for period in ["in-sample(전반부)", "out-sample(후반부)"]:
            if period not in pivot.columns:
                continue
            holds[period] = pd.notna(pivot.loc[down_rows, period].max()) and \
                pivot.loc[down_rows, period].max() > pivot.loc["증가", period]
        consistent = all(holds.values()) and len(holds) == 2
        print(f"  lag={lag}: in/out 방향 일관 = {'예' if consistent else '아니오'} "
              f"(세부: {holds})")

    return merged


# =====================================================================
# 아이디어 2: 리드타임 정량화
# =====================================================================

def _month_end(year: int, month: int) -> dt.date:
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, last_day)


def _next_month(year: int, month: int) -> tuple:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def lead_time_analysis(firm_month_index: pd.DataFrame) -> pd.DataFrame:
    """region-month 단위로 압박지수를 집계해, 실제 하락국면(phase)과 동시에
    압박지수도 임계를 넘는 '동시 포착' 에피소드에 대해, 은행신호 관측가능일과
    공식 통계 공표(가정)일의 차이를 주 단위로 계산.

    주의: 이는 '관측 가능 시점'의 비교이지 '선행성' 주장이 아니다 — 같은 달을
    가리키는 신호를, 은행은 월말 직후 볼 수 있고 공식 통계는 몇 주 뒤에 공표된다는
    데이터 접근성 격차를 정량화한 것.
    """
    print("\n" + "=" * 70)
    print("아이디어 2. 리드타임 정량화 (은행신호 관측가능일 vs 공식통계 공표일, 가정 기반)")
    print("=" * 70)

    region_col, phase_col = COLS["region"], COLS["phase"]

    region_month = (firm_month_index.groupby([region_col, "_midx_"])
                    .agg(ym=(COLS["ym"], "first"),
                         phase=(phase_col, "first"),
                         mean_pressure=("pressure_index", "mean"))
                    .reset_index())

    down_all = region_month[region_month["phase"].isin(DOWN_PHASES)].copy()
    episodes = down_all[down_all["mean_pressure"] > PRESSURE_SIGNAL_THRESHOLD].copy()

    # --- 보완1: 선택편향 점검 - "포착한 경우만" 평균내면 리드타임이 실제보다 좋게 나옴.
    # 전체 하락국면을 분모로 놓고 포착률(recall)을 반드시 같이 본다.
    n_down_total = len(down_all)
    n_captured = len(episodes)
    n_missed = n_down_total - n_captured
    capture_rate = n_captured / n_down_total if n_down_total else np.nan

    capture_table = pd.DataFrame([{
        "전체 하락국면 수(a)": n_down_total,
        "은행신호 포착 수(b, 동시포착)": n_captured,
        "놓친 수(c=a-b)": n_missed,
        "포착률(b/a)": round(capture_rate, 4) if pd.notna(capture_rate) else np.nan,
    }])
    print("\n[보완1: 하락국면 포착률 (선택편향 점검)]")
    print(capture_table.to_string(index=False))

    if pd.notna(capture_rate) and capture_rate < CAPTURE_RATE_WARN_THRESHOLD:
        print(f"[경고] 포착률이 {CAPTURE_RATE_WARN_THRESHOLD*100:.0f}% 미만 - 아래 리드타임은 "
              f"'은행이 실제로 맞춘 일부 사례'에서만 계산된 것이며, 전체 하락국면을 대표하지 "
              f"않는다. 리드타임 수치가 좋게 나와도 신뢰 주의.")
    # --- 보완1 끝 ---

    if episodes.empty:
        print("[경고] 동시포착 에피소드가 없어 리드타임을 계산할 수 없습니다.")
        return pd.DataFrame([{
            "포착률(b/a)": round(capture_rate, 4) if pd.notna(capture_rate) else np.nan,
            "n_episodes(=b, 포착된 것만)": 0,
            "리드타임(주)_vs_수출통계_평균": np.nan,
            "리드타임(주)_vs_수출통계_중앙값": np.nan,
            "리드타임(주)_vs_광공업생산지수_평균": np.nan,
            "리드타임(주)_vs_광공업생산지수_중앙값": np.nan,
        }])

    rows = []
    for _, r in episodes.iterrows():
        year, month = int(r["ym"]) // 100, int(r["ym"]) % 100
        bank_signal_date = _month_end(year, month) + dt.timedelta(days=BANK_SIGNAL_LAG_DAYS)

        ny, nm = _next_month(year, month)
        export_publish_date = dt.date(ny, nm, min(EXPORT_STAT_PUBLISH_DAY, calendar.monthrange(ny, nm)[1]))
        iip_publish_date = _month_end(ny, nm)

        rows.append({
            "region": r[region_col],
            "ym": int(r["ym"]),
            "lead_weeks_vs_export": (export_publish_date - bank_signal_date).days / 7,
            "lead_weeks_vs_iip": (iip_publish_date - bank_signal_date).days / 7,
        })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "포착률(b/a)": round(capture_rate, 4) if pd.notna(capture_rate) else np.nan,
        "n_episodes(=b, 포착된 것만)": len(detail),
        "리드타임(주)_vs_수출통계_평균": round(detail["lead_weeks_vs_export"].mean(), 2),
        "리드타임(주)_vs_수출통계_중앙값": round(detail["lead_weeks_vs_export"].median(), 2),
        "리드타임(주)_vs_광공업생산지수_평균": round(detail["lead_weeks_vs_iip"].mean(), 2),
        "리드타임(주)_vs_광공업생산지수_중앙값": round(detail["lead_weeks_vs_iip"].median(), 2),
    }])

    print("\n[리드타임 요약]")
    print(summary.to_string(index=False))
    print(f"\n(가정: 은행신호는 월말+{BANK_SIGNAL_LAG_DAYS}일에 관측 가능, 수출통계는 익월 {EXPORT_STAT_PUBLISH_DAY}일, "
          f"광공업생산지수는 익월 말일 공표. 실제 공표일정 확인되면 상단 상수만 교체.)")
    print(f"주의(선택편향): 위 리드타임은 전체 하락국면 {n_down_total}개 중 은행이 실제로 "
          f"포착한 {n_captured}개(포착률 {capture_rate*100:.1f}%)에 대해서만 계산된 것이다. "
          f"\"포착한 경우에 한해 평균 {summary['리드타임(주)_vs_수출통계_평균'].iloc[0]:.1f}주 "
          f"앞선다\"로 읽어야 하며, 놓친 {n_missed}개 사례는 이 평균에 반영되지 않았다.")
    print("주의(동행 vs 선행): 이는 같은 참조월(ym)에 대한 '관측 가능 시점'의 격차이지, 은행신호가 "
          "미래를 예측(선행)한다는 의미가 아님(은행/공식 통계 모두 동일한 달의 동행 신호를 담고 있음).")

    return summary


def compare_lead_time_by_scope(df: pd.DataFrame, lag: int = DEFAULT_PRESSURE_LAG,
                                region_of_interest: str = "대구") -> pd.DataFrame:
    """개선된(기본 lag=3) 압박지수로 통합 vs 지역한정(기본 대구)의 포착률·리드타임을
    나란히 비교. build_pressure_index + lead_time_analysis를 그대로 재사용."""
    print("\n" + "#" * 70)
    print(f"[포착률/리드타임 비교] lag={lag}, 통합 vs {region_of_interest} 한정")
    print("#" * 70)

    rows = []
    for label, region in [("통합", None), (f"{region_of_interest} 한정", region_of_interest)]:
        idea1 = build_pressure_index(df, lag=lag, region=region)
        idea2 = lead_time_analysis(idea1["firm_month_index"])
        if idea2.empty:
            rows.append({"범위": label, "포착률(b/a)": np.nan, "n_episodes": 0,
                         "리드타임(주)_vs_수출통계_평균": np.nan,
                         "리드타임(주)_vs_광공업생산지수_평균": np.nan})
        else:
            r = idea2.iloc[0]
            rows.append({
                "범위": label,
                "포착률(b/a)": r["포착률(b/a)"],
                "n_episodes": r["n_episodes(=b, 포착된 것만)"],
                "리드타임(주)_vs_수출통계_평균": r["리드타임(주)_vs_수출통계_평균"],
                "리드타임(주)_vs_광공업생산지수_평균": r["리드타임(주)_vs_광공업생산지수_평균"],
            })

    comparison = pd.DataFrame(rows)
    print(f"\n[통합 vs {region_of_interest} 한정 포착률·리드타임 비교, lag={lag}]")
    print(comparison.to_string(index=False))

    if len(comparison) == 2 and comparison["포착률(b/a)"].notna().all():
        delta = comparison["포착률(b/a)"].iloc[1] - comparison["포착률(b/a)"].iloc[0]
        print(f"\n{region_of_interest} 한정 포착률이 통합 대비 {delta*100:+.1f}%p "
              f"{'개선' if delta > 0 else ('악화' if delta < 0 else '변화없음')}.")

    return comparison


# =====================================================================
# 아이디어 3: 업종별 신호 사전
# =====================================================================

def debug_lp_regression(df: pd.DataFrame, industry: str, account_col: str, h: int = 0) -> dict:
    """특정 업종 하나에 대해 LP 회귀를 한 단계씩 수동 실행하며 각 단계 값을 출력.
    coef가 NaN으로 나오는 원인을 찾을 때 이 함수로 재현해서 볼 것.

    industry_signal_dictionary()가 내부적으로 쓰는 것과 완전히 같은 로직
    (_lp_regression_one)을 그대로 호출하되, 중간 산출물을 전부 출력한다.
    """
    print("=" * 70)
    print(f"[디버그] 업종='{industry}', 계정='{account_col}', h={h}")
    print("=" * 70)

    id_col, ym_col = COLS["id"], COLS["ym"]
    industry_col, phase_col, exposed_col = COLS["industry"], COLS["phase"], COLS["exposed"]

    work = _add_midx(df, ym_col)
    sub_ind = work[work[industry_col] == industry].copy()
    print(f"1) 업종 필터 후 행수: {len(sub_ind):,}, 고유 법인수: {sub_ind[id_col].nunique():,}")

    tmp = sub_ind[[id_col, "_midx_", ym_col, account_col, phase_col, exposed_col]].dropna(
        subset=["_midx_", account_col, phase_col, exposed_col]
    ).copy()
    print(f"2) 결측 제거 후 행수: {len(tmp):,} "
          f"(제거된 행: {len(sub_ind) - len(tmp):,} - midx/{account_col}/{phase_col}/{exposed_col} 결측)")

    tmp["_log_val_"] = np.log1p(tmp[account_col].clip(lower=0))
    base = merge_exact_lag(tmp, id_col, "_midx_", "_log_val_", lag=1, new_col="_base_log_")
    fwd = merge_exact_lead(tmp, id_col, "_midx_", "_log_val_", h=h, new_col="_fwd_log_")
    tmp["_y_h_"] = fwd - base
    n_base_missing = pd.isna(base).sum()
    n_fwd_missing = pd.isna(fwd).sum()
    print(f"3) 종속변수(y_h) 구성: base(t-1) 결측 {n_base_missing:,}건, fwd(t+{h}) 결측 {n_fwd_missing:,}건, "
          f"y_h 유효행 {tmp['_y_h_'].notna().sum():,}건")

    tmp["_down_"] = tmp[phase_col].isin(DOWN_PHASES).astype(int)
    tmp["_down_x_exposed_"] = tmp["_down_"] * tmp[exposed_col].astype(int)

    sub = tmp.dropna(subset=["_y_h_"]).drop_duplicates(subset=[id_col, "_midx_"]).copy()
    print(f"4) y_h 결측 제거 + (법인,월) 중복 제거 후 최종 표본: {len(sub):,}행, "
          f"법인수={sub[id_col].nunique():,}, 시점수={sub['_midx_'].nunique():,}")

    if sub.empty:
        print("[중단] 표본이 비었음.")
        return {}

    n_treated = int(sub.loc[(sub["_down_"] == 1) & (sub[exposed_col] == 1), id_col].nunique())
    print(f"5) _down_ 분포: {sub['_down_'].value_counts().to_dict()}, "
          f"_down_x_exposed_ 분포: {sub['_down_x_exposed_'].value_counts().to_dict()}, "
          f"처치법인(down=1&exposed=1) 수={n_treated}")
    print(f"   _down_ 분산={sub['_down_'].var(ddof=0):.6f}, "
          f"_down_x_exposed_ 분산={sub['_down_x_exposed_'].var(ddof=0):.6f}, "
          f"y_h 분산={sub['_y_h_'].var(ddof=0):.6f}")
    print(f"   exposed 고유값 수={sub[exposed_col].nunique()} "
          f"({'상수! down×exposed = down 이 되어 완전공선 위험' if sub[exposed_col].nunique() == 1 else '정상(0/1 둘 다 존재)'})")

    # [실제 원인으로 확인됨] PanelOLS의 time 인덱스는 숫자 또는 날짜형이어야 하는데
    # ym_col("기준년월")은 문자열("202301")일 수 있어 "The index on the time
    # dimension must be either numeric or date-like" 에러가 났다. 이미 만들어둔
    # 정수 시점 인덱스 _midx_(=year*12+month)를 시간축으로 써야 한다.
    sub_idx = sub.set_index([id_col, "_midx_"])
    exog = sub_idx[["_down_", "_down_x_exposed_"]].astype(float)
    y = sub_idx["_y_h_"]
    rank = np.linalg.matrix_rank(exog.values)
    print(f"   exog rank={rank} / 컬럼수={exog.shape[1]} ({'완전공선!' if rank < exog.shape[1] else '정상'})")

    print("6) PanelOLS(entity_effects=True, time_effects=True, drop_absorbed=True) 적합 시도...")
    try:
        model = PanelOLS(y, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = model.fit(cov_type="clustered", cluster_entity=True)
        print(f"   -> 성공. 살아남은 파라미터: {list(res.params.index)}")
        print(res.params.to_string())
        if "_down_x_exposed_" in res.params.index:
            print(f"   coef={res.params['_down_x_exposed_']:.6f}, "
                  f"se={res.std_errors['_down_x_exposed_']:.6f}, "
                  f"p={res.pvalues['_down_x_exposed_']:.6f}")
        else:
            print("   [문제] _down_x_exposed_가 흡수되어 결과 파라미터에 없음(drop_absorbed).")
        return {"result": res}
    except Exception as e:  # noqa: BLE001
        print(f"   -> 실패: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}


def _lp_regression_one(df: pd.DataFrame, id_col: str, midx_col: str, ym_col: str,
                        account_col: str, phase_col: str, exposed_col: str, h: int) -> dict:
    tmp = df[[id_col, midx_col, ym_col, account_col, phase_col, exposed_col]].dropna(
        subset=[midx_col, account_col, phase_col, exposed_col]
    ).copy()
    tmp["_log_val_"] = np.log1p(tmp[account_col].clip(lower=0))
    # Jordà(2005) LP 관행: 기준시점을 t가 아니라 t-1(충격 직전)로 잡는다.
    # 기준을 t로 잡으면 h=0에서 y_h = ln(x_t)-ln(x_t) = 0 이 항상 성립해 계수가
    # 식별 불가(전부 흡수됨)한 축퇴 문제가 생긴다 (실제로 한번 이렇게 짰다가 h=0
    # 계수가 전부 NaN으로 나오는 걸 보고 발견해서 수정함).
    base = merge_exact_lag(tmp, id_col, midx_col, "_log_val_", lag=1, new_col="_base_log_")
    fwd = merge_exact_lead(tmp, id_col, midx_col, "_log_val_", h=h, new_col="_fwd_log_")
    tmp["_y_h_"] = fwd - base
    tmp["_down_"] = tmp[phase_col].isin(DOWN_PHASES).astype(int)
    tmp["_down_x_exposed_"] = tmp["_down_"] * tmp[exposed_col].astype(int)

    sub = tmp.dropna(subset=["_y_h_"]).drop_duplicates(subset=[id_col, midx_col]).copy()
    if sub.empty:
        return {"n_obs": 0, "n_treated": 0, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": "y_h 유효쌍 없음(해당 h만큼의 선/후행 관측이 표본에 없음)"}

    n_treated = int(sub.loc[(sub["_down_"] == 1) & (sub[exposed_col] == 1), id_col].nunique())

    # --- 사전 점검: PanelOLS에 넣기 전에 "완전공선/무변동"으로 회귀가 애초에 불가능한
    # 경우를 직접 걸러 이유를 남긴다. 실제 데이터는 특정 업종이 한 지역에 쏠려있거나
    # (down이 region×time 단위로 결정되므로 지역이 하나뿐이면 down이 시간FE와
    # 완전공선), 노출 법인이 전부 같은 시점에만 관측되는 등(불균형 패널) 합성
    # 데이터에서는 못 본 방식으로 무너질 수 있다.
    n_entities = sub[id_col].nunique()
    n_times = sub[midx_col].nunique()
    down_var = sub["_down_"].var(ddof=0)
    inter_var = sub["_down_x_exposed_"].var(ddof=0)
    y_var = sub["_y_h_"].var(ddof=0)

    if pd.isna(y_var) or y_var == 0:
        # linearmodels가 이 경우 깔끔한 에러 대신 ZeroDivisionError를 던지는 걸 실제로
        # 재현해서 확인함 - 종속변수 자체가 상수면 애초에 회귀할 게 없으므로 미리 차단.
        return {"n_obs": len(sub), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": "종속변수(y_h)가 이 표본에서 전혀 변하지 않음(상수) - 회귀 불가"}

    if n_times < 2:
        return {"n_obs": len(sub), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": f"고유 시점(ym)이 {n_times}개뿐 - 연월FE와 함께 식별 불가"}
    if pd.isna(down_var) or down_var == 0:
        const_state = "전부 하락국면" if sub["_down_"].iloc[0] == 1 else "전부 증가국면(하락 없음)"
        return {"n_obs": len(sub), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": f"이 표본에서 하락국면(_down_)이 시간 내내 변하지 않음({const_state})"}
    if pd.isna(inter_var) or inter_var == 0:
        return {"n_obs": len(sub), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": "down×exposed 상호작용에 변동이 없음(처치법인이 하락국면에 걸린 적이 없거나, "
                          "노출법인이 아예 없음)"}

    # exposed가 이 업종 안에서 상수(전부 1, 또는 전부 0)이면 _down_x_exposed_ = _down_
    # 이 되어 두 회귀변수가 완전히 같아진다("exog does not have full column rank").
    # (주: 처음엔 이게 도매/자동차/섬유/1차금속 전체 NaN의 원인이라고 생각했으나,
    # debug_lp_regression으로 실제 추적해보니 '도매 및 상품 중개업'은 exposed가
    # 0/1 둘 다 있어 이 조건에 해당하지 않았다 - 그래도 다른 업종에서는 실제로
    # 발생할 수 있는 조건이라 점검은 유지한다. 도매 등 전체가 NaN이었던 진짜 원인은
    # 아래 set_index에서 시간축으로 ym_col(문자열 "202301")을 그대로 써서
    # "The index on the time dimension must be either numeric or date-like" 에러가
    # 났던 것 - _midx_(정수)로 바꿔서 해결함.)
    if sub[exposed_col].nunique() == 1:
        return {"n_obs": len(sub), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": f"이 업종은 exposed가 상수(전부 {int(sub[exposed_col].iloc[0])}) - "
                          f"down×exposed가 down과 완전히 같아져 공선성으로 회귀 불가"}

    sub = sub.set_index([id_col, midx_col])
    exog = sub[["_down_", "_down_x_exposed_"]].astype(float)
    y = sub["_y_h_"]

    # 안전망: 위에서 못 잡은 다른 형태의 완전공선도 PanelOLS를 부르기 전에 수치적으로 확인.
    if np.linalg.matrix_rank(exog.values) < exog.shape[1]:
        return {"n_obs": len(sub), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                "reason": "exog(설계행렬)이 완전공선(rank 부족) - down/down×exposed 조합을 다시 점검할 것"}

    try:
        model = PanelOLS(y, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = model.fit(cov_type="clustered", cluster_entity=True)
        if "_down_x_exposed_" not in res.params.index:
            # drop_absorbed=True면 완전공선 변수는 조용히 제거되고 예외를 안 던진다.
            # 우리가 진짜 보고 싶은 계수 자체가 흡수돼서 없어진 경우를 명시적으로 잡는다.
            return {"n_obs": int(res.nobs), "n_treated": n_treated, "coef": np.nan, "se": np.nan, "p": np.nan,
                    "reason": "down×exposed 항이 법인FE+연월FE에 완전히 흡수되어 제거됨(drop_absorbed)"}
        coef = float(res.params["_down_x_exposed_"])
        se = float(res.std_errors["_down_x_exposed_"])
        p = float(res.pvalues["_down_x_exposed_"])
        n_obs = int(res.nobs)
        reason = ""
    except Exception as e:  # noqa: BLE001
        coef = se = p = np.nan
        n_obs = len(sub)
        reason = f"{type(e).__name__}: {e}"

    return {"n_obs": n_obs, "n_treated": n_treated, "coef": coef, "se": se, "p": p, "reason": reason}


def industry_signal_dictionary(df: pd.DataFrame) -> dict:
    """업종_중분류 × 계정 × horizon(0~6)별 LP 반응계수(하락국면×exposed 상호작용)를
    추정해 '업종별 관측 매뉴얼'을 만든다. 처치법인(down=1&exposed=1) 40 미만 셀은
    신뢰주의 플래그."""
    print("\n" + "=" * 70)
    print("아이디어 3. 업종별 신호 사전 (LP, h=0~6, 법인FE+연월FE, 법인클러스터SE)")
    print("=" * 70)

    id_col, ym_col = COLS["id"], COLS["ym"]
    industry_col, phase_col, exposed_col = COLS["industry"], COLS["phase"], COLS["exposed"]

    work = _add_midx(df, ym_col)
    accounts = {role: col for role, col in ACCOUNTS_FOR_DICTIONARY.items() if col in work.columns}
    missing_accounts = [role for role, col in ACCOUNTS_FOR_DICTIONARY.items() if col not in work.columns]
    if missing_accounts:
        print(f"[경고] 데이터에 없어 생략되는 계정: {missing_accounts}")

    industries = sorted(work[industry_col].dropna().unique().tolist())
    print(f"[진행] 업종 {len(industries)}개 × 계정 {len(accounts)}개 × horizon {len(list(LP_HORIZONS))}개 추정 시작...")

    all_rows = []
    for industry in industries:
        sub_ind = work[work[industry_col] == industry]
        for role, col in accounts.items():
            for h in LP_HORIZONS:
                r = _lp_regression_one(sub_ind, id_col, "_midx_", ym_col, col, phase_col, exposed_col, h)
                all_rows.append({
                    "업종": industry, "계정": role, "h": h,
                    "n_obs": r["n_obs"], "n_처치법인": r["n_treated"],
                    "coef": round(r["coef"], 6) if pd.notna(r["coef"]) else np.nan,
                    "se": round(r["se"], 6) if pd.notna(r["se"]) else np.nan,
                    "p": round(r["p"], 6) if pd.notna(r["p"]) else np.nan,
                    "신뢰주의": r["n_treated"] < MIN_TREATED_FOR_CONFIDENCE,
                    "실패사유": r.get("reason", ""),
                })
        print(f"  [진행] '{industry}' 완료")

    full_table = pd.DataFrame(all_rows)
    print("\n[전체 LP 계수 표 (업종×계정×h)] - 참고용 전체본, 신뢰주의=True인 셀은 표본부족")
    print(full_table.to_string(index=False))

    # coef가 NaN인데 신뢰주의도 아닌(=표본은 충분한데 회귀 자체가 실패한) 셀은 별도로
    # 부각 - "표본부족"이 아니라 진짜 원인(공선/흡수/예외)이 있다는 신호.
    coef_nan_but_sample_ok = full_table[full_table["coef"].isna() & (~full_table["신뢰주의"])]
    if not coef_nan_but_sample_ok.empty:
        print(f"\n[경고] 처치법인은 {MIN_TREATED_FOR_CONFIDENCE}개 이상인데 coef가 NaN인 셀 "
              f"{len(coef_nan_but_sample_ok)}개 발견 - '표본부족'이 아니라 회귀 자체가 실패한 것.")
        reason_counts = coef_nan_but_sample_ok["실패사유"].value_counts()
        print("[실패사유별 건수]")
        print(reason_counts.to_string())

    # 업종별 '관측 매뉴얼': 신뢰 가능한 셀 중 |coef| 최댓값을 갖는 (계정,h) 선택
    manual_rows = []
    for industry in industries:
        ind_rows = full_table[full_table["업종"] == industry]
        cand = ind_rows[(~ind_rows["신뢰주의"]) & ind_rows["coef"].notna()]
        if cand.empty:
            if (ind_rows["신뢰주의"]).all():
                verdict = "표본부족(신뢰불가)"
            else:
                verdict = "회귀실패(신뢰불가) - 실패사유 컬럼 확인"
            manual_rows.append({"업종": industry, "추천계정": None, "추천h": None,
                                 "coef": np.nan, "p": np.nan, "판정": verdict})
            continue
        best = cand.loc[cand["coef"].abs().idxmax()]
        manual_rows.append({
            "업종": industry, "추천계정": best["계정"], "추천h": int(best["h"]),
            "coef": best["coef"], "p": best["p"],
            "판정": "신뢰가능" if best["p"] < 0.05 else "신뢰가능(비유의)",
        })

    manual_table = pd.DataFrame(manual_rows)
    print("\n[업종별 관측 매뉴얼] - 업종마다 '이 계정/시차를 봐라'")
    print(manual_table.to_string(index=False))

    return {"full_table": full_table, "manual_table": manual_table}


# =====================================================================
# 사용 예시 (실행하지 않음 - 참고용)
# =====================================================================
"""
# df: 법인ID/기준년월/사업장_시도/업종_중분류/요구불예금잔액/운전_할인어음잔액/
#     exposed/exp_yoy/phase 가 이미 있는 병합 완료 데이터프레임

from financial_pressure_signals import (
    ensure_exposed_column, ensure_phase_column,
    build_pressure_index, lead_time_analysis, industry_signal_dictionary,
    compare_lag_definitions, compare_lead_time_by_scope,
    COLS,
)

# 없으면 폴백으로 채우기 (exp_yoy는 이미 있다고 가정 - 없으면 STEP15/16으로 먼저 병합)
df = ensure_exposed_column(df, COLS["id"], "외환_수출실적금액", "외환_수입실적금액")
df = ensure_phase_column(df, COLS["exp_yoy"], COLS["region"],
                          deep_decline_cutoff={"대구": -22.21, "경북": -14.92})

# 기본(lag=3, 개선판)으로 실행
idea1 = build_pressure_index(df)  # lag=DEFAULT_PRESSURE_LAG(3), region=None(통합)
idea2 = lead_time_analysis(idea1["firm_month_index"])
idea3 = industry_signal_dictionary(df)

# lag=1(원래) vs lag=3(개선) 비교표
compare_lag_definitions(df)                       # 통합
compare_lag_definitions(df, region="대구")         # 대구 한정

# 통합 vs 대구한정 포착률·리드타임 비교 (개선된 lag=3 기준)
compare_lead_time_by_scope(df, lag=3, region_of_interest="대구")
"""


# =====================================================================
# [추가] 달력(영업일수) 보정 재검증 — 민영 파트 ① (2026-09-28)
# 위의 원본 함수는 한 줄도 바꾸지 않았다. 아래 함수만 새로 추가했다.
# 원본 정의는 그대로 쓴다: ln(x+1) lag개월 차분, in-sample 표준화, 신호 문턱
# PRESSURE_SIGNAL_THRESHOLD, 같은 달 포착, 시간순 중앙값 분할, in/out 방향 정의.
# 바뀌는 것 (사전 고정):
#   (a) 수출 % YoY에서 영업일수 전년차 성분 제거 (대경 pooled 기울기) → 컷오프 재계산 → 3국면 재분류
#   (b) 요구불·할인어음 lag개월 차분에서 같은 창 영업일수 변화 성분 제거 (pooled 기울기)
# 새 지표(오경보율·신호 켜짐 비율·위약)의 분모는 대구·경북 지역×월만 쓴다.
# =====================================================================

DG_REGIONS = ("대구", "경북")
BIZDAY_FILE = r"C:\test\data\ext_bizday.csv"
PHASE_PERIOD = (202301, 202512)   # 컷오프 계산 기간 (원본 −22.21/−14.92가 재현되는 기간)
NPLAC_CAPTURE = 200
PLACEBO_SEED = 2026


def load_bizdays(path: str = BIZDAY_FILE) -> pd.Series:
    """영업일수를 midx(= 연*12 + 월) 인덱스 Series로 읽는다."""
    b = pd.read_csv(path, encoding="utf-8-sig")
    return pd.Series(b["bizdays"].astype(float).values,
                     index=(b["ym"] // 100) * 12 + b["ym"] % 100)


def _classify(yoy, cutoff):
    if pd.isna(yoy):
        return np.nan
    if yoy >= 0:
        return "증가"
    return "깊은하락" if yoy < cutoff else "하락"


def adjust_export_phase(df: pd.DataFrame, biz: pd.Series) -> tuple:
    """(a) 대경 지역×월 수출 % YoY에서 영업일수 전년차 성분을 pooled 기울기로 제거한다
    (평균 수준은 유지, segment12c cal_adjust와 같은 방식). 보정 YoY로 지역별 컷오프
    (평균 − 1SD, ddof=1, PHASE_PERIOD)를 다시 계산하고 3국면으로 재분류해 df에 반영한다."""
    region_col, ym_col = COLS["region"], COLS["ym"]
    rm = (df[df[region_col].isin(DG_REGIONS)]
          .drop_duplicates([region_col, ym_col])[[region_col, ym_col, "exp_yoy", "phase"]].copy())
    rm = rm[(rm[ym_col] >= PHASE_PERIOD[0]) & (rm[ym_col] <= PHASE_PERIOD[1])]
    midx = (rm[ym_col] // 100) * 12 + rm[ym_col] % 100
    rm["bd_yoy"] = biz.reindex(midx).values - biz.reindex(midx - 12).values
    ok = rm["exp_yoy"].notna() & rm["bd_yoy"].notna()
    slope, _ = np.polyfit(rm.loc[ok, "bd_yoy"], rm.loc[ok, "exp_yoy"], 1)
    corr = np.corrcoef(rm.loc[ok, "bd_yoy"], rm.loc[ok, "exp_yoy"])[0, 1]
    rm["exp_yoy_adj"] = rm["exp_yoy"] - slope * rm["bd_yoy"]

    cut_before = {g: s.mean() - s.std(ddof=1) for g, s in rm.groupby(region_col)["exp_yoy"]}
    cut_after = {g: s.mean() - s.std(ddof=1) for g, s in rm.groupby(region_col)["exp_yoy_adj"]}
    rm["phase_adj"] = [_classify(y, cut_after[g]) for y, g in zip(rm["exp_yoy_adj"], rm[region_col])]

    counts = []
    for g, sub in rm.groupby(region_col):
        for label, col, cuts in [("보정 전", "phase", cut_before), ("보정 후", "phase_adj", cut_after)]:
            vc = sub[col].value_counts()
            counts.append({"지역": g, "구분": label, "컷오프": round(cuts[g], 2),
                           "증가": int(vc.get("증가", 0)), "하락": int(vc.get("하락", 0)),
                           "깊은하락": int(vc.get("깊은하락", 0))})

    out = df.merge(rm[[region_col, ym_col, "exp_yoy_adj", "phase_adj"]], on=[region_col, ym_col], how="left")
    dg_rows = out[region_col].isin(DG_REGIONS) & out["phase_adj"].notna()
    out.loc[dg_rows, "exp_yoy"] = out.loc[dg_rows, "exp_yoy_adj"]
    out.loc[dg_rows, "phase"] = out.loc[dg_rows, "phase_adj"]
    out = out.drop(columns=["exp_yoy_adj", "phase_adj"])
    info = {"slope": slope, "corr": corr, "n": int(ok.sum()),
            "cut_before": cut_before, "cut_after": cut_after, "phase_counts": pd.DataFrame(counts)}
    return out, info


def _firm_month_changes(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """원본 build_pressure_index와 같은 방식의 ln(x+1) lag개월 차분 (정확히 lag개월 전 관측만)."""
    id_col, ym_col = COLS["id"], COLS["ym"]
    work = _add_midx(df, ym_col).sort_values([id_col, "_midx_"]).reset_index(drop=True)
    for col, chg_name in [(COLS["deposit"], "_dep_chg_"), (COLS["bill"], "_bill_chg_")]:
        log_col = f"_log_{chg_name}"
        work[log_col] = np.log1p(work[col].clip(lower=0))
        prev_log = merge_exact_lag(work, id_col, "_midx_", log_col, lag=lag, new_col=f"_prev_{log_col}")
        work[chg_name] = work[log_col].values - prev_log
    return work.dropna(subset=["_dep_chg_", "_bill_chg_"]).copy()


def estimate_bank_slopes(df: pd.DataFrame, biz: pd.Series, lag: int = DEFAULT_PRESSURE_LAG) -> dict:
    """(b) 요구불·할인어음 lag개월 차분 ~ 같은 창 영업일수 변화(bizdays(t) − bizdays(t−lag)).
    전체 기간·전체 법인×월로 pooled 기울기 하나씩 추정한다."""
    v = _firm_month_changes(df, lag)
    dbd = biz.reindex(v["_midx_"]).values - biz.reindex(v["_midx_"] - lag).values
    out = {"lag": lag}
    for key, col in [("dep", "_dep_chg_"), ("bill", "_bill_chg_")]:
        y = v[col].values
        ok = np.isfinite(dbd) & np.isfinite(y)
        slope, _ = np.polyfit(dbd[ok], y[ok], 1)
        out[key] = {"slope": slope, "corr": np.corrcoef(dbd[ok], y[ok])[0, 1], "n": int(ok.sum())}
    return out


def build_pressure_index_adj(df: pd.DataFrame, lag: int = DEFAULT_PRESSURE_LAG,
                             region: "str | None" = None, bank_slopes: "dict | None" = None,
                             biz: "pd.Series | None" = None) -> pd.DataFrame:
    """원본 build_pressure_index의 계산 부분만 옮긴 것 (진단 출력 생략). bank_slopes를 주면
    표준화 전에 차분에서 slope × 영업일수 변화를 뺀다. 반환: 법인×월 압박지수 (집계용)."""
    id_col, ym_col, region_col, phase_col = COLS["id"], COLS["ym"], COLS["region"], COLS["phase"]
    if region is not None:
        df = df[df[region_col] == region]
    valid = _firm_month_changes(df, lag)
    if bank_slopes is not None:
        dbd = biz.reindex(valid["_midx_"]).values - biz.reindex(valid["_midx_"] - lag).values
        valid["_dep_chg_"] = valid["_dep_chg_"].values - bank_slopes["dep"]["slope"] * dbd
        valid["_bill_chg_"] = valid["_bill_chg_"].values - bank_slopes["bill"]["slope"] * dbd
        valid = valid[np.isfinite(dbd)].copy()
    unique_midx = np.sort(valid["_midx_"].unique())
    median_midx = unique_midx[len(unique_midx) // 2]
    valid["_period_"] = np.where(valid["_midx_"] <= median_midx, "in-sample(전반부)", "out-sample(후반부)")
    ins = valid[valid["_period_"] == "in-sample(전반부)"]
    valid["pressure_index"] = (-(valid["_dep_chg_"] - ins["_dep_chg_"].mean()) / ins["_dep_chg_"].std()
                               + (valid["_bill_chg_"] - ins["_bill_chg_"].mean()) / ins["_bill_chg_"].std())
    return valid[[id_col, ym_col, "_midx_", region_col, phase_col, "_period_", "pressure_index"]].copy()


def inout_consistency(fmi: pd.DataFrame) -> dict:
    """원본 compare_lag_definitions와 같은 정의: 두 구간 모두 하락계열 최대 평균 > 증가 평균."""
    s = fmi.groupby(["_period_", COLS["phase"]])["pressure_index"].mean().unstack()
    holds = {}
    for period in ["in-sample(전반부)", "out-sample(후반부)"]:
        down = [p for p in DOWN_PHASES if p in s.columns]
        holds[period] = bool(period in s.index and "증가" in s.columns and down
                             and s.loc[period, down].max() > s.loc[period, "증가"])
    return {"in": holds["in-sample(전반부)"], "out": holds["out-sample(후반부)"],
            "일관": all(holds.values()), "means": s}


def capture_metrics(fmi: pd.DataFrame, nplac: int = NPLAC_CAPTURE, seed: int = PLACEBO_SEED) -> dict:
    """대구·경북 지역×월 기준 포착률·오경보율·신호 켜짐 비율 + 위약(지역별 달 순서 독립 섞기).
    포착률 정의는 원본 lead_time_analysis와 같다 (하락국면 중 같은 달 신호 켜짐 비율)."""
    region_col = COLS["region"]
    rm = (fmi[fmi[region_col].isin(DG_REGIONS)]
          .groupby([region_col, "_midx_"])
          .agg(phase=(COLS["phase"], "first"), mean_pressure=("pressure_index", "mean"))
          .reset_index())
    on = (rm["mean_pressure"] > PRESSURE_SIGNAL_THRESHOLD).values
    down = rm["phase"].isin(DOWN_PHASES).values

    def cap(o):
        return o[down].mean() if down.any() else np.nan

    actual = cap(on)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero((rm[region_col] == g).values) for g in rm[region_col].unique()]
    sims = []
    for _ in range(nplac):
        perm_on = on.copy()
        for idx in groups:
            perm_on[idx] = on[rng.permutation(idx)]
        sims.append(cap(perm_on))
    sims = np.array(sims)
    n_on = int(on.sum())
    return {"지역월": len(rm), "하락월": int(down.sum()), "포착": int((on & down).sum()),
            "포착률": actual, "신호켜짐": n_on, "신호켜짐비율": on.mean(),
            "오경보율": ((on & ~down).sum() / n_on) if n_on else np.nan,
            "위약_p": (np.sum(sims >= actual) + 1) / (nplac + 1), "위약_평균포착률": sims.mean()}
