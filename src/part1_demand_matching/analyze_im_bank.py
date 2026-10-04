# -*- coding: utf-8 -*-
"""
iM뱅크 법인 익명 패널 데이터 진단 스크립트
- 원본 행/법인ID/개별 레코드는 절대 출력하지 않음 (집계 결과만 출력)
- 파일 저장(to_csv 등) 없음, 화면 출력만
- 실행: python analyze_im_bank.py [csv경로]
"""

import sys
import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix

try:
    from linearmodels.panel import PanelOLS
except ImportError as e:  # noqa: BLE001
    raise ImportError(
        "linearmodels 패키지가 필요합니다. 'pip install linearmodels' 후 다시 실행하세요."
    ) from e

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

# =====================================================================
# 0. 설정: 파일 경로 & 컬럼 맵 (실제 컬럼명에 맞게 값만 채우세요)
# =====================================================================

FILE_PATH = r"C:\test\data\raw\(iM뱅크) 2026 교육용 법인 익명데이터.xlsx"  # 필요시 명령행 인자로 덮어씀
SHEET_NAME = 0  # 엑셀 파일일 때 읽을 시트 (이름 또는 인덱스)
# 대구/경북 수출입 데이터가 지역별로 별도 K-stat(한국무역협회) 파일에 나뉘어 있음
# (data\raw 폴더에 위치, 파일명에 지역이 안 담겨 있어 수동 매핑)
EXTERNAL_DIR = r"C:\test\data\raw"
EXTERNAL_FILES = {
    "K-stat 무역통계 - 한국무역협회 (1).xls": "대구",
    "K-stat 무역통계 - 한국무역협회 (2).xls": "경북",
}

COLUMN_MAP = {
    "id_col": "법인ID",
    "ym_col": "기준년월",                # 예: 202301 (int 또는 str)
    "운전자금": "여신_운전자금대출잔액",
    "시설자금": "여신_시설자금대출잔액",
    "요구불": "요구불예금잔액",
    "거래금액": "",                      # 있으면 채우고, 없으면 빈 문자열로 둘 것
    "수출실적": "외환_수출실적금액",
    "수입실적": "외환_수입실적금액",
    "시도": "사업장_시도",
    "업종중분류": "업종_중분류",
    "거치식": "거치식예금잔액",
    "적립식": "적립식예금잔액",
    "요구불입금": "요구불입금금액",
    "요구불출금": "요구불출금금액",
    "무역금융": "운전_무역금융잔액",
    "업종대분류": "업종_대분류",
    "운전_할인어음": "운전_할인어음잔액",
    "운전_당좌대출": "운전_당좌대출잔액",
    "운전_일반자금대출": "운전_일반자금대출잔액",
    "운전_주택자금대출": "운전_주택자금대출잔액",
    "운전_기업구매자금대출": "운전_기업구매자금대출잔액",
    "운전_외상매출채권담보대출": "운전_외상매출채권담보대출잔액",
    "운전_기타운전자금대출": "운전_기타운전자금대출잔액",
    "시설_일반자금대출": "시설_일반자금대출잔액",
    "시설_에너지절약시설대출": "시설_에너지절약시설대출잔액",
    "시설_주택자금대출": "시설_주택자금대출잔액",
    "시설_기타시설자금대출": "시설_기타시설자금대출잔액",
}

BALANCE_VARS = ["운전자금", "시설자금", "요구불", "거래금액"]
FX_VARS = ["수출실적", "수입실적"]
ACCOUNT_VARS_STEP8 = ["운전자금", "시설자금", "요구불", "거치식", "적립식"]

WORKING_CAPITAL_SUBITEMS = [
    "운전_할인어음", "운전_당좌대출", "운전_일반자금대출", "무역금융",
    "운전_주택자금대출", "운전_기업구매자금대출", "운전_외상매출채권담보대출", "운전_기타운전자금대출",
]
FACILITY_SUBITEMS = [
    "시설_일반자금대출", "시설_에너지절약시설대출", "시설_주택자금대출", "시설_기타시설자금대출",
]

NO_VARIATION_EPS = 1e-9


# =====================================================================
# 1. 로딩 유틸
# =====================================================================

def load_data_with_fallback(path: str, sheet_name=0) -> pd.DataFrame:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""

    if ext in ("xlsx", "xlsm"):
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        print(f"[로딩] 엑셀 파일(openpyxl)로 성공적으로 읽음. (sheet={sheet_name})")
        return df
    if ext == "xls":
        df = pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
        print(f"[로딩] 엑셀 파일(xlrd)로 성공적으로 읽음. (sheet={sheet_name})")
        return df

    encodings = ["utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[로딩] 인코딩 '{enc}' 로 성공적으로 읽음.")
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[로딩] 인코딩 '{enc}' 시도 실패: {type(e).__name__}")
    raise RuntimeError(f"모든 인코딩 시도 실패. 마지막 에러: {last_err}")


def resolve_columns(df: pd.DataFrame, column_map: dict) -> dict:
    """COLUMN_MAP 중 실제 df에 존재하는 컬럼만 반환. 없는 건 경고만 출력."""
    resolved = {}
    for role, colname in column_map.items():
        if not colname:
            continue
        if colname in df.columns:
            resolved[role] = colname
        else:
            print(f"[경고] COLUMN_MAP['{role}'] = '{colname}' 가 데이터에 없어 건너뜁니다.")
    return resolved


def normalize_ym_value(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) == 6:
        y, m = int(digits[:4]), int(digits[4:6])
    elif len(digits) == 8:
        # YYYYMMDD 형태로 들어온 경우 앞 6자리만 사용
        y, m = int(digits[:4]), int(digits[4:6])
    else:
        return np.nan
    if 1 <= m <= 12 and 2000 <= y <= 2100:
        return y * 100 + m
    return np.nan


def add_normalized_ym(df: pd.DataFrame, ym_col: str) -> str:
    new_col = "__ym_norm__"
    df[new_col] = df[ym_col].map(normalize_ym_value)
    n_bad = df[new_col].isna().sum()
    if n_bad > 0:
        print(f"[경고] 기준년월 정규화 실패 {n_bad}건 (NaN 처리됨).")
    return new_col


def add_month_index(df: pd.DataFrame, ym_norm_col: str) -> str:
    new_col = "__month_idx__"
    year = (df[ym_norm_col] // 100)
    month = (df[ym_norm_col] % 100)
    df[new_col] = year * 12 + month
    return new_col


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =====================================================================
# STEP 1. 기본 구조
# =====================================================================

def step1_basic_structure(df: pd.DataFrame, cols: dict, ym_norm_col: str):
    section("STEP 1. 기본 구조")

    id_col = cols.get("id_col")
    print(f"행수: {len(df):,}")
    if id_col:
        print(f"고유 법인수: {df[id_col].nunique():,}")
    else:
        print("[경고] id_col 미확인으로 고유 법인수 계산 생략.")

    valid_ym = df[ym_norm_col].dropna()
    if not valid_ym.empty:
        print(f"기준년월 min: {int(valid_ym.min())}, max: {int(valid_ym.max())}")
        print(f"고유 개월수: {valid_ym.nunique()}")
    else:
        print("[경고] 유효한 기준년월 값이 없습니다.")

    print("\n[컬럼별 dtype / 결측수 / 결측비율]")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_ratio": (df.isna().mean() * 100).round(2),
    })
    print(info.to_string())

    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"\n메모리 사용량: {mem_mb:,.2f} MB")


# =====================================================================
# STEP 2. 키 유일성
# =====================================================================

def step2_key_uniqueness(df: pd.DataFrame, cols: dict, ym_norm_col: str):
    section("STEP 2. 키 유일성 ([id_col, 기준년월] 중복 점검)")

    id_col = cols.get("id_col")
    if not id_col:
        print("[경고] id_col 미확인으로 STEP2 생략.")
        return

    dup_mask = df.duplicated(subset=[id_col, ym_norm_col], keep=False)
    dup_count = int(dup_mask.sum())
    print(f"중복 키(행) 건수: {dup_count:,}")

    if dup_count > 0:
        dup_by_id = df.loc[dup_mask].groupby(id_col).size()
        print(f"중복이 발생한 법인 수: {dup_by_id.shape[0]:,}")
        print("\n[법인별 중복 행 개수 분포 (describe)]")
        print(dup_by_id.describe().to_string())
    else:
        print("중복 없음.")


# =====================================================================
# STEP 3. 단위·분포 진단
# =====================================================================

def _describe_series(s: pd.Series) -> pd.Series:
    n = len(s)
    missing = s.isna().sum()
    valid = s.dropna()
    pos = (valid > 0).sum()
    zero = (valid == 0).sum()

    pct = [.01, .05, .25, .5, .75, .9, .95, .99]
    desc = valid.describe(percentiles=pct) if not valid.empty else pd.Series(dtype=float)

    out = {
        "count": n,
        "missing": missing,
        "missing_ratio(%)": round(missing / n * 100, 2) if n else np.nan,
        "nunique": s.nunique(dropna=True),
        "positive_count": pos,
        "positive_ratio(%)": round(pos / len(valid) * 100, 2) if len(valid) else np.nan,
        "zero_count": zero,
    }
    for k in ["min", "1%", "5%", "25%", "50%", "75%", "90%", "95%", "99%", "max", "mean", "std"]:
        out[k] = desc.get(k, np.nan)
    return pd.Series(out)


def _describe_var(df: pd.DataFrame, colname: str) -> pd.Series:
    return _describe_series(df[colname])


def step3_unit_distribution(df: pd.DataFrame, cols: dict):
    section("STEP 3. 단위·분포 진단")

    balance_cols = [(k, cols[k]) for k in BALANCE_VARS if k in cols]
    fx_cols = [(k, cols[k]) for k in FX_VARS if k in cols]

    if balance_cols:
        print("\n[잔액류 (백만원 추정)]")
        tbl = pd.DataFrame({role: _describe_var(df, colname) for role, colname in balance_cols})
        print(tbl.to_string())
    else:
        print("[경고] 잔액류 컬럼이 하나도 확인되지 않아 생략.")

    if fx_cols:
        print("\n[외환실적 (백만달러 추정)]")
        tbl = pd.DataFrame({role: _describe_var(df, colname) for role, colname in fx_cols})
        print(tbl.to_string())
    else:
        print("[경고] 외환실적 컬럼이 하나도 확인되지 않아 생략.")

    if balance_cols and fx_cols:
        print("\n[스케일 비교: 잔액류 vs 외환실적 max 값]")
        rows = []
        for role, colname in balance_cols + fx_cols:
            rows.append({"변수": role, "max": df[colname].max(), "median(>0)": df.loc[df[colname] > 0, colname].median()})
        print(pd.DataFrame(rows).to_string(index=False))


# =====================================================================
# STEP 4. 무변동 비율 (월간 / 12개월)
# =====================================================================

def compute_no_variation_ratio(df: pd.DataFrame, id_col: str, midx_col: str,
                                value_col: str, lag: int, eps: float = NO_VARIATION_EPS):
    tmp = df[[id_col, midx_col, value_col]].dropna(subset=[midx_col, value_col]).copy()
    if tmp.empty:
        return {"valid_pairs": 0, "no_variation_ratio(%)": np.nan}

    neg = (tmp[value_col] < 0).sum()
    if neg > 0:
        warnings.warn(f"{value_col}: 음수 값 {neg}건 발견, log1p 계산 위해 0으로 clip")
    tmp["log_val"] = np.log1p(tmp[value_col].clip(lower=0))

    cur = tmp[[id_col, midx_col, "log_val"]]
    prev = tmp[[id_col, midx_col, "log_val"]].copy()
    prev[midx_col] = prev[midx_col] + lag
    prev = prev.rename(columns={"log_val": "prev_log"})

    merged = cur.merge(prev, on=[id_col, midx_col], how="inner")
    if merged.empty:
        return {"valid_pairs": 0, "no_variation_ratio(%)": np.nan}

    diff = (merged["log_val"] - merged["prev_log"]).abs()
    no_var_ratio = (diff < eps).mean() * 100
    return {"valid_pairs": int(len(merged)), "no_variation_ratio(%)": round(no_var_ratio, 2)}


def step4_no_variation(df: pd.DataFrame, cols: dict, midx_col: str):
    section("STEP 4. 종속변수 후보별 무변동 비율")

    id_col = cols.get("id_col")
    if not id_col:
        print("[경고] id_col 미확인으로 STEP4 생략.")
        return

    target_vars = [k for k in BALANCE_VARS if k in cols]
    if not target_vars:
        print("[경고] 대상 변수(운전자금/시설자금/요구불/거래금액) 컬럼이 없어 생략.")
        return

    rows_month = []
    rows_year = []
    for role in target_vars:
        colname = cols[role]
        r1 = compute_no_variation_ratio(df, id_col, midx_col, colname, lag=1)
        r12 = compute_no_variation_ratio(df, id_col, midx_col, colname, lag=12)
        rows_month.append({"변수": role, **r1})
        rows_year.append({"변수": role, **r12})

    print("\n[월간(직전 1개월) 무변동 비율: |ln(x+1) 차분| < 1e-9]")
    print(pd.DataFrame(rows_month).to_string(index=False))

    print("\n[전년동월(직전 12개월) 무변동 비율: |ln(x+1) 차분| < 1e-9]")
    print(pd.DataFrame(rows_year).to_string(index=False))

    print("\n(참고: 시설자금은 월간 기준 약 97.1% 부근이 정상 범위로 알려져 있음 - 대조용)")


# =====================================================================
# STEP 5. 처치군(외환 노출) 정의 재현
# =====================================================================

def step5_treatment_groups(df: pd.DataFrame, cols: dict):
    section("STEP 5. 처치군(외환 노출) 정의 재현")

    id_col = cols.get("id_col")
    export_col = cols.get("수출실적")
    import_col = cols.get("수입실적")

    if not id_col or not export_col:
        print("[경고] id_col 또는 수출실적 컬럼 미확인으로 STEP5 생략.")
        return

    agg_dict = {export_col: "max"}
    if import_col:
        agg_dict[import_col] = "max"

    firm_level = df.groupby(id_col).agg(agg_dict)

    export_exposed = firm_level[export_col].fillna(0) > 0
    if import_col:
        fx_exposed = (firm_level[export_col].fillna(0) > 0) | (firm_level[import_col].fillna(0) > 0)
    else:
        fx_exposed = export_exposed.copy()
        print("[경고] 수입실적 컬럼이 없어 외환노출 = 수출노출과 동일하게 계산됨.")

    control = ~fx_exposed

    total_firms = firm_level.shape[0]
    summary = pd.DataFrame({
        "법인수": [export_exposed.sum(), fx_exposed.sum(), control.sum()],
        "전체대비비율(%)": [
            round(export_exposed.sum() / total_firms * 100, 2),
            round(fx_exposed.sum() / total_firms * 100, 2),
            round(control.sum() / total_firms * 100, 2),
        ],
    }, index=["수출노출", "외환노출(수출 또는 수입)", "대조군(외환실적 전무)"])
    print(f"\n전체 법인수: {total_firms:,}")
    print(summary.to_string())

    sido_col = cols.get("시도")
    if sido_col:
        firm_sido = df.groupby(id_col)[sido_col].agg(
            lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan
        )

        def region_breakdown(mask, label):
            ids_in_group = firm_sido.loc[mask.reindex(firm_sido.index, fill_value=False)]
            n = len(ids_in_group)
            if n == 0:
                print(f"\n[{label}] 법인수 0, 시도 분포 생략")
                return
            is_daegu = ids_in_group.astype(str).str.contains("대구", na=False)
            is_gyeongbuk = ids_in_group.astype(str).str.contains("경북|경상북도", na=False)
            print(f"\n[{label}] 시도 분포 (n={n:,})")
            print(f"  대구 비중: {is_daegu.sum():,} ({is_daegu.mean()*100:.2f}%)")
            print(f"  경북 비중: {is_gyeongbuk.sum():,} ({is_gyeongbuk.mean()*100:.2f}%)")
            print(f"  대구+경북 비중: {(is_daegu | is_gyeongbuk).sum():,} ({(is_daegu | is_gyeongbuk).mean()*100:.2f}%)")

        region_breakdown(export_exposed, "수출노출")
        region_breakdown(fx_exposed, "외환노출")
        region_breakdown(control, "대조군")
    else:
        print("\n[경고] 시도 컬럼이 없어 지역 분포 생략.")


# =====================================================================
# STEP 23. 데이터 정의/정합성 진단 (사용자 질문 검증, 전체 모집단 기준)
# =====================================================================

def step23_data_definition_checks(df: pd.DataFrame, cols: dict, ym_norm_col: str, midx_col: str):
    section("STEP 23. 데이터 정의/정합성 진단 (전체 모집단 기준)")

    id_col = cols["id_col"]

    # (Q2) 업종/시도가 법인별로 36개월 동안 고정되는가
    print("[진행] 업종_대분류/업종_중분류/사업장_시도의 법인별 시간불변성 확인 중...")
    for role in ["업종대분류", "업종중분류", "시도"]:
        colname = cols.get(role)
        if not colname:
            print(f"[경고] {role} 컬럼 없음, 생략.")
            continue
        nun = df.groupby(id_col)[colname].nunique(dropna=True)
        n_changed = int((nun > 1).sum())
        print(f"\n[{role}={colname}] 법인별 고유값 개수 분포")
        print(nun.describe().to_string())
        print(f"관측기간 중 값이 바뀐(고유값>=2) 법인 수: {n_changed:,} "
              f"({n_changed / len(nun) * 100:.2f}%)")

    # (Q3/Q4) 총액 컬럼 = 하위 항목 합계인가
    def check_sum(label: str, total_role: str, sub_roles: list):
        total_col = cols.get(total_role)
        missing_sub = [r for r in sub_roles if r not in cols]
        sub_cols = [cols[r] for r in sub_roles if r in cols]
        if not total_col or missing_sub:
            print(f"\n[경고] {label} 검증 생략 (없는 컬럼: "
                  f"{missing_sub if missing_sub else [total_role]}).")
            return
        print(f"\n[진행] {label} = 하위항목 합계 검증 중...")
        diff = df[total_col] - df[sub_cols].sum(axis=1)
        print(_describe_series(diff).to_string())
        match_ratio = (diff.abs() < 1e-6).mean() * 100
        print(f"완전 일치 비율(|총액-하위합|<1e-6): {match_ratio:.2f}%")

    check_sum("여신_운전자금대출잔액", "운전자금", WORKING_CAPITAL_SUBITEMS)
    check_sum("여신_시설자금대출잔액", "시설자금", FACILITY_SUBITEMS)

    # (Q5) Δ요구불잔액 == 요구불입금 - 요구불출금 인가
    balance_col = cols.get("요구불")
    deposit_col = cols.get("요구불입금")
    withdraw_col = cols.get("요구불출금")
    if balance_col and deposit_col and withdraw_col:
        print("\n[진행] Δ요구불잔액(t) vs (요구불입금-요구불출금)(t) 정합성 확인 중 (valid monthly pair만)...")
        tmp = df[[id_col, midx_col, balance_col, deposit_col, withdraw_col]].dropna(subset=[midx_col]).copy()
        tmp["_net_flow_"] = tmp[deposit_col] - tmp[withdraw_col]

        cur = tmp[[id_col, midx_col, balance_col, "_net_flow_"]]
        prev = tmp[[id_col, midx_col, balance_col]].copy()
        prev[midx_col] = prev[midx_col] + 1
        prev = prev.rename(columns={balance_col: "_prev_balance_"})
        merged = cur.merge(prev, on=[id_col, midx_col], how="inner")

        merged["_delta_balance_"] = merged[balance_col] - merged["_prev_balance_"]
        merged["_recon_diff_"] = merged["_delta_balance_"] - merged["_net_flow_"]

        print(_describe_series(merged["_recon_diff_"]).to_string())
        match_ratio = (merged["_recon_diff_"].abs() < 1.0).mean() * 100
        print(f"Δ잔액 = 순유입 일치 비율(|오차|<1, 백만원 단위 반올림 감안): {match_ratio:.2f}%")
    else:
        print("\n[경고] 요구불잔액/입금/출금 컬럼 중 일부가 없어 Δ잔액 정합성 확인 생략.")

    # (Q7) 공백률 정의 두 가지를 전체 모집단 기준으로 비교
    print("\n[진행] 공백률 정의별 비교 (전체 모집단, 지역/업종 제한 없음)...")
    months_per_firm_all = df.groupby(id_col)[ym_norm_col].nunique()
    total_months_all = df[ym_norm_col].nunique()
    n_firms_all = months_per_firm_all.shape[0]

    cell_gap_ratio = (1 - len(df) / (n_firms_all * total_months_all)) * 100
    firm_any_gap_ratio = (months_per_firm_all < total_months_all).mean() * 100

    print(f"전체 법인수={n_firms_all:,}, 전체 기준월수={total_months_all}")
    print(f"정의(A) 셀 단위 공백률 [1 - 총행수/(법인수*월수)]: {cell_gap_ratio:.2f}%")
    print(f"정의(B) 법인 단위 '하나라도 공백 있음' 비율 "
          f"[관측월수<{total_months_all}인 법인 비율]: {firm_any_gap_ratio:.2f}%")
    print("(참고: STEP7의 31.56%는 대구·경북·비금융 분석표본 한정 셀단위 공백률로, "
          "위 두 수치와는 표본이 다름.)")


# =====================================================================
# 공용 헬퍼 (STEP 6~10에서 사용)
# =====================================================================

def get_firm_mode(df: pd.DataFrame, id_col: str, col: str) -> pd.Series:
    """법인별 최빈값(카테고리형 컬럼의 대표값)."""
    return df.groupby(id_col)[col].agg(
        lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan
    )


def compute_exposure_masks(df: pd.DataFrame, id_col: str, export_col: str, import_col: str):
    """법인 단위 수출노출/외환노출/대조군 boolean Series (index=id_col) 반환."""
    agg_dict = {export_col: "max"}
    if import_col:
        agg_dict[import_col] = "max"
    firm_level = df.groupby(id_col).agg(agg_dict)

    export_exposed = firm_level[export_col].fillna(0) > 0
    if import_col:
        fx_exposed = (firm_level[export_col].fillna(0) > 0) | (firm_level[import_col].fillna(0) > 0)
    else:
        fx_exposed = export_exposed.copy()
    control = ~fx_exposed
    return export_exposed, fx_exposed, control


def count_valid_pairs(df: pd.DataFrame, id_col: str, midx_col: str, lag: int) -> int:
    """값과 무관하게, (법인, 월인덱스) 관측이 정확히 lag개월 전에도 존재하는 쌍의 개수."""
    keys = df[[id_col, midx_col]].dropna().drop_duplicates()
    shifted = keys.copy()
    shifted[midx_col] = shifted[midx_col] + lag
    merged = keys.merge(shifted, on=[id_col, midx_col], how="inner")
    return int(len(merged))


def compute_diff_stats(df: pd.DataFrame, id_col: str, midx_col: str, value_col: str,
                        lag: int = 1, use_log: bool = True, eps: float = NO_VARIATION_EPS) -> dict:
    """정확히 lag개월 전 관측이 존재하는 쌍에서만 차분 계산.
    use_log=True면 ln(x+1) 차분(x<0은 0으로 clip), False면 원값 차분(음수 가능 변수용)."""
    tmp = df[[id_col, midx_col, value_col]].dropna(subset=[midx_col, value_col]).copy()
    empty_result = {"valid_pairs": 0, "no_variation_ratio(%)": np.nan, "std": np.nan, "IQR": np.nan}
    if tmp.empty:
        return empty_result

    if use_log:
        tmp["_val_"] = np.log1p(tmp[value_col].clip(lower=0))
    else:
        tmp["_val_"] = tmp[value_col]

    cur = tmp[[id_col, midx_col, "_val_"]]
    prev = cur.copy()
    prev[midx_col] = prev[midx_col] + lag
    prev = prev.rename(columns={"_val_": "_prev_val_"})

    merged = cur.merge(prev, on=[id_col, midx_col], how="inner")
    if merged.empty:
        return empty_result

    diff = merged["_val_"] - merged["_prev_val_"]
    absdiff = diff.abs()
    no_var_ratio = (absdiff < eps).mean() * 100
    q25, q75 = diff.quantile([.25, .75])
    return {
        "valid_pairs": int(len(merged)),
        "no_variation_ratio(%)": round(no_var_ratio, 2),
        "std": round(diff.std(), 4),
        "IQR": round(q75 - q25, 4),
    }


# =====================================================================
# STEP 6. 분석 표본 확정
# =====================================================================

def step6_analysis_sample(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    section("STEP 6. 분석 표본 확정")

    id_col = cols["id_col"]
    sido_col = cols.get("시도")
    daebun_col = cols.get("업종대분류")

    if sido_col:
        firm_sido = get_firm_mode(df, id_col, sido_col)
        print("\n[6a] 사업장_시도 값별 고유 법인수")
        print(firm_sido.value_counts(dropna=False).to_string())
    else:
        firm_sido = None
        print("\n[경고] 시도 컬럼이 없어 6a 생략.")

    if daebun_col:
        firm_daebun = get_firm_mode(df, id_col, daebun_col)
        print("\n[6b] 업종_대분류 값별 고유 법인수")
        print(firm_daebun.value_counts(dropna=False).to_string())
    else:
        firm_daebun = None
        print("\n[경고] 업종대분류 컬럼이 없어 6b 생략.")

    # 6c. 표본 필터
    all_ids = set(df[id_col].unique())

    if firm_sido is not None:
        region_mask = firm_sido.astype(str).str.contains("대구|경북|경상북도", na=False)
        region_ok_ids = set(firm_sido[region_mask].index)
        if region_mask.sum() == 0:
            print("\n[경고] '대구'/'경북' 문자열과 매칭되는 법인이 없습니다. 6a 결과를 보고 조건을 다시 확인하세요.")
    else:
        region_ok_ids = all_ids
        print("\n[경고] 시도 컬럼 부재로 지역 필터 미적용(전체 법인 통과).")

    if firm_daebun is not None:
        fin_mask = firm_daebun.astype(str).str.contains("금융|보험", na=False)
        if fin_mask.sum() == 0:
            print("[경고] '금융'/'보험' 문자열과 매칭되는 법인이 없습니다. 6b 결과를 보고 조건을 다시 확인하세요.")
        fin_ok_ids = set(firm_daebun[~fin_mask].index)
    else:
        fin_ok_ids = all_ids
        print("[경고] 업종대분류 컬럼 부재로 금융/보험 제외 필터 미적용.")

    keep_ids = region_ok_ids & fin_ok_ids
    sample_df = df[df[id_col].isin(keep_ids)].copy()

    if sample_df.empty:
        print("\n[경고] 분석 표본이 비어 있습니다. 필터 조건 및 COLUMN_MAP 라벨을 재확인하세요.")

    # 6d. 표본 요약
    export_col = cols.get("수출실적")
    import_col = cols.get("수입실적")

    print(f"\n[6d] 분석 표본(대구·경북 & 금융/보험 제외) 요약")
    print(f"행수: {len(sample_df):,}")
    print(f"고유 법인수: {sample_df[id_col].nunique():,}")

    if export_col and not sample_df.empty:
        export_exp, fx_exp, control = compute_exposure_masks(sample_df, id_col, export_col, import_col)
        total_firms = len(export_exp)
        summary = pd.DataFrame({
            "법인수": [export_exp.sum(), fx_exp.sum(), control.sum()],
            "전체대비비율(%)": [
                round(export_exp.sum() / total_firms * 100, 2),
                round(fx_exp.sum() / total_firms * 100, 2),
                round(control.sum() / total_firms * 100, 2),
            ],
        }, index=["수출노출", "외환노출(수출 또는 수입)", "대조군(외환실적 전무)"])
        print(summary.to_string())
    else:
        print("[경고] 수출실적 컬럼이 없거나 표본이 비어 노출군 집계 생략.")

    return sample_df


# =====================================================================
# STEP 7. 패널 연속성 진단 (분석 표본 대상)
# =====================================================================

def step7_panel_continuity(sample_df: pd.DataFrame, cols: dict, ym_norm_col: str, midx_col: str):
    section("STEP 7. 패널 연속성 진단 (분석 표본)")

    id_col = cols["id_col"]
    if sample_df.empty:
        print("[경고] 분석 표본이 비어 있어 STEP7 생략.")
        return

    months_per_firm = sample_df.groupby(id_col)[ym_norm_col].nunique()
    total_months = sample_df[ym_norm_col].nunique()

    print(f"기준 전체 월수(표본 내 고유 기준년월 수): {total_months}")
    print("\n[법인별 관측 개월수 분포]")
    print(months_per_firm.describe().to_string())

    print(f"\n36개월(균형) 법인수: {(months_per_firm == 36).sum():,}")
    print(f"30개월 이상 법인수: {(months_per_firm >= 30).sum():,}")
    print(f"24개월 이상 법인수: {(months_per_firm >= 24).sum():,}")

    n_firms = months_per_firm.shape[0]
    expected_cells = n_firms * total_months
    actual_cells = len(sample_df)
    gap_ratio = (1 - actual_cells / expected_cells) * 100 if expected_cells else np.nan
    print(f"\n관측 공백률 (1 - 실제행수/(법인수*기대개월수)): {gap_ratio:.2f}%")

    n_pair_1 = count_valid_pairs(sample_df, id_col, midx_col, lag=1)
    n_pair_12 = count_valid_pairs(sample_df, id_col, midx_col, lag=12)
    print(f"\n월간(1개월 전) valid_pair 수: {n_pair_1:,}")
    print(f"연간(12개월 전) valid_pair 수: {n_pair_12:,}")


# =====================================================================
# STEP 8. 계정별 반응성 비교 (분석 표본, 외환노출군 vs 대조군)
# =====================================================================

def step8_account_responsiveness(sample_df: pd.DataFrame, cols: dict, midx_col: str):
    section("STEP 8. 계정별 반응성 비교 (외환노출군 vs 대조군)")

    id_col = cols["id_col"]
    export_col = cols.get("수출실적")
    import_col = cols.get("수입실적")

    if sample_df.empty or not export_col:
        print("[경고] 표본이 비어있거나 수출실적 컬럼이 없어 STEP8 생략.")
        return

    _, fx_exposed, control = compute_exposure_masks(sample_df, id_col, export_col, import_col)
    exposed_ids = set(fx_exposed[fx_exposed].index)
    control_ids = set(control[control].index)

    df_exposed = sample_df[sample_df[id_col].isin(exposed_ids)]
    df_control = sample_df[sample_df[id_col].isin(control_ids)]

    target_vars = [k for k in ACCOUNT_VARS_STEP8 if k in cols]
    if not target_vars:
        print("[경고] 대상 계정(운전자금/시설자금/요구불/거치식/적립식) 컬럼이 없어 생략.")
        return

    rows_exposed, rows_control = [], []
    for role in target_vars:
        colname = cols[role]
        rows_exposed.append({"계정": role, **compute_diff_stats(df_exposed, id_col, midx_col, colname, lag=1)})
        rows_control.append({"계정": role, **compute_diff_stats(df_control, id_col, midx_col, colname, lag=1)})

    print(f"\n[외환노출군] 법인수={len(exposed_ids):,} - 월간 ln(x+1) 차분 기준")
    print(pd.DataFrame(rows_exposed).to_string(index=False))

    print(f"\n[대조군] 법인수={len(control_ids):,} - 월간 ln(x+1) 차분 기준")
    print(pd.DataFrame(rows_control).to_string(index=False))


# =====================================================================
# STEP 9. 요구불 흐름과 계절성 (분석 표본)
# =====================================================================

def step9_demand_deposit_flow(sample_df: pd.DataFrame, cols: dict, ym_norm_col: str, midx_col: str):
    section("STEP 9. 요구불 흐름과 계절성")

    id_col = cols["id_col"]
    deposit_col = cols.get("요구불입금")
    withdraw_col = cols.get("요구불출금")
    balance_col = cols.get("요구불")
    export_col = cols.get("수출실적")
    import_col = cols.get("수입실적")

    if sample_df.empty or not deposit_col or not withdraw_col:
        print("[경고] 표본이 비어있거나 요구불입금/출금 컬럼이 없어 STEP9 생략.")
        return

    net_flow = sample_df[deposit_col] - sample_df[withdraw_col]

    print("\n[9a] 순유입(요구불입금-요구불출금) 분포")
    print(_describe_series(net_flow).to_string())

    month_num = (sample_df[ym_norm_col] % 100)
    print("\n[9b] 월(1~12)별 평균 - 요구불잔액 / 순유입")
    tbl_rows = []
    for m in range(1, 13):
        m_mask = month_num == m
        row = {"월": m, "표본수": int(m_mask.sum())}
        if balance_col:
            row["요구불잔액_평균"] = round(sample_df.loc[m_mask, balance_col].mean(), 2)
        row["순유입_평균"] = round(net_flow.loc[m_mask].mean(), 2)
        tbl_rows.append(row)
    print(pd.DataFrame(tbl_rows).to_string(index=False))

    if export_col:
        tmp = sample_df[[id_col, midx_col]].copy()
        tmp["_net_flow_"] = net_flow
        _, fx_exposed, control = compute_exposure_masks(sample_df, id_col, export_col, import_col)
        exposed_ids = set(fx_exposed[fx_exposed].index)
        control_ids = set(control[control].index)

        tmp_exposed = tmp[tmp[id_col].isin(exposed_ids)]
        tmp_control = tmp[tmp[id_col].isin(control_ids)]

        r_exp = compute_diff_stats(tmp_exposed, id_col, midx_col, "_net_flow_", lag=1, use_log=False)
        r_ctl = compute_diff_stats(tmp_control, id_col, midx_col, "_net_flow_", lag=1, use_log=False)

        print("\n[9c] 순유입 월간 무변동 비율 (원값 차분 기준, 순유입은 음수 가능하여 로그 미적용)")
        print(pd.DataFrame([{"그룹": "외환노출군", **r_exp}, {"그룹": "대조군", **r_ctl}]).to_string(index=False))
    else:
        print("\n[경고] 수출실적 컬럼이 없어 9c(노출군 비교) 생략.")


# =====================================================================
# STEP 10. 내부 수출 시계열 재현 (분석 표본, 외환노출 법인)
# =====================================================================

def step10_export_timeseries(sample_df: pd.DataFrame, cols: dict, ym_norm_col: str):
    section("STEP 10. 내부 수출 시계열 재현")

    id_col = cols["id_col"]
    export_col = cols.get("수출실적")
    import_col = cols.get("수입실적")

    if sample_df.empty or not export_col:
        print("[경고] 표본이 비어있거나 수출실적 컬럼이 없어 STEP10 생략.")
        return

    _, fx_exposed, _ = compute_exposure_masks(sample_df, id_col, export_col, import_col)
    exposed_ids = set(fx_exposed[fx_exposed].index)
    df_exposed = sample_df[sample_df[id_col].isin(exposed_ids)]

    monthly_export_sum = df_exposed.groupby(ym_norm_col)[export_col].sum().sort_index()
    print(f"\n월별 수출실적 합계 시계열 (개월수={monthly_export_sum.shape[0]})")
    print(monthly_export_sum.describe().to_string())

    mean_v = monthly_export_sum.mean()
    std_v = monthly_export_sum.std()
    cv = std_v / mean_v if mean_v else np.nan
    print(f"\n평균: {mean_v:,.2f}, 표준편차: {std_v:,.2f}, 변동계수(CV=std/mean): {cv:.4f}")

    exporters_per_month = sample_df.loc[sample_df[export_col] > 0].groupby(ym_norm_col)[id_col].nunique()
    print("\n월별 '수출실적>0' 법인수 분포")
    print(exporters_per_month.describe().to_string())
    print(f"min={exporters_per_month.min()}, median={exporters_per_month.median()}, max={exporters_per_month.max()}")


# =====================================================================
# STEP 11~14 공용: 회귀용 데이터셋 구성 및 모형 추정 헬퍼
# =====================================================================

def build_regression_frames(sample_df: pd.DataFrame, cols: dict, ym_norm_col: str, midx_col: str):
    """dep_bal(요구불잔액 로그 월간차분)·dep_flow(순유입 asinh) 회귀용 프레임 생성.
    내부 컬럼명은 patsy 수식 안전성을 위해 영문(firm_id/year/month/exposure)으로 통일."""
    id_col = cols["id_col"]
    balance_col = cols.get("요구불")
    deposit_col = cols.get("요구불입금")
    withdraw_col = cols.get("요구불출금")
    export_col = cols.get("수출실적")
    import_col = cols.get("수입실적")

    if not export_col:
        print("[경고] 수출실적 컬럼이 없어 노출 더미를 만들 수 없습니다. STEP11~14 생략됩니다.")
        return None, None

    _, fx_exposed, _ = compute_exposure_masks(sample_df, id_col, export_col, import_col)
    exposure_map = fx_exposed.astype(int)

    dep_bal_frame = None
    if balance_col:
        tmp = sample_df[[id_col, midx_col, ym_norm_col, balance_col]].dropna(subset=[midx_col, balance_col]).copy()
        tmp["_log_bal_"] = np.log1p(tmp[balance_col].clip(lower=0))
        cur = tmp[[id_col, midx_col, ym_norm_col, "_log_bal_"]]
        prev = tmp[[id_col, midx_col, "_log_bal_"]].copy()
        prev[midx_col] = prev[midx_col] + 1
        prev = prev.rename(columns={"_log_bal_": "_prev_log_bal_"})
        merged = cur.merge(prev, on=[id_col, midx_col], how="inner")
        merged["dep_bal"] = merged["_log_bal_"] - merged["_prev_log_bal_"]

        dep_bal_frame = pd.DataFrame({
            "firm_id": merged[id_col].values,
            "year": (merged[ym_norm_col] // 100).astype(int).values,
            "month": (merged[ym_norm_col] % 100).astype(int).values,
            "dep_bal": merged["dep_bal"].values,
        })
        dep_bal_frame["exposure"] = dep_bal_frame["firm_id"].map(exposure_map).astype(int)
    else:
        print("[경고] 요구불예금잔액 컬럼이 없어 dep_bal 생성 생략.")

    dep_flow_frame = None
    if deposit_col and withdraw_col:
        flow_df = sample_df[[id_col, midx_col, ym_norm_col, deposit_col, withdraw_col]].dropna(
            subset=[deposit_col, withdraw_col]
        ).copy()
        net_flow = flow_df[deposit_col] - flow_df[withdraw_col]

        dep_flow_frame = pd.DataFrame({
            "firm_id": flow_df[id_col].values,
            "year": (flow_df[ym_norm_col] // 100).astype(int).values,
            "month": (flow_df[ym_norm_col] % 100).astype(int).values,
            "dep_flow": np.arcsinh(net_flow).values,
        })
        dep_flow_frame["exposure"] = dep_flow_frame["firm_id"].map(exposure_map).astype(int)
    else:
        print("[경고] 요구불입금/요구불출금 컬럼이 없어 dep_flow 생성 생략.")

    return dep_bal_frame, dep_flow_frame


def run_ols_cluster(formula: str, data: pd.DataFrame, cluster_col: str = "firm_id"):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})


def run_firm_fe_model(frame: pd.DataFrame, dep_col: str, cluster_col: str = "firm_id"):
    """법인 고정효과(within 추정): 반응변수·설계행렬을 법인평균으로 demean 후 OLS."""
    design = dmatrix("C(month) + C(year)", data=frame, return_type="dataframe")
    if "Intercept" in design.columns:
        design = design.drop(columns=["Intercept"])
    grp = frame[cluster_col]
    y = frame[dep_col]
    y_dm = y - y.groupby(grp).transform("mean")
    x_dm = design.sub(design.groupby(grp).transform("mean"), axis=0)
    return sm.OLS(y_dm, x_dm).fit(cov_type="cluster", cov_kwds={"groups": grp})


def print_coef_table(model, title: str, extra_note: str = ""):
    tbl = pd.DataFrame({
        "coef": model.params,
        "std err": model.bse,
        "p-value": model.pvalues,
    }).round(6)
    print(f"\n[{title}]")
    print(f"관측치 수: {int(model.nobs):,}")
    if extra_note:
        print(extra_note)
    print(tbl.to_string())


def print_exposure_effect_by_year(model, frame: pd.DataFrame, dep_label: str):
    """모형 C 계수에서 연도별 외환노출 총효과(기준연도=자체 계수, 이후연도=기준+상호작용)를 t_test로 합산."""
    years = sorted(frame["year"].dropna().unique().tolist())
    if not years:
        return
    base_year = years[0]
    rows = []
    for yr in years:
        if yr == base_year:
            contrast = "exposure"
        else:
            match = [c for c in model.params.index
                     if c.startswith("exposure:C(year)") and str(int(yr)) in c]
            if not match:
                continue
            contrast = f"exposure + {match[0]}"
        try:
            test = model.t_test(contrast)
            rows.append({
                "연도": int(yr),
                "효과(coef)": round(float(np.ravel(test.effect)[0]), 6),
                "std err": round(float(np.ravel(test.sd)[0]), 6),
                "p-value": round(float(np.ravel(test.pvalue)[0]), 6),
            })
        except Exception as e:  # noqa: BLE001
            print(f"[경고] {yr}년 외환노출 효과 계산 실패: {e}")

    if rows:
        print(f"\n[{dep_label}] 연도별 외환노출 총효과 "
              f"({base_year}=자체계수, 이후연도={base_year} 대비 상호작용 합산)")
        print(pd.DataFrame(rows).to_string(index=False))


# =====================================================================
# STEP 11. 종속변수 기술통계 재확인
# =====================================================================

def step11_dep_var_stats(dep_bal_frame, dep_flow_frame):
    section("STEP 11. 종속변수 기술통계 재확인")

    if dep_bal_frame is not None:
        print("[진행] dep_bal 기술통계 계산 중...")
        print("\n[dep_bal = ln(요구불잔액+1) 월간 1차 차분, valid monthly pair만]")
        print(_describe_series(dep_bal_frame["dep_bal"]).to_string())
    else:
        print("[경고] dep_bal 프레임 없음, 생략.")

    if dep_flow_frame is not None:
        print("\n[진행] dep_flow 기술통계 계산 중...")
        print("\n[dep_flow = asinh(요구불입금 - 요구불출금)]")
        print(_describe_series(dep_flow_frame["dep_flow"]).to_string())

        q = dep_flow_frame["dep_flow"].quantile([.01, .99])
        print(f"\ndep_flow 1%/99% 경계값: {q.iloc[0]:.4f} / {q.iloc[1]:.4f}")
    else:
        print("[경고] dep_flow 프레임 없음, 생략.")


# =====================================================================
# STEP 12. 노출 여부에 따른 단순 비교
# =====================================================================

def _group_stats_by_exposure(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for label, mask in [("외환노출", frame["exposure"] == 1), ("대조군", frame["exposure"] == 0)]:
        s = frame.loc[mask, value_col]
        rows.append({
            "그룹": label,
            "n": len(s),
            "mean": round(s.mean(), 6),
            "std": round(s.std(), 6),
            "median": round(s.median(), 6),
            "no_variation_ratio(%)": round((s.abs() < NO_VARIATION_EPS).mean() * 100, 2) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def step12_exposure_comparison(dep_bal_frame, dep_flow_frame):
    section("STEP 12. 노출 여부에 따른 단순 비교")
    print("[진행] 그룹별 비교 통계 계산 중...")

    if dep_bal_frame is not None:
        print("\n[dep_bal] 외환노출 vs 대조군")
        print(_group_stats_by_exposure(dep_bal_frame, "dep_bal").to_string(index=False))

        pivot = dep_bal_frame.groupby(["year", "exposure"])["dep_bal"].mean().unstack("exposure")
        pivot.columns = [("노출" if c == 1 else "대조군") for c in pivot.columns]
        print("\n[연도 × 노출여부] dep_bal 평균 피벗")
        print(pivot.round(6).to_string())
    else:
        print("[경고] dep_bal 프레임 없음, 생략.")

    if dep_flow_frame is not None:
        print("\n[dep_flow] 외환노출 vs 대조군")
        print(_group_stats_by_exposure(dep_flow_frame, "dep_flow").to_string(index=False))
    else:
        print("[경고] dep_flow 프레임 없음, 생략.")


# =====================================================================
# STEP 13. 패널 고정효과 회귀 (dep_bal)
# =====================================================================

def step13_fe_regressions(dep_bal_frame):
    section("STEP 13. 패널 고정효과 회귀 (dep_bal)")

    if dep_bal_frame is None:
        print("[경고] dep_bal 프레임 없음, STEP13 생략.")
        return

    print("[진행] 모형 A 추정 중 (OLS + 법인 클러스터 SE, 표본이 커서 시간이 걸릴 수 있습니다)...")
    model_a = run_ols_cluster("dep_bal ~ exposure + C(month) + C(year)", dep_bal_frame)
    print_coef_table(
        model_a,
        "모형 A: dep_bal ~ 외환노출 + C(월) + C(연도)  (법인고정효과 없음, 클러스터 SE)",
    )

    print("\n[진행] 모형 B 추정 중 (법인 고정효과, within 추정)...")
    model_b = run_firm_fe_model(dep_bal_frame, "dep_bal")
    print_coef_table(
        model_b,
        "모형 B: dep_bal ~ C(월) + C(연도) + 법인고정효과",
        extra_note="※ 외환노출은 법인불변 변수라 법인고정효과에 흡수되어 주효과 계수는 존재하지 않음(설계상 정상).",
    )

    print("\n[진행] 모형 C 추정 중 (연도×노출 상호작용, 핵심 모형)...")
    model_c = run_ols_cluster("dep_bal ~ exposure * C(year) + C(month)", dep_bal_frame)
    print_coef_table(
        model_c,
        "모형 C: dep_bal ~ 외환노출 * C(연도) + C(월)  (법인고정효과 없음, 클러스터 SE)",
    )
    print_exposure_effect_by_year(model_c, dep_bal_frame, "dep_bal")


# =====================================================================
# STEP 14. dep_flow로 동일 반복 (모형 C)
# =====================================================================

def step14_dep_flow_regression(dep_flow_frame):
    section("STEP 14. dep_flow로 동일 반복 (모형 C)")

    if dep_flow_frame is None:
        print("[경고] dep_flow 프레임 없음, STEP14 생략.")
        return

    print("[진행] dep_flow 모형 C 추정 중 (표본이 커서 시간이 걸릴 수 있습니다)...")
    model_c = run_ols_cluster("dep_flow ~ exposure * C(year) + C(month)", dep_flow_frame)
    print_coef_table(
        model_c,
        "모형 C (dep_flow): dep_flow ~ 외환노출 * C(연도) + C(월)  (법인고정효과 없음, 클러스터 SE)",
    )
    print_exposure_effect_by_year(model_c, dep_flow_frame, "dep_flow")

    print("\n(참고: dep_bal 결과와 부호·유의성을 대조해 임계효과 가설의 1차 증거로 판단할 것)")


# =====================================================================
# STEP 15~19 공용 헬퍼
# =====================================================================

def ym_to_midx(ym_series: pd.Series) -> pd.Series:
    return (ym_series // 100) * 12 + (ym_series % 100)


def merge_lag_same_df(df: pd.DataFrame, group_cols: list, midx_col: str,
                       value_col: str, lag: int, new_col: str):
    """같은 df 안에서 group_cols 기준, midx_col이 정확히 lag 이전인 행의 value_col 값을 붙여 반환."""
    src = df[group_cols + [midx_col, value_col]].copy()
    src[midx_col] = src[midx_col] + lag
    src = src.rename(columns={value_col: new_col})
    merged = df[group_cols + [midx_col]].merge(src, on=group_cols + [midx_col], how="left")
    return merged[new_col].values


def lookup_external_value(internal_df: pd.DataFrame, region_col: str, midx_col: str,
                           external_df: pd.DataFrame, ext_region_col: str, ext_midx_col: str,
                           ext_value_col: str, lag: int):
    """internal_df의 (region, midx-lag) 키로 external_df에서 값을 조회해 반환 (internal 행순서 유지)."""
    key = internal_df[[region_col, midx_col]].copy()
    key["_lookup_midx_"] = key[midx_col] - lag
    ext_small = external_df[[ext_region_col, ext_midx_col, ext_value_col]].rename(
        columns={ext_region_col: region_col, ext_midx_col: "_lookup_midx_"}
    )
    merged = key.merge(ext_small, on=[region_col, "_lookup_midx_"], how="left")
    return merged[ext_value_col].values


def parse_kstat_file(path: str, region: str) -> pd.DataFrame:
    """K-stat(한국무역협회) 수출입 통계 xls 파싱.
    헤더가 표 형태가 아니라 '2025년' 같은 연도 소계 행과 '1월'~'12월' 데이터 행이
    섞여 있는 리포트 레이아웃이라, 연도 행을 만나면 cur_year를 갱신하고
    월 행을 만나면 그 연도의 월 데이터를 뽑아낸다."""
    raw = pd.read_excel(path, header=None, engine="xlrd")
    rows = []
    cur_year = None
    for _, r in raw.iterrows():
        c0 = str(r[0]).strip()

        if c0.endswith("년") and c0[:-1].strip().isdigit():
            cur_year = int(c0[:-1].strip())
            continue

        digits = "".join(ch for ch in c0 if ch.isdigit())
        if c0.endswith("월") and digits.isdigit() and cur_year is not None:
            month = int(digits)
            if 1 <= month <= 12:
                def num(x):
                    try:
                        return float(str(x).replace(",", "").strip())
                    except (ValueError, TypeError):
                        return np.nan

                rows.append({
                    "region": region,
                    "ym": cur_year * 100 + month,
                    "export_usd": num(r[1]),
                    "exp_yoy_pct": num(r[2]),
                    "import_usd": num(r[5]),
                    "imp_yoy_pct": num(r[6]),
                })
    return pd.DataFrame(rows)


def attach_region_and_ym(frame: pd.DataFrame, sample_df: pd.DataFrame, cols: dict):
    """dep_bal_frame/dep_flow_frame(firm_id/year/month 보유)에 region(대구/경북 정규화)·ym·midx 컬럼 부착."""
    id_col = cols["id_col"]
    sido_col = cols.get("시도")
    if not sido_col:
        print("[경고] 시도 컬럼이 없어 region 매핑 불가.")
        return None

    firm_region_raw = get_firm_mode(sample_df, id_col, sido_col)

    def norm_region(x):
        s = str(x)
        if "대구" in s:
            return "대구"
        if "경북" in s or "경상북도" in s:
            return "경북"
        return np.nan

    firm_region = firm_region_raw.map(norm_region)

    out = frame.copy()
    out["region"] = out["firm_id"].map(firm_region)
    out["ym"] = out["year"] * 100 + out["month"]
    out["midx"] = out["year"] * 12 + out["month"]

    n_missing_region = out["region"].isna().sum()
    if n_missing_region > 0:
        print(f"[경고] region 라벨 매핑 실패 {n_missing_region}건 (분석표본이 대구/경북로 이미 필터되어 있어야 함).")
    return out


def region_month_cells(sub_indexed: pd.DataFrame) -> int:
    """(firm_id, ym) MultiIndex를 가진 프레임에서 고유 (region, ym) 셀 수."""
    tmp = sub_indexed.reset_index()
    return tmp[["region", "ym"]].drop_duplicates().shape[0]


def build_month_year_dummies(sub: pd.DataFrame) -> pd.DataFrame:
    month_dum = pd.get_dummies(sub["month"], prefix="month", drop_first=True).astype(float)
    year_dum = pd.get_dummies(sub["year"], prefix="year", drop_first=True).astype(float)
    return pd.concat([month_dum, year_dum], axis=1)


def fit_panel_spec_a(frame: pd.DataFrame, dep_col: str, lag_col: str):
    """모형 A: dep ~ exp_yoy_lagk + exp_yoy_lagk*exposure + C(월)+C(연도), 법인FE, 클러스터SE."""
    sub = frame.dropna(subset=[lag_col, dep_col]).drop_duplicates(subset=["firm_id", "ym"]).copy()
    sub = sub.set_index(["firm_id", "ym"])

    exog = pd.DataFrame({
        lag_col: sub[lag_col],
        f"{lag_col}_x_exposure": sub[lag_col] * sub["exposure"],
    })
    exog = pd.concat([exog, build_month_year_dummies(sub)], axis=1)

    y = sub[dep_col]
    model = PanelOLS(y, exog, entity_effects=True, drop_absorbed=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return res, sub


def fit_panel_spec_b(frame: pd.DataFrame, dep_col: str, lag_col: str, cluster_mode: str = "firm"):
    """모형 B: dep ~ exp_yoy_lagk*exposure, 법인FE+연월FE.
    cluster_mode: 'firm'(기본, 법인 클러스터) / 'region_month'(region×ym 셀 클러스터) / 'region'(region 클러스터, 2개뿐)."""
    sub = frame.dropna(subset=[lag_col, dep_col]).drop_duplicates(subset=["firm_id", "ym"]).copy()
    sub = sub.set_index(["firm_id", "ym"])

    exog = pd.DataFrame({
        lag_col: sub[lag_col],
        f"{lag_col}_x_exposure": sub[lag_col] * sub["exposure"],
    })

    y = sub[dep_col]
    model = PanelOLS(y, exog, entity_effects=True, time_effects=True, drop_absorbed=True)

    if cluster_mode == "firm":
        res = model.fit(cov_type="clustered", cluster_entity=True)
    elif cluster_mode == "region_month":
        cluster_id = sub["region"].astype(str) + "_" + sub.index.get_level_values("ym").astype(str)
        res = model.fit(cov_type="clustered", clusters=cluster_id)
    elif cluster_mode == "region":
        cluster_id = sub["region"].astype(str)
        res = model.fit(cov_type="clustered", clusters=cluster_id)
    else:
        raise ValueError(f"알 수 없는 cluster_mode: {cluster_mode}")

    return res, sub


def get_coef_row(res, term: str) -> dict:
    if term not in res.params.index:
        return {"coef": np.nan, "std err": np.nan, "p-value": np.nan}
    return {
        "coef": round(float(res.params[term]), 6),
        "std err": round(float(res.std_errors[term]), 6),
        "p-value": round(float(res.pvalues[term]), 6),
    }


# =====================================================================
# STEP 15. 외부 데이터 검증·변환
# =====================================================================

def step15_load_external_export_data(script_dir: str):
    section("STEP 15. 외부 데이터 검증·변환")

    frames = []
    for filename, region_label in EXTERNAL_FILES.items():
        path = os.path.join(EXTERNAL_DIR, filename)
        if not os.path.exists(path):
            print(f"[경고] 파일 없음: {filename}")
            continue

        print(f"[진행] {region_label} 외부 데이터(K-stat) 파싱 중: {path}")
        sub = parse_kstat_file(path, region_label)

        print(f"  [{region_label}] 파싱 결과: {len(sub)}행")
        if sub.empty:
            print(f"[경고] {region_label}: 파싱된 행이 없습니다. 파일 레이아웃이 예상과 다를 수 있습니다.")
            continue

        ym_year = sub["ym"] // 100
        print(f"  ym 범위: {sub['ym'].min()} ~ {sub['ym'].max()}")
        for y in [2022, 2023, 2024, 2025]:
            n = int((ym_year == y).sum())
            status = "OK" if n == 12 else "WARN(부족)"
            print(f"    {y}년 월수: {n}  [{status}]")
        print(f"  export_usd 결측: {sub['export_usd'].isna().sum()}, "
              f"import_usd 결측: {sub['import_usd'].isna().sum()}")

        sub["ym"] = sub["ym"].astype(int)
        frames.append(sub)

    if not frames:
        print("\n[경고] 유효하게 로딩된 외부 데이터가 없습니다. STEP15~19를 진행할 수 없습니다.")
        return None

    ext = pd.concat(frames, ignore_index=True)

    print("\n[검증] region 고유값별 행수")
    print(ext["region"].value_counts(dropna=False).to_string())
    expected_regions = {"대구", "경북"}
    actual_regions = set(ext["region"].dropna().unique().tolist())
    if actual_regions != expected_regions:
        print(f"[경고] region 고유값이 기대값과 다릅니다. 기대={expected_regions}, 실제={actual_regions}")

    ym_min, ym_max, n_unique_ym = ext["ym"].min(), ext["ym"].max(), ext["ym"].nunique()
    print(f"\nym 범위: {ym_min} ~ {ym_max}, 고유 개월수: {n_unique_ym}")
    expected_yms = {y * 100 + m for y in range(2022, 2026) for m in range(1, 13)}
    if ym_min != 202201 or ym_max != 202512 or n_unique_ym != 48:
        print("[경고] ym 범위/개월수가 기대(202201~202512, 48개월)와 다릅니다.")
    for region, g in ext.groupby("region"):
        region_yms = set(g["ym"].unique().tolist())
        n_diff = len(expected_yms.symmetric_difference(region_yms))
        print(f"  region={region}: 보유 개월수={len(region_yms)}, 기대(48개월) 대비 불일치 개월수={n_diff}")

    dup_count = int(ext.duplicated(subset=["region", "ym"]).sum())
    print(f"\nregion×ym 중복 건수: {dup_count}")

    need_set = {(r, y * 100 + m) for r in ["대구", "경북"] for y in [2023, 2024, 2025] for m in range(1, 13)}
    have_set = set(zip(ext["region"], ext["ym"]))
    missing_analysis_period = need_set - have_set
    print(f"분석기간(2023~2025, 대구·경북 72개 셀) 중 누락: {len(missing_analysis_period)}개")
    if missing_analysis_period:
        print(f"  누락 예시: {sorted(missing_analysis_period)[:6]}")

    ext = ext.sort_values(["region", "ym"]).reset_index(drop=True)
    ext["_midx_"] = ym_to_midx(ext["ym"])

    for src_col, in [("export_usd",), ("import_usd",)]:
        n_nonpos = (ext[src_col] <= 0).sum()
        if n_nonpos > 0:
            warnings.warn(f"{src_col}: 0 이하 값 {n_nonpos}건 발견, log 계산 위해 최소값으로 clip")

    print("\n[진행] region별 전년동월(12개월전) 로그증감률(exp_yoy/imp_yoy) 계산 중...")
    for value_col, new_col in [("export_usd", "exp_yoy"), ("import_usd", "imp_yoy")]:
        log_col = f"_log_{new_col}_"
        ext[log_col] = np.log(ext[value_col].clip(lower=1e-9))
        lag_vals = merge_lag_same_df(ext, ["region"], "_midx_", log_col, lag=12, new_col=f"_prev_{log_col}")
        ext[new_col] = ext[log_col].values - lag_vals

    n_valid_by_region = ext.dropna(subset=["exp_yoy"]).groupby("region").size()
    print("\n[진행] region별 exp_yoy 유효 개수 (기대: 각 36개월, 202301~202512)")
    print(n_valid_by_region.to_string())

    print("\n[exp_yoy region별 describe]")
    desc = ext.dropna(subset=["exp_yoy"]).groupby("region")["exp_yoy"].describe(
        percentiles=[.01, .05, .25, .5, .75, .9, .95, .99]
    )
    print(desc.to_string())

    if "exp_yoy_pct" in ext.columns:
        check = ext.dropna(subset=["exp_yoy", "exp_yoy_pct"]).copy()
        check["exp_yoy_pct_from_log"] = (np.exp(check["exp_yoy"]) - 1) * 100
        check["diff_pct_point"] = check["exp_yoy_pct_from_log"] - check["exp_yoy_pct"]
        print("\n[검증] 자체 계산 exp_yoy(로그) vs K-stat 제공 exp_yoy_pct(원자료 증감률) 일치도")
        print(f"  비교 가능 행수: {len(check)}, 평균 절대오차(%p): {check['diff_pct_point'].abs().mean():.4f}, "
              f"최대 절대오차(%p): {check['diff_pct_point'].abs().max():.4f}")

    return ext


# =====================================================================
# STEP 16. 내부-외부 결합
# =====================================================================

def step16_merge_external(dep_bal_frame, sample_df: pd.DataFrame, cols: dict, ext_df):
    section("STEP 16. 내부-외부 결합 (dep_bal)")

    if dep_bal_frame is None or ext_df is None:
        print("[경고] dep_bal 프레임 또는 외부데이터가 없어 STEP16 생략.")
        return None

    print("[진행] region 매핑 및 시차별(k=0~3) exp_yoy/imp_yoy 병합 중...")
    frame = attach_region_and_ym(dep_bal_frame, sample_df, cols)
    if frame is None:
        return None

    n_before = len(frame)
    for k in [0, 1, 2, 3]:
        frame[f"exp_yoy_lag{k}"] = lookup_external_value(
            frame, "region", "midx", ext_df, "region", "_midx_", "exp_yoy", lag=k
        )
        frame[f"imp_yoy_lag{k}"] = lookup_external_value(
            frame, "region", "midx", ext_df, "region", "_midx_", "imp_yoy", lag=k
        )
    n_after = len(frame)

    print(f"\n병합 전 행수: {n_before:,}, 병합 후 행수: {n_after:,}, "
          f"증가여부: {'예(문제)' if n_after > n_before else '아니오(정상)'}")

    rows = []
    for k in [0, 1, 2, 3]:
        col = f"exp_yoy_lag{k}"
        n_missing = int(frame[col].isna().sum())
        rows.append({
            "lag": k,
            "미결합_행수": n_missing,
            "미결합_비율(%)": round(n_missing / n_after * 100, 2) if n_after else np.nan,
            "유효_행수": n_after - n_missing,
        })
    print("\n[시차별 exp_yoy 병합 진단]")
    print(pd.DataFrame(rows).to_string(index=False))

    n_cells = frame.dropna(subset=["region"]).drop_duplicates(subset=["region", "ym"]).shape[0]
    print(f"\nregion×month 고유 셀 수: {n_cells}")

    return frame


# =====================================================================
# STEP 17. 진짜 β3 회귀 (시차별 전부 보고)
# =====================================================================

def step17_beta3_regressions(merged_frame, dep_label: str):
    section(f"STEP 17. 진짜 β3 회귀 ({dep_label}, 시차별 전부 보고)")

    if merged_frame is None:
        print("[경고] 병합 프레임 없음, STEP17 생략.")
        return []

    rows_a, rows_b = [], []
    for k in [0, 1, 2, 3]:
        lag_col = f"exp_yoy_lag{k}"

        print(f"[진행] lag={k} 모형 A 추정 중 (법인FE + 클러스터SE)...")
        try:
            res_a, sub_a = fit_panel_spec_a(merged_frame, dep_label, lag_col)
            c1 = get_coef_row(res_a, lag_col)
            c3 = get_coef_row(res_a, f"{lag_col}_x_exposure")
            rows_a.append({
                "lag": k, "n_obs": int(res_a.nobs),
                "n_firm": sub_a.index.get_level_values(0).nunique(),
                "region_month_cells": region_month_cells(sub_a),
                "beta1(exp_yoy)_coef": c1["coef"], "beta1_std_err": c1["std err"], "beta1_p": c1["p-value"],
                "beta3(교호작용)_coef": c3["coef"], "beta3_std_err": c3["std err"], "beta3_p": c3["p-value"],
            })
        except Exception as e:  # noqa: BLE001
            print(f"[경고] lag={k} 모형 A 추정 실패: {e}")

        print(f"[진행] lag={k} 모형 B 추정 중 (법인FE + 연월FE + 클러스터SE)...")
        try:
            res_b, sub_b = fit_panel_spec_b(merged_frame, dep_label, lag_col)
            c3b = get_coef_row(res_b, f"{lag_col}_x_exposure")
            t_val = float(res_b.tstats[f"{lag_col}_x_exposure"])
            rows_b.append({
                "lag": k, "n_obs": int(res_b.nobs),
                "n_firm": sub_b.index.get_level_values(0).nunique(),
                "region_month_cells": region_month_cells(sub_b),
                "beta3_coef": c3b["coef"], "beta3_std_err": c3b["std err"], "beta3_p": c3b["p-value"],
                "abs_t": round(abs(t_val), 4),
            })
        except Exception as e:  # noqa: BLE001
            print(f"[경고] lag={k} 모형 B 추정 실패: {e}")

    print(f"\n[{dep_label} - 모형 A] dep ~ exp_yoy_lagk + exp_yoy_lagk*exposure + C(월)+C(연도), 법인FE, 클러스터SE")
    print(pd.DataFrame(rows_a).to_string(index=False) if rows_a else "(결과 없음)")

    print(f"\n[{dep_label} - 모형 B] dep ~ exp_yoy_lagk*exposure, 법인FE+연월FE, 클러스터SE")
    print(pd.DataFrame(rows_b).to_string(index=False) if rows_b else "(결과 없음)")

    return rows_b


# =====================================================================
# STEP 18. 조건부성 결정 테스트
# =====================================================================

def run_conditionality_test(merged_frame, rows_b: list, dep_label: str):
    if not rows_b:
        print("[경고] 모형 B 결과 없음, 조건부성 테스트 생략.")
        return

    valid_rows = [r for r in rows_b if not pd.isna(r.get("abs_t"))]
    if not valid_rows:
        print("[경고] 유효한 |t| 값이 없어 k* 선택 불가.")
        return
    best = max(valid_rows, key=lambda r: r["abs_t"])
    k_star = best["lag"]
    lag_col = f"exp_yoy_lag{k_star}"
    print(f"[진행] |t| 최대 시차 k*={k_star} (|t|={best['abs_t']:.4f}) 선택, 표본별 재추정 중...")

    samples = {
        "전체표본": merged_frame,
        "2023제외": merged_frame[merged_frame["year"] != 2023],
        "2023만": merged_frame[merged_frame["year"] == 2023],
    }

    rows = []
    for label, sub_frame in samples.items():
        try:
            res_b, sub = fit_panel_spec_b(sub_frame, dep_label, lag_col)
            c3b = get_coef_row(res_b, f"{lag_col}_x_exposure")
            rows.append({
                "표본": label, "k*": k_star,
                "n_obs": int(res_b.nobs), "n_firm": sub.index.get_level_values(0).nunique(),
                "beta3_coef": c3b["coef"], "beta3_std_err": c3b["std err"], "beta3_p": c3b["p-value"],
            })
        except Exception as e:  # noqa: BLE001
            print(f"[경고] '{label}' 표본 추정 실패: {e}")
            rows.append({"표본": label, "k*": k_star, "n_obs": np.nan, "n_firm": np.nan,
                         "beta3_coef": np.nan, "beta3_std_err": np.nan, "beta3_p": np.nan})

    print(f"\n[{dep_label}] k*={k_star} 기준 표본별 β3 비교 (전체 vs 2023제외 vs 2023만)")
    print(pd.DataFrame(rows).to_string(index=False))


def step18_conditionality_test(merged_frame, rows_b: list, dep_label: str = "dep_bal"):
    section(f"STEP 18. 조건부성 결정 테스트 ({dep_label})")
    run_conditionality_test(merged_frame, rows_b, dep_label)


# =====================================================================
# STEP 19. dep_flow로 [B]·STEP18 반복
# =====================================================================

def step19_dep_flow_repeat(dep_flow_frame, sample_df: pd.DataFrame, cols: dict, ext_df):
    section("STEP 19. dep_flow로 모형 [B] 및 조건부성 테스트 반복")

    if dep_flow_frame is None or ext_df is None:
        print("[경고] dep_flow 프레임 또는 외부데이터 없음, STEP19 생략.")
        return

    print("[진행] dep_flow에 region/시차별 exp_yoy 병합 중...")
    frame = attach_region_and_ym(dep_flow_frame, sample_df, cols)
    if frame is None:
        return

    for k in [0, 1, 2, 3]:
        frame[f"exp_yoy_lag{k}"] = lookup_external_value(
            frame, "region", "midx", ext_df, "region", "_midx_", "exp_yoy", lag=k
        )

    rows_b = []
    for k in [0, 1, 2, 3]:
        lag_col = f"exp_yoy_lag{k}"
        print(f"[진행] dep_flow lag={k} 모형 B 추정 중...")
        try:
            res_b, sub_b = fit_panel_spec_b(frame, "dep_flow", lag_col)
            c3b = get_coef_row(res_b, f"{lag_col}_x_exposure")
            t_val = float(res_b.tstats[f"{lag_col}_x_exposure"])
            rows_b.append({
                "lag": k, "n_obs": int(res_b.nobs),
                "n_firm": sub_b.index.get_level_values(0).nunique(),
                "region_month_cells": region_month_cells(sub_b),
                "beta3_coef": c3b["coef"], "beta3_std_err": c3b["std err"], "beta3_p": c3b["p-value"],
                "abs_t": round(abs(t_val), 4),
            })
        except Exception as e:  # noqa: BLE001
            print(f"[경고] dep_flow lag={k} 모형 B 추정 실패: {e}")

    print("\n[dep_flow - 모형 B] dep_flow ~ exp_yoy_lagk*exposure, 법인FE+연월FE, 클러스터SE")
    print(pd.DataFrame(rows_b).to_string(index=False) if rows_b else "(결과 없음)")

    print("\n[진행] dep_flow 조건부성(2023 제외/한정) 테스트 중...")
    run_conditionality_test(frame, rows_b, dep_label="dep_flow")

    print("\n(참고: dep_bal STEP17~18 결과와 부호·유의성을 대조해 일관성을 판단할 것)")


# =====================================================================
# STEP 20. 지역클러스터 강건성
# =====================================================================

def step20_region_cluster_robustness(merged_frame, dep_label: str = "dep_bal", k: int = 2):
    section(f"STEP 20. 지역클러스터 강건성 ({dep_label}, k*={k} 모형 B)")

    if merged_frame is None:
        print("[경고] 병합 프레임 없음, STEP20 생략.")
        return

    lag_col = f"exp_yoy_lag{k}"
    specs = [
        ("firm", "(a) 법인 클러스터 (기존)"),
        ("region_month", "(b) region×month 클러스터 (가장 보수적)"),
        ("region", "(c) region 클러스터 (2개, 참고용)"),
    ]

    rows = []
    for cluster_mode, label in specs:
        print(f"[진행] {label} 추정 중...")
        try:
            res, sub = fit_panel_spec_b(merged_frame, dep_label, lag_col, cluster_mode=cluster_mode)
            c3 = get_coef_row(res, f"{lag_col}_x_exposure")

            if cluster_mode == "firm":
                n_clusters = sub.index.get_level_values(0).nunique()
            elif cluster_mode == "region_month":
                n_clusters = region_month_cells(sub)
            else:
                n_clusters = sub["region"].nunique()

            rows.append({
                "클러스터": label, "n_clusters": n_clusters,
                "beta3_coef": c3["coef"], "beta3_std_err": c3["std err"], "beta3_p": c3["p-value"],
            })
        except Exception as e:  # noqa: BLE001
            print(f"[경고] {label} 추정 실패: {e}")

    print(f"\n[{dep_label}] k*={k} 모형 B, 클러스터 기준별 β3 비교")
    print(pd.DataFrame(rows).to_string(index=False) if rows else "(결과 없음)")
    print("\n주의: (c) region 클러스터는 그룹 수가 단 2개뿐이라 클러스터-로버스트 SE의 점근적 타당성이 "
          "성립하지 않습니다. 참고용으로만 보고, 유의성 판단은 (a)/(b) 기준을 우선하세요.")


# =====================================================================
# STEP 21. 비선형(임계) 직접 검정
# =====================================================================

def fit_panel_spec_nonlinear(frame: pd.DataFrame, dep_col: str, lag_col: str):
    """dep ~ (neg+pos)*exposure + C(월)+C(연도), 법인FE, 클러스터SE(법인).
    neg = min(exp_yoy_lagk, 0), pos = max(exp_yoy_lagk, 0)."""
    sub = frame.dropna(subset=[lag_col, dep_col]).drop_duplicates(subset=["firm_id", "ym"]).copy()
    sub = sub.set_index(["firm_id", "ym"])

    neg = sub[lag_col].clip(upper=0)
    pos = sub[lag_col].clip(lower=0)
    exog = pd.DataFrame({
        "neg": neg,
        "pos": pos,
        "neg_x_exposure": neg * sub["exposure"],
        "pos_x_exposure": pos * sub["exposure"],
    })
    exog = pd.concat([exog, build_month_year_dummies(sub)], axis=1)

    y = sub[dep_col]
    model = PanelOLS(y, exog, entity_effects=True, drop_absorbed=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return res, sub


def step21_nonlinear_threshold_test(merged_frame, dep_label: str = "dep_bal", k: int = 2):
    section(f"STEP 21. 비선형(임계) 직접 검정 ({dep_label}, k*={k})")

    if merged_frame is None:
        print("[경고] 병합 프레임 없음, STEP21 생략.")
        return

    lag_col = f"exp_yoy_lag{k}"
    print(f"[진행] {lag_col} 부호분리(neg={{min,0}}/pos={{max,0}})*exposure 모형 추정 중 "
          f"(법인FE + C(월)+C(연도), 클러스터SE)...")
    try:
        res, sub = fit_panel_spec_nonlinear(merged_frame, dep_label, lag_col)
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 모형 추정 실패: {e}")
        return

    rows = []
    for term, label in [
        ("neg", "neg (하락국면 주효과)"),
        ("pos", "pos (상승국면 주효과)"),
        ("neg_x_exposure", "neg×exposure (핵심)"),
        ("pos_x_exposure", "pos×exposure (핵심)"),
    ]:
        rows.append({"항": label, **get_coef_row(res, term)})

    print(f"\n[{dep_label}] n_obs={int(res.nobs):,}, n_firm={sub.index.get_level_values(0).nunique():,}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n해석: |neg×exposure| 계수가 |pos×exposure|보다 유의하게 크면, 노출군이 수출 '하락' 국면에서만 "
          "요구불예금이 반응한다는 임계효과의 직접 증거로 볼 수 있음(비대칭 반응).")


# =====================================================================
# STEP 22. 위약(플라시보) 검정
# =====================================================================

def step22_placebo_test(merged_frame, dep_label: str = "dep_bal", k: int = 2,
                         n_iter: int = 30, seed: int = 42):
    section(f"STEP 22. 위약(플라시보) 검정 ({dep_label}, k*={k}, {n_iter}회)")

    if merged_frame is None:
        print("[경고] 병합 프레임 없음, STEP22 생략.")
        return

    lag_col = f"exp_yoy_lag{k}"
    term = f"{lag_col}_x_exposure"

    print("[진행] 실제(참) β3 추정 중...")
    try:
        res_real, _ = fit_panel_spec_b(merged_frame, dep_label, lag_col)
        real_beta3 = float(res_real.params[term])
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 실제 모형 추정 실패, STEP22 생략: {e}")
        return

    firm_exposure = merged_frame.drop_duplicates(subset=["firm_id"]).set_index("firm_id")["exposure"]
    firm_ids = firm_exposure.index.to_numpy()
    n_firms = len(firm_ids)
    n_exposed = int(firm_exposure.sum())
    actual_ratio = n_exposed / n_firms * 100 if n_firms else np.nan

    print(f"[진행] 위약검정 {n_iter}회 반복 중 (법인수={n_firms:,}, 가짜노출 법인수={n_exposed:,}, "
          f"비율={actual_ratio:.2f}%, seed={seed}, 표본이 커서 시간이 걸릴 수 있습니다)...")

    np.random.seed(seed)
    base_frame = merged_frame.drop(columns=["exposure"])

    placebo_betas = []
    placebo_tstats = []
    for i in range(n_iter):
        fake_exposed_ids = np.random.choice(firm_ids, size=n_exposed, replace=False)
        fake_exposure_map = pd.Series(0, index=firm_ids)
        fake_exposure_map.loc[fake_exposed_ids] = 1

        trial_frame = base_frame.copy()
        trial_frame["exposure"] = trial_frame["firm_id"].map(fake_exposure_map)

        try:
            res_p, _ = fit_panel_spec_b(trial_frame, dep_label, lag_col)
            placebo_betas.append(float(res_p.params[term]))
            placebo_tstats.append(float(res_p.tstats[term]))
        except Exception as e:  # noqa: BLE001
            print(f"[경고] {i + 1}회차 추정 실패: {e}")

        if (i + 1) % 10 == 0:
            print(f"  [진행] {i + 1}/{n_iter} 완료")

    if not placebo_betas:
        print("[경고] 위약 추정이 모두 실패했습니다. STEP22 결과 없음.")
        return

    placebo_betas = np.array(placebo_betas)
    placebo_tstats = np.array(placebo_tstats)
    pct_sig = (np.abs(placebo_tstats) > 1.96).mean() * 100
    percentile_rank = (placebo_betas < real_beta3).mean() * 100

    print(f"\n[{dep_label}] 실제 β3 = {real_beta3:.6f} (성공한 위약 반복수: {len(placebo_betas)}/{n_iter})")
    print(f"가짜 β3 평균: {placebo_betas.mean():.6f}, 표준편차: {placebo_betas.std(ddof=1):.6f}")
    print(f"가짜 β3 중 |t|>1.96 비율: {pct_sig:.2f}%")
    print(f"실제 β3의 가짜 분포 내 백분위: {percentile_rank:.2f}% "
          f"(0% 또는 100%에 가까울수록 실제 신호가 우연이 아닐 가능성이 높음)")


# =====================================================================
# main
# =====================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    print(f"[시작] 파일 경로: {path}")

    df = load_data_with_fallback(path, sheet_name=SHEET_NAME)
    cols = resolve_columns(df, COLUMN_MAP)

    if "id_col" not in cols or "ym_col" not in cols:
        raise RuntimeError("id_col / ym_col 은 필수입니다. COLUMN_MAP을 확인하세요.")

    ym_norm_col = add_normalized_ym(df, cols["ym_col"])
    midx_col = add_month_index(df, ym_norm_col)

    df = df.sort_values([cols["id_col"], ym_norm_col]).reset_index(drop=True)

    step1_basic_structure(df, cols, ym_norm_col)
    step2_key_uniqueness(df, cols, ym_norm_col)
    step3_unit_distribution(df, cols)
    step4_no_variation(df, cols, midx_col)
    step5_treatment_groups(df, cols)
    step23_data_definition_checks(df, cols, ym_norm_col, midx_col)

    sample_df = step6_analysis_sample(df, cols)
    step7_panel_continuity(sample_df, cols, ym_norm_col, midx_col)
    step8_account_responsiveness(sample_df, cols, midx_col)
    step9_demand_deposit_flow(sample_df, cols, ym_norm_col, midx_col)
    step10_export_timeseries(sample_df, cols, ym_norm_col)

    dep_bal_frame, dep_flow_frame = build_regression_frames(sample_df, cols, ym_norm_col, midx_col)
    step11_dep_var_stats(dep_bal_frame, dep_flow_frame)
    step12_exposure_comparison(dep_bal_frame, dep_flow_frame)
    step13_fe_regressions(dep_bal_frame)
    step14_dep_flow_regression(dep_flow_frame)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ext_df = step15_load_external_export_data(script_dir)

    merged_bal_frame = step16_merge_external(dep_bal_frame, sample_df, cols, ext_df)
    rows_b_bal = step17_beta3_regressions(merged_bal_frame, dep_label="dep_bal")
    step18_conditionality_test(merged_bal_frame, rows_b_bal, dep_label="dep_bal")

    step19_dep_flow_repeat(dep_flow_frame, sample_df, cols, ext_df)

    step20_region_cluster_robustness(merged_bal_frame, dep_label="dep_bal", k=2)
    step21_nonlinear_threshold_test(merged_bal_frame, dep_label="dep_bal", k=2)
    step22_placebo_test(merged_bal_frame, dep_label="dep_bal", k=2, n_iter=30, seed=42)

    section("완료")


if __name__ == "__main__":
    main()
