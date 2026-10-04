# -*- coding: utf-8 -*-
"""
financial_pressure_signals.py의 검증된 함수들이 만든 결과를 보고서/팀공유용
파일로 저장한다. 새로운 분석/통계량 계산은 하지 않는다 - 이미 검증된 함수들의
반환값을 CSV/마크다운으로 옮겨 적을 뿐이다.

전제: df에 exp_yoy/phase/exposed가 이미 병합되어 있음.
저장 위치: C:\\test\\results\\ (없으면 생성)
인코딩: 전부 utf-8-sig (한글 엑셀 호환)
"""

import os

import numpy as np
import pandas as pd

from financial_pressure_signals import (
    COLS, DOWN_PHASES,
    build_pressure_index, lead_time_analysis, industry_signal_dictionary,
    compare_lag_definitions, compare_lead_time_by_scope,
)

RESULTS_DIR = r"C:\test\results"


def ensure_results_dir(results_dir: str = RESULTS_DIR) -> str:
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_df(df: pd.DataFrame, filename: str, results_dir: str = RESULTS_DIR) -> str:
    path = os.path.join(results_dir, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[저장] {path} ({len(df):,}행)")
    return path


def _lag_consistency_table(merged: pd.DataFrame, lags: tuple) -> pd.DataFrame:
    """compare_lag_definitions()가 반환한 국면×기간별 비교표에서, 각 lag에 대해
    in/out 두 기간 모두 하락계열 평균지수가 증가국면보다 높은지(방향 일관성)를
    표로 추출한다. compare_lag_definitions()가 화면에 찍는 것과 동일한 판정
    로직을 재사용해 표 형태로 옮겨 적은 것뿐, 새로운 계산은 아니다."""
    phase_col = COLS["phase"]
    rows = []
    for lag in lags:
        col = f"lag={lag}"
        if col not in merged.columns:
            continue
        pivot = merged.pivot(index=phase_col, columns="_period_", values=col)
        if "증가" not in pivot.index or not (DOWN_PHASES & set(pivot.index)):
            rows.append({"lag": lag, "in_consistent": np.nan, "out_consistent": np.nan, "둘다일관": np.nan})
            continue
        down_rows = [p for p in DOWN_PHASES if p in pivot.index]
        holds = {}
        for period in ["in-sample(전반부)", "out-sample(후반부)"]:
            if period not in pivot.columns:
                holds[period] = np.nan
                continue
            down_max = pivot.loc[down_rows, period].max()
            up_val = pivot.loc["증가", period]
            holds[period] = bool(pd.notna(down_max) and pd.notna(up_val) and down_max > up_val)
        both = holds.get("in-sample(전반부)") and holds.get("out-sample(후반부)")
        rows.append({
            "lag": lag,
            "in_consistent": holds.get("in-sample(전반부)"),
            "out_consistent": holds.get("out-sample(후반부)"),
            "둘다일관": both,
        })
    return pd.DataFrame(rows)


def extract_idea1(df: pd.DataFrame, results_dir: str = RESULTS_DIR,
                   lag: int = 3, region_of_interest: str = "대구") -> dict:
    print("\n" + "#" * 70)
    print("아이디어 1 결과 추출")
    print("#" * 70)

    idea1_full = build_pressure_index(df, lag=lag, region=None)
    save_df(idea1_full["phase_summary"], "idea1_phase_summary.csv", results_dir)

    lag_compare_all = compare_lag_definitions(df, region=None, lags=(1, lag))
    save_df(lag_compare_all, "idea1_lag_compare_통합.csv", results_dir)
    consistency_all = _lag_consistency_table(lag_compare_all, lags=(1, lag))
    save_df(consistency_all, "idea1_lag_consistency_통합.csv", results_dir)

    lag_compare_region = compare_lag_definitions(df, region=region_of_interest, lags=(1, lag))
    save_df(lag_compare_region, f"idea1_lag_compare_{region_of_interest}.csv", results_dir)
    consistency_region = _lag_consistency_table(lag_compare_region, lags=(1, lag))
    save_df(consistency_region, f"idea1_lag_consistency_{region_of_interest}.csv", results_dir)

    return {
        "phase_summary": idea1_full["phase_summary"],
        "lag_compare_all": lag_compare_all,
        "consistency_all": consistency_all,
        "lag_compare_region": lag_compare_region,
        "consistency_region": consistency_region,
        "region_of_interest": region_of_interest,
        "lag": lag,
    }


def extract_idea2(df: pd.DataFrame, results_dir: str = RESULTS_DIR,
                   lag: int = 3, region_of_interest: str = "대구") -> dict:
    print("\n" + "#" * 70)
    print("아이디어 2 결과 추출")
    print("#" * 70)

    comparison = compare_lead_time_by_scope(df, lag=lag, region_of_interest=region_of_interest)
    comparison = comparison.copy()
    if "포착률(b/a)" in comparison.columns:
        comparison["놓친비율(1-포착률)"] = 1 - comparison["포착률(b/a)"]
    save_df(comparison, "idea2_capture_leadtime_compare.csv", results_dir)

    return {"comparison": comparison, "region_of_interest": region_of_interest, "lag": lag}


def extract_idea3(df: pd.DataFrame, results_dir: str = RESULTS_DIR) -> dict:
    print("\n" + "#" * 70)
    print("아이디어 3 결과 추출")
    print("#" * 70)

    idea3 = industry_signal_dictionary(df)
    full_table = idea3["full_table"]
    manual_table = idea3["manual_table"]

    save_df(full_table, "idea3_full_table.csv", results_dir)

    significant = full_table[(full_table["p"] < 0.05) & (~full_table["신뢰주의"])].copy()
    save_df(significant, "idea3_significant.csv", results_dir)

    verdict_summary = manual_table["판정"].value_counts().rename_axis("판정").reset_index(name="업종수")
    save_df(verdict_summary, "idea3_verdict_summary.csv", results_dir)

    save_df(manual_table, "idea3_industry_account_summary.csv", results_dir)

    return {
        "full_table": full_table,
        "manual_table": manual_table,
        "significant": significant,
        "verdict_summary": verdict_summary,
    }


def write_overall_summary(idea1: dict, idea2: dict, idea3: dict,
                           results_dir: str = RESULTS_DIR) -> str:
    """세 아이디어의 핵심 숫자만 모은 요약 마크다운. 전부 위에서 이미 계산된
    결과 객체에서 값을 뽑아 문장에 끼워 넣을 뿐, 새로 계산하지 않는다."""
    print("\n" + "#" * 70)
    print("통합 요약 파일 생성")
    print("#" * 70)

    lines = ["# 자금압박 동행신호 - 핵심 결과 요약", ""]

    # --- 아이디어 1 ---
    lines.append("## 아이디어 1. 자금압박지수")
    region = idea1["region_of_interest"]
    lag = idea1["lag"]
    cons_region = idea1["consistency_region"]
    row3 = cons_region[cons_region["lag"] == lag]
    both_ok = bool(row3["둘다일관"].iloc[0]) if not row3.empty and pd.notna(row3["둘다일관"].iloc[0]) else None
    if both_ok is True:
        lines.append(f"- **{region} 한정, lag={lag}(3개월 누적)에서 in/out-sample 둘 다 방향 일관** "
                      f"- 과적합이 아니라 일반화되는 신호로 판단됨.")
    elif both_ok is False:
        lines.append(f"- {region} 한정, lag={lag}에서도 in/out 방향이 완전히 일관되지는 않음 - 추가 검증 필요.")
    else:
        lines.append(f"- {region} 한정 lag={lag} 방향일관성 판정 불가(데이터 부족).")
    lines.append(f"- 상세 수치: `idea1_phase_summary.csv`, `idea1_lag_compare_{region}.csv` 참고.")
    lines.append("")

    # --- 아이디어 2 ---
    lines.append("## 아이디어 2. 리드타임")
    comp = idea2["comparison"]
    if len(comp) >= 2 and "포착률(b/a)" in comp.columns:
        row_all = comp.iloc[0]
        row_region = comp.iloc[1]
        lines.append(f"- 포착률: 통합 {row_all['포착률(b/a)']*100:.1f}% vs "
                      f"{idea2['region_of_interest']} 한정 {row_region['포착률(b/a)']*100:.1f}%")
        if "리드타임(주)_vs_수출통계_평균" in comp.columns:
            lines.append(f"- 리드타임(수출통계 대비): 통합 {row_all['리드타임(주)_vs_수출통계_평균']:.1f}주, "
                          f"{idea2['region_of_interest']} 한정 {row_region['리드타임(주)_vs_수출통계_평균']:.1f}주")
        if "리드타임(주)_vs_광공업생산지수_평균" in comp.columns:
            lines.append(f"- 리드타임(광공업생산지수 대비): 통합 {row_all['리드타임(주)_vs_광공업생산지수_평균']:.1f}주, "
                          f"{idea2['region_of_interest']} 한정 {row_region['리드타임(주)_vs_광공업생산지수_평균']:.1f}주")
        if "놓친비율(1-포착률)" in comp.columns:
            lines.append(f"- 선택편향 경고: 통합 기준 하락국면의 {row_all['놓친비율(1-포착률)']*100:.1f}%는 "
                          f"은행신호가 놓친 사례 - 리드타임은 포착된 경우에 한해서만 유효.")
    else:
        lines.append("- 비교 결과 없음(포착 에피소드 부족).")
    lines.append("- 상세 수치: `idea2_capture_leadtime_compare.csv` 참고.")
    lines.append("")

    # --- 아이디어 3 ---
    lines.append("## 아이디어 3. 업종별 신호 사전")
    verdict = idea3["verdict_summary"]
    sig = idea3["significant"]
    n_sig_industries = sig["업종"].nunique() if not sig.empty else 0
    lines.append(f"- 유의(p<0.05 & 신뢰주의=False) 업종×계정×h 셀: {len(sig)}개 "
                 f"(고유 업종 {n_sig_industries}개)")
    for _, r in verdict.iterrows():
        lines.append(f"  - 판정 '{r['판정']}': {r['업종수']}개 업종")
    if not sig.empty:
        top_accounts = sig["계정"].value_counts()
        dominant_account = top_accounts.idxmax()
        lines.append(f"- 유의한 셀에서 가장 많이 등장한 계정: **{dominant_account}** "
                      f"({top_accounts.max()}/{len(sig)}건)")
    lines.append("- 상세 수치: `idea3_full_table.csv`(전체), `idea3_significant.csv`(유의한 것만), "
                  "`idea3_industry_account_summary.csv`(업종별 매뉴얼) 참고.")
    lines.append("")

    # --- 종합 ---
    lines.append("## 종합")
    dominant_note = ""
    if not sig.empty:
        dominant_note = f"아이디어3에서 유의한 반응의 다수가 '{sig['계정'].value_counts().idxmax()}' 계정에서 나타났다는 점은, "
    lines.append(f"{dominant_note}아이디어1(요구불 압박지수)·아이디어2({idea2['region_of_interest']} 한정 리드타임) 결과와 "
                 f"함께 놓고 볼 때 서로 다른 각도(지수 방향성/관측시점/업종별 반응)에서 같은 계정을 "
                 f"가리키고 있는지 위 세 CSV를 대조해서 확인할 것.")
    lines.append("")
    lines.append("(이 요약은 위 함수들이 반환한 결과를 그대로 옮겨 적은 것이며, 별도 재계산은 하지 않았음.)")

    content = "\n".join(lines)
    path = os.path.join(RESULTS_DIR if results_dir is None else results_dir, "summary_overall.md")
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(content)
    print(f"[저장] {path}")
    return path


def export_all_results(df: pd.DataFrame, results_dir: str = RESULTS_DIR,
                        lag: int = 3, region_of_interest: str = "대구") -> None:
    ensure_results_dir(results_dir)
    idea1 = extract_idea1(df, results_dir, lag=lag, region_of_interest=region_of_interest)
    idea2 = extract_idea2(df, results_dir, lag=lag, region_of_interest=region_of_interest)
    idea3 = extract_idea3(df, results_dir)
    write_overall_summary(idea1, idea2, idea3, results_dir)
    print("\n[완료] 모든 결과 파일이 " + results_dir + " 에 저장되었습니다.")


# =====================================================================
# 사용 예시 (실행하지 않음 - 참고용)
# =====================================================================
"""
from extract_results_to_files import export_all_results

# df: 법인ID/기준년월/사업장_시도/업종_중분류/요구불예금잔액/운전_할인어음잔액/
#     exposed/exp_yoy/phase 가 이미 있는 병합 완료 데이터프레임
export_all_results(df, lag=3, region_of_interest="대구")
"""
