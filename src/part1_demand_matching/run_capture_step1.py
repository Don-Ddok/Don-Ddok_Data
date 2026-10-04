# -*- coding: utf-8 -*-
"""1단계: 포착률 원본 재현 (financial_pressure_signals.py 수정 없이 호출만)
노트북 21번 셀과 같은 호출: compare_lag_definitions(통합/대구) + compare_lead_time_by_scope(lag=3)
데이터: C:\\test\\data\\processed\\df_ready.csv (읽기 전용), 출력은 화면만.
"""
import sys
import pandas as pd
sys.path.insert(0, r"C:\test\code")
import financial_pressure_signals as fps

C = fps.COLS
use = [C["id"], C["ym"], C["region"], C["industry"], C["deposit"], C["bill"], "exp_yoy", "phase", "exposed"]
df = pd.read_csv(r"C:\test\data\processed\df_ready.csv", usecols=use, encoding="utf-8-sig")
print(f"[데이터] {len(df):,}행, 기준년월 {df[C['ym']].min()}~{df[C['ym']].max()}")

print("\n━━━ 통합 lag 비교 ━━━")
fps.compare_lag_definitions(df)
print("\n━━━ 대구 한정 lag 비교 ━━━")
fps.compare_lag_definitions(df, region="대구")
print("\n━━━ 범위별 포착률·리드타임 ━━━")
fps.compare_lead_time_by_scope(df, lag=3)
