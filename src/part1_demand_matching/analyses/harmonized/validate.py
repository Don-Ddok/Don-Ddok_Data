# -*- coding: utf-8 -*-
"""검증 단계 (SPEC §8): run_common을 did_loan_robust L1 사양으로 h=6 → 커밋된 값(0bc5a39)과 비교
L1: 파트3 패널, 운전자금 보유 이력 법인, 매칭 가중치(노출 1 / 대조 뽑힌 횟수÷3), 윈저 없음, 보정YoY
허용 오차: β3 소수 넷째 자리, SE 상대차 1% 이내
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from hlib import Panel, run_common

P3 = r"C:\test\external\part3_work\통계 프로젝트\여신업종분석\단계별 분석"
sys.path.insert(0, P3)
from common import match_weights   # 수정 없이 import

REF = {"β3": -0.3041, "SE": 0.1636, "p": 0.0735}   # did_loan_robust results_table.csv, L1 h=6

p = pd.read_parquet(os.path.join(P3, "step1_loan_industry_panel.parquet"))
p = p[p["운전자금_보유이력"] == 1]
P = Panel(p, "여신_운전자금대출잔액")
W = match_weights(pd.read_parquet(os.path.join(P3, "step3_psm_matched.parquet")))
w = P.d["법인ID"].map(W).fillna(0).to_numpy()
r, _ = run_common(P, [6], weights=w, winsor=False, label="L1재현")
got = r.iloc[0]
ref = pd.read_csv(r"C:\test\outputs\did_loan_robust\results_table.csv")
ref = ref[(ref["모형"] == "L1") & (ref["h"] == 6)].iloc[0]
print(f"\n커밋값: β3 {ref['β3']:.6f}, SE {ref['SE']:.6f}, p {ref['p']:.4f}, N {int(ref['N'])}")
print(f"재현값: β3 {got['β3']:.6f}, SE {got['SE']:.6f}, p {got['p']:.4f}, N {int(got['N'])}")
ok_b = round(got["β3"], 4) == round(ref["β3"], 4)
rel = got["SE"] / ref["SE"] - 1
ok_se = abs(rel) <= 0.01
print(f"β3 소수 넷째 자리 일치: {ok_b} / SE 상대차 {rel:+.5%} (≤1%: {ok_se}) / N 일치: {int(got['N']) == int(ref['N'])}")
print("검증 통과" if (ok_b and ok_se) else "검증 실패 — 본 실행 중단")
