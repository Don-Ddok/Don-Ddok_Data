# -*- coding: utf-8 -*-
"""포착률·오경보율 달력 보정 재검증 (민영 파트 ①) — ①원본 ②대경 한정 ③대경 한정+달력 보정
판정 (사전 고정, ③ 통합 기준): in/out 방향 일관 + 포착률 > 신호 켜짐 비율 + 위약 p < 0.05
   → 셋 다 충족이면 "관측 시점 격차로 사용 가능", 하나라도 아니면 "착시로 판명"
대구 한정은 판정에 쓰지 않음 (탐색적). 데이터는 읽기만, 결과는 C:\\test\\분석결과.
실행: py -3.11 run_capture_adj.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\test\code")
import financial_pressure_signals_adj as fa

OUT = r"C:\test\분석결과"
os.makedirs(OUT, exist_ok=True)
C = fa.COLS
LAG = fa.DEFAULT_PRESSURE_LAG   # 3

use = [C["id"], C["ym"], C["region"], C["industry"], C["deposit"], C["bill"], "exp_yoy", "phase"]
df = pd.read_csv(r"C:\test\data\processed\df_ready.csv", usecols=use, encoding="utf-8-sig")
biz = fa.load_bizdays()
dg = df[df[C["region"]].isin(fa.DG_REGIONS)].copy()
print(f"[데이터] 전체 {len(df):,}행 / 대경 {len(dg):,}행, 법인 {dg[C['id']].nunique():,}")

# ---- ③ 보정 준비 ----
dg_adj, ex = fa.adjust_export_phase(dg, biz)
bank = fa.estimate_bank_slopes(dg, biz, LAG)
print(f"\n[달력 보정] 수출 % YoY ~ 영업일수 전년차: 1일당 {ex['slope']:+.3f}%p, 상관 {ex['corr']:+.3f}, 관측 {ex['n']}")
for k, nm in [("dep", "요구불"), ("bill", "할인어음")]:
    b = bank[k]
    print(f"[달력 보정] {nm} ln 3개월 차분 ~ 영업일수 3개월 변화: 1일당 {b['slope']:+.5f}, "
          f"상관 {b['corr']:+.4f}, 관측 {b['n']:,}")
print("\n[컷오프·국면별 월 수, 2023-01~2025-12]")
print(ex["phase_counts"].to_string(index=False))

VERSIONS = [("① 원본", df, None), ("② 대경 한정", dg, None), ("③ 대경+달력 보정", dg_adj, bank)]
SCOPES = [("통합", None), ("대구(탐색적)", "대구"), ("경북", "경북")]

rows = []
for vname, data, slopes in VERSIONS:
    for sname, region in SCOPES:
        fmi = fa.build_pressure_index_adj(data, lag=LAG, region=region, bank_slopes=slopes, biz=biz)
        io = fa.inout_consistency(fmi)
        cm = fa.capture_metrics(fmi)
        rows.append({"버전": vname, "범위": sname, **cm,
                     "in": io["in"], "out": io["out"], "in/out 일관": io["일관"]})
        print(f"done {vname} {sname}", flush=True)
R = pd.DataFrame(rows)

# ---- ① 검증: 원본 함수 결과(51.1% / 68.2%, 통합 불일치·대구 일관)와 같아야 함 ----
o = R[R["버전"] == "① 원본"].set_index("범위")
assert round(o.loc["통합", "포착률"], 4) == 0.5111 and round(o.loc["대구(탐색적)", "포착률"], 4) == 0.6818, "① 재현 실패"
assert (not o.loc["통합", "in/out 일관"]) and o.loc["대구(탐색적)", "in/out 일관"], "① in/out 재현 실패"
print("\n[검증] ① 원본 재현 일치 (통합 51.1%, 대구 68.2%, in/out 통합 불일치·대구 일관)")

R.to_csv(os.path.join(OUT, "민영_포착률보정_비교.csv"), index=False, encoding="utf-8-sig")
ex["phase_counts"].to_csv(os.path.join(OUT, "민영_포착률보정_국면.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
show = ["버전", "범위", "포착률", "포착", "하락월", "오경보율", "신호켜짐비율", "지역월",
        "위약_p", "위약_평균포착률", "in", "out", "in/out 일관"]
print("\n=== 보정 전후 비교 (새 지표 분모: 대구·경북 지역×월) ===")
print(R[show].round(4).to_string(index=False))

# ---- 판정 (사전 고정, ③ 통합) ----
j = R[(R["버전"] == "③ 대경+달력 보정") & (R["범위"] == "통합")].iloc[0]
c1, c2, c3 = bool(j["in/out 일관"]), bool(j["포착률"] > j["신호켜짐비율"]), bool(j["위약_p"] < 0.05)
verdict = "관측 시점 격차로 사용 가능" if (c1 and c2 and c3) else "착시로 판명"
print(f"\n=== 최종 판정 (③ 통합): {verdict} ===")
print(f"  1. in/out 방향 일관 = {c1}  (in={j['in']}, out={j['out']})")
print(f"  2. 포착률 {j['포착률']:.3f} > 신호 켜짐 비율 {j['신호켜짐비율']:.3f} = {c2}")
print(f"  3. 위약 p {j['위약_p']:.3f} < 0.05 = {c3}")
