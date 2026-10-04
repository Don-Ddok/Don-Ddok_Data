# -*- coding: utf-8 -*-
"""요구불 하방 비대칭 재확인 (민영 파트 ③) — R0 원본 / R1 이중 군집 / R2 R1 + 달력 보정 국면
판정 (사전 고정): 메인 3국면 직접 검정(증가 vs 깊은하락)을 R1·R2 각각 9개 시차 안에서 Holm 보정.
   R1·R2 모두 보정 후 유의한 칸이 없으면 "비대칭 없음 유지", 남는 칸이 있으면 "재검토 필요".
R2는 국면만 달력 보정 국면으로 바꾸고, 충격은 원래 exp_yoy/10을 그대로 쓴다.
데이터는 읽기만, 결과는 C:\\test\\분석결과. 실행: py -3.11 run_asym.py
"""
import os, sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, r"C:\test\code")
import asym_lp_adj as A
import financial_pressure_signals_adj as fa

OUT = r"C:\test\분석결과"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)

df = A.load_df()
# R2용 국면: adjust_export_phase로 재분류한 국면만 가져오고 exp_yoy는 원래 값 유지
adj, ex = fa.adjust_export_phase(df, fa.load_bizdays())
key = ["사업장_시도", "기준년월"]
ph = adj.drop_duplicates(key)[key + ["phase"]].rename(columns={"phase": "phase_cal"})
df = df.merge(ph, on=key, how="left")
n_changed = (df.drop_duplicates(key).query("사업장_시도 in ['대구','경북']")
             .pipe(lambda x: (x["phase"] != x["phase_cal"]).sum()))
print(f"[R2 국면] 달력 보정으로 국면이 바뀐 대경 지역×월: {n_changed}")

main = []
for spec, cov, pcol in [("R0", "R0", "phase"), ("R1", "twoway", "phase"), ("R2", "twoway", "phase_cal")]:
    r = A.run_lp(df, cov=cov, phase_col=pcol, **A.MAIN)
    r.insert(0, "사양", spec)
    if spec in ("R1", "R2"):
        r["직접검정_p_Holm"] = multipletests(r["직접검정_p"], method="holm")[1]
    main.append(r)
    print(f"done {spec}", flush=True)
M = pd.concat(main, ignore_index=True)

rob = []
for name, s in A.ROBUST.items():
    r = A.run_lp(df, cov="twoway", **s)
    r.insert(0, "강건성", name)
    rob.append(r)
    print(f"done R1 {name}", flush=True)
RB = pd.concat(rob, ignore_index=True)

M.to_csv(os.path.join(OUT, "민영_비대칭재확인_메인.csv"), index=False, encoding="utf-8-sig")
RB.to_csv(os.path.join(OUT, "민영_비대칭재확인_강건성R1.csv"), index=False, encoding="utf-8-sig")

print("\n=== 메인 3국면: 증가·깊은하락 계수, 직접 검정 ===")
cols = ["사양", "h", "증가_b", "증가_t", "증가_p", "깊은하락_b", "깊은하락_t", "깊은하락_p",
        "차이", "직접검정_p", "직접검정_p_Holm", "G월", "n"]
print(M[[c for c in cols if c in M]].round(4).to_string(index=False))
print("\n=== 강건성 4종 (R1) ===")
print(RB[["강건성", "h", "pos_b", "pos_t", "pos_p", "neg_b", "neg_t", "neg_p", "직접검정_p", "G월"]]
      .round(4).to_string(index=False))

# ---- 판정 ----
J = M[M["사양"].isin(["R1", "R2"])]
raw = J[J["직접검정_p"] < 0.05]
holm = J[J["직접검정_p_Holm"] < 0.05]
print(f"\n[직접 검정] R1·R2 18번 중 p<0.05: {len(raw)}칸")
if len(raw):
    print(raw[["사양", "h", "직접검정_p", "직접검정_p_Holm"]].round(4).to_string(index=False))
verdict = "비대칭 없음 유지" if holm.empty else "재검토 필요"
print(f"[판정] Holm 보정 후 유의 칸 {len(holm)}개 → {verdict}")
r1 = M[M["사양"] == "R1"]
sig = r1[(r1["h"].between(2, 6)) & (r1["증가_p"] < 0.05)]["h"].tolist()
print(f"[증가 국면, R1] h=2~6 중 p<0.05 시차: {sig}")
