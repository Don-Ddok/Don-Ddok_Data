# -*- coding: utf-8 -*-
"""매칭모델 변경전후 비교 (Before=사전식 정렬·히스테리시스 없음, 재현 재실행 / After=현재 확정본)
입력: 분석결과\matching\_before_reproduction\*_before.csv, 현재 persona/recommend_firm_month.csv(After)
출력: step4_summary.txt, step4_top20_compare.csv, step4_transition_compare.csv (집계만)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\test\분석결과\matching"
BEF = os.path.join(BASE, "_before_reproduction")
OUT = r"C:\test\outputs\devreview"
LOG = open(os.path.join(OUT, "step4_summary.txt"), "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n")


def load(tag, suffix):
    r = pd.read_csv(os.path.join(BEF if tag == "before" else BASE, f"recommend_firm_month{suffix}.csv"),
                     usecols=["법인ID", "연월", "단계", "세그먼트", "페르소나", "우선순위순번"], dtype={"법인ID": str}, low_memory=False)
    p = pd.read_csv(os.path.join(BEF if tag == "before" else BASE, f"persona_firm_month{suffix}.csv"),
                     usecols=["법인ID", "연월", "요구불잔액"], dtype={"법인ID": str})
    d = r.merge(p, on=["법인ID", "연월"], how="left")
    return d


bef = load("before", "_before")
aft = load("after", "")
log("# 매칭모델 변경전후 비교 — 순위 점수화 + 페르소나 히스테리시스\n")
log(f"Before 행 {len(bef):,} / After 행 {len(aft):,} (같아야 함: {len(bef)==len(aft)})\n")

# ---------------- 1. 상위권 순위 비교 (2025-12 정점·유지, 노출) ----------------
log("## 1. 2025-12 정점·유지(노출) 상위 20위 비교")
b20 = bef[(bef["연월"] == 202512) & (bef["단계"] == "정점·유지") & bef["세그먼트"].isin(["수출형", "수입형"])].sort_values("우선순위순번").head(20)
a20 = aft[(aft["연월"] == 202512) & (aft["단계"] == "정점·유지") & aft["세그먼트"].isin(["수출형", "수입형"])].sort_values("우선순위순번").head(20)
overlap = len(set(b20["법인ID"]) & set(a20["법인ID"]))
log(f"- 구성원 겹침 {overlap}/20, 순서까지 동일: {b20['법인ID'].tolist() == a20['법인ID'].tolist()}")

# 3개월 변화(momentum) 계산해서 왜 바뀌었는지 표시
full_p = pd.read_csv(os.path.join(BASE, "persona_firm_month.csv"), usecols=["법인ID", "연월", "요구불잔액"], dtype={"법인ID": str})
full_p["_m"] = (full_p["연월"] // 100) * 12 + full_p["연월"] % 100
key = pd.Index(pd.factorize(full_p["법인ID"])[0].astype(np.int64) * 10000 + full_p["_m"].to_numpy())
fid_all = pd.factorize(full_p["법인ID"])[0].astype(np.int64)


def chg3(ids):
    sub = full_p[full_p["법인ID"].isin(ids) & (full_p["연월"] == 202512)][["법인ID", "요구불잔액"]].copy()
    prevmap = full_p[full_p["연월"] == 202509].set_index("법인ID")["요구불잔액"]
    sub["_prev"] = sub["법인ID"].map(prevmap)
    sub["최근3개월변화"] = sub["요구불잔액"] - sub["_prev"]
    return sub.set_index("법인ID")["최근3개월변화"]


chg = chg3(set(b20["법인ID"]) | set(a20["법인ID"]))
b20 = b20.assign(순위_before=b20["우선순위순번"].to_numpy(), 최근3개월변화=b20["법인ID"].map(chg))
a20 = a20.assign(순위_after=a20["우선순위순번"].to_numpy(), 최근3개월변화=a20["법인ID"].map(chg))
cmp = b20[["법인ID", "요구불잔액", "순위_before"]].merge(a20[["법인ID", "순위_after"]], on="법인ID", how="outer")
cmp = cmp.merge(chg.rename("최근3개월변화"), on="법인ID", how="left")
cmp["요구불잔액"] = cmp["요구불잔액"].fillna(cmp["법인ID"].map(full_p[full_p["연월"] == 202512].set_index("법인ID")["요구불잔액"]))
cmp = cmp.sort_values("순위_before", na_position="last")
cmp.to_csv(os.path.join(OUT, "step4_top20_compare.csv"), index=False, encoding="utf-8-sig")
log(cmp.round(0).to_string(index=False))
moved = cmp.dropna(subset=["순위_before", "순위_after"]).copy()
moved["이동"] = moved["순위_before"] - moved["순위_after"]
biggest = moved.reindex(moved["이동"].abs().sort_values(ascending=False).index).head(5)
log("\n- 가장 크게 이동한 5곳(순위_before - 순위_after, 양수=순위 상승):")
log(biggest[["요구불잔액", "최근3개월변화", "순위_before", "순위_after", "이동"]].round(0).to_string(index=False))

# ---------------- 2. 페르소나 전환 비교 ----------------
log("\n## 2. 페르소나 전환 비교 (전체 기간, 노출)")


def transitions(d, tag):
    p = pd.read_csv(os.path.join(BEF if tag == "before" else BASE, f"persona_firm_month{'_before' if tag=='before' else ''}.csv"),
                     usecols=["법인ID", "연월", "페르소나"], dtype={"법인ID": str})
    p["페르소나"] = p["페르소나"].fillna("")
    p["_m"] = (p["연월"] // 100) * 12 + p["연월"] % 100
    p = p.sort_values(["법인ID", "_m"]).reset_index(drop=True)
    fid = p["법인ID"].to_numpy(); m = p["_m"].to_numpy(); per = p["페르소나"].to_numpy()
    pf, pm, pp_ = np.roll(fid, 1), np.roll(m, 1), np.roll(per, 1)
    consec = (fid == pf) & (m == pm + 1)
    exposed_pair = consec & (per != "") & (pp_ != "")
    switched = exposed_pair & (per != pp_)
    n2 = np.roll(per, -2); n2m = np.roll(m, -2); n2f = np.roll(fid, -2)
    nxt, nxtm, nxtf = np.roll(per, -1), np.roll(m, -1), np.roll(fid, -1)
    seq_ok = (fid == nxtf) & (nxtm == m + 1) & (fid == n2f) & (n2m == m + 2)
    sw2 = seq_ok & (per != nxt) & (per != "") & (nxt != "")
    bounce = sw2 & (n2 == per)
    total_exposed_month = int((per != "").sum())
    return {"구분": tag, "노출 법인×월": total_exposed_month, "월간 전환 건수": int(switched.sum()),
            "월간 전환율(%)": switched.sum() / max(exposed_pair.sum(), 1) * 100,
            "왕복 대상 전환 건수": int(sw2.sum()), "단기왕복 건수": int(bounce.sum()),
            "단기왕복 비율(%)": bounce.sum() / max(sw2.sum(), 1) * 100}


tb = transitions(None, "before")
ta = transitions(None, "after")
tc = pd.DataFrame([tb, ta])
tc.to_csv(os.path.join(OUT, "step4_transition_compare.csv"), index=False, encoding="utf-8-sig")
log(tc.round(2).to_string(index=False))

LOG.close()
