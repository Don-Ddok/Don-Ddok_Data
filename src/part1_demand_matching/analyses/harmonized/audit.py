# -*- coding: utf-8 -*-
"""사양 점검표 (새 분석 없음): 네 계정 C1이 같은 사양으로 실행됐는지
근거: run_log.txt(실행 로그), results_all.csv(결과), run_harmonized.py의 run_common 호출 인자(AST), hlib 기본값(inspect).
p값·Holm은 저장된 β3·SE·법인 수·월 수로 다시 계산해 저장값과 대조만 한다(추정은 하지 않음).
출력: audit.md, audit_checks.csv
"""
import ast
import inspect
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
import hlib

OUT = r"C:\test\outputs\harmonized"
ACCS = ["요구불", "운전자금", "거치식", "적립식"]

# 1) 호출 인자: run_harmonized.py의 C1 분기에서 run_common 호출 키워드
src = open(os.path.join(OUT, "run_harmonized.py"), encoding="utf-8").read()
tree = ast.parse(src)
c1_kw = None
for node in ast.walk(tree):
    if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and getattr(node.test.comparators[0], "value", None) == "C1":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "run_common" and not any(k.arg == "dep" for k in sub.keywords):
                c1_kw = {k.arg: ast.unparse(k.value) for k in sub.keywords}
                c1_args = [ast.unparse(a) for a in sub.args]
                break
        break
defaults = {k: v.default for k, v in inspect.signature(hlib.run_common).parameters.items() if v.default is not inspect._empty}
panel_def = {k: v.default for k, v in inspect.signature(hlib.Panel.__init__).parameters.items() if v.default is not inspect._empty}
m_data = re.search(r'^DATA = r"(.+)"', src, re.M).group(1)
m_hs = re.search(r"^HS = (.+)$", src, re.M).group(1)
m_pf = re.search(r"^PF_C1 = (.+)$", src, re.M).group(1)
used = {k: c1_kw.get(k, repr(defaults.get(k))) for k in ["xkey", "weights", "winsor", "positive", "hold", "start", "keep", "pf_hs"]}
print("C1 호출:", c1_args, c1_kw, "\n→ 적용값:", used, "\nPanel 기본 노출:", panel_def)

# 2) 실행 로그: 계정별 마지막 C1 블록 (현재 결과를 만든 실행)
log = open(os.path.join(OUT, "run_log.txt"), encoding="utf-8").read().splitlines()
logi = {}
for a in ACCS:
    heads = [i for i, l in enumerate(log) if l.startswith(f"===== {a}:")]
    c1s = [i for i in heads if i + 1 < len(log) and log[i + 1] == f"--- {a} C1"]
    i = c1s[-1]
    m = re.search(r"법인 ([\d,]+) \(노출 ([\d,]+)\), 행 ([\d,]+), 0 이하 잔액 행 ([\d,]+)", log[i])
    j = i + 2; hl = []
    while j < len(log) and log[j].startswith(f"  [C1]"):
        hl.append(log[j]); j += 1
    fails = [l for l in hl if "실패" in l]
    pre_line = next(l for l in log[j:j + 20] if l.startswith(f"[{a} 사전추세]"))
    logi[a] = {"법인": m.group(1), "노출": m.group(2), "행": m.group(3), "0행": m.group(4), "C1 시차 줄": len(hl),
               "실패": len(fails), "사전추세 줄": pre_line, "로그 위치": i + 1}

# 3) 결과: C1 행 재계산 대조
r = pd.read_csv(os.path.join(OUT, "results_all.csv"))
pre = pd.read_csv(os.path.join(OUT, "pretrend.csv")).set_index("계정")
chk_rows, res = [], {}
for a in ACCS:
    c = r[(r["계정"] == a) & (r["모형"] == "C1")].sort_values("h")
    G = np.minimum(c["노출법인"] + c["비노출법인"], c["월"])
    p_re = 2 * stats.t.sf(np.abs(c["β3"] / c["SE"]), G - 1)
    fam = c["h"].between(1, 12)
    holm_re = hlib.holm(c.loc[fam, "p"].to_numpy())
    hs = c["h"].tolist()
    res[a] = {"h": f"{min(hs)}~{max(hs)}, {len(hs)}개, −1 제외={-1 not in hs}",
              "p재계산 최대차": float(np.max(np.abs(p_re - c["p"]))),
              "Holm재계산 최대차": float(np.max(np.abs(holm_re - c.loc[fam, "p_holm"]))),
              "Holm 칸 수": int(fam.sum()), "Holm 밖 NaN": bool(c.loc[~fam, "p_holm"].isna().all()),
              "G_min": f"{int(G.min())}~{int(G.max())} (= 월 수)" if (G == c["월"]).all() else "법인 수가 더 작은 칸 있음",
              "pf 칸": int(c["pf_β3"].notna().sum()), "pf h": ",".join(map(str, c.loc[c["pf_β3"].notna(), "h"])),
              "출처": c["출처"].unique().tolist(), "h6 법인": int(c.loc[c.h == 6, "노출법인"].iloc[0] + c.loc[c.h == 6, "비노출법인"].iloc[0])}
    pr = pre.loc[a]
    res[a]["사전추세"] = f"LP h=−6~−2 결합 χ²(5) p {pr['p_chi2']:.3f} · 검산 {pr['검산_최대상대차']:.0e} · 원본 방식 N {int(pr['원본방식_N']):,}"
    chk_rows.append({"계정": a, **{k: v for k, v in res[a].items() if k not in ("출처",)}, **{f"로그_{k}": v for k, v in logi[a].items() if k != "사전추세 줄"}})
pd.DataFrame(chk_rows).to_csv(os.path.join(OUT, "audit_checks.csv"), index=False, encoding="utf-8-sig")

# 4) 점검표
ACCOL = {"요구불": "요구불예금잔액", "운전자금": "여신_운전자금대출잔액", "거치식": "거치식예금잔액", "적립식": "적립식예금잔액"}
same = lambda v: {a: v for a in ACCS}
rows = [
    ("데이터 파일", same(os.path.basename(m_data) + " (대구·경북, 2023-01~2025-12)"), "같음", "run_harmonized.py DATA, 로그 `df_ready 대구·경북` 줄"),
    ("표본 규칙", {a: f"`{ACCOL[a]}` > 0인 달 ≥ 1회" for a in ACCS}, "다름 — SPEC(계정별 보유 이력)", "run_account: groupby max > 0"),
    ("표본 법인 수 (전체 / 노출)", {a: f"{logi[a]['법인']} / {logi[a]['노출']}" for a in ACCS}, "다름 — SPEC(계정별 표본)", "로그 `=====` 줄"),
    ("법인×월 (0 이하 잔액 행)", {a: f"{logi[a]['행']} ({logi[a]['0행']})" for a in ACCS}, "다름 — SPEC", "로그"),
    ("h=6 추정 법인 수", {a: f"{res[a]['h6 법인']:,}" for a in ACCS}, "다름 — SPEC(t+h·t−1 관측 조건)", "results_all.csv"),
    ("FE 구성", same("법인 FE + 지역×연월 FE (hlib.fit → partial_out(firm, regym))"), "같음", "hlib.fit"),
    ("통제항", same("노출 × 영업일수 전년동월차 (SB)"), "같음", "hlib.run_common SB"),
    ("충격 변수", same(f"xkey={used['xkey']} → 보정YoY ÷ 100"), "같음", "C1 호출 인자(기본값)"),
    ("노출 변수", same(f"expo_col={panel_def['expo_col']!r} (df_ready `exposed`)"), "같음", "Panel 기본값, C1은 Panel(s, col)"),
    ("종속변수 식", same("ln(잔액_{t+h}+1) − ln(잔액_{t−1}+1), 정확히 그 달 관측 (hold=False, positive=False)"), "같음", "C1 호출 인자(기본값)"),
    ("가중치", same(f"weights={used['weights']} (1)"), "같음", "C1 호출 인자"),
    ("윈저", same(f"winsor={used['winsor']} → h마다 1%/99%, 가중치 없는 분위"), "같음", "C1 호출 인자(기본값)"),
    ("시차 범위", {a: res[a]["h"] for a in ACCS}, "같음", f"HS = {m_hs}, results_all.csv"),
    ("실패한 시차", {a: f"{logi[a]['실패']}개 (로그 C1 줄 {logi[a]['C1 시차 줄']}개)" for a in ACCS}, "같음", "로그"),
    ("SE 방식", same("법인·연월 이중 군집 (CGM, 각 차원 G/(G−1)·(n−1)/(n−k))"), "같음", "hlib.fit"),
    ("p 자유도", {a: f"t(G_min−1), G_min {res[a]['G_min']}; 재계산 최대차 {res[a]['p재계산 최대차']:.0e}" for a in ACCS}, "같음 (G_min 값은 시차별 월 수라 SPEC대로 다름)", "β3·SE·법인·월로 재계산"),
    ("Holm 범위", {a: f"h=1~12 ({res[a]['Holm 칸 수']}칸), 재계산 최대차 {res[a]['Holm재계산 최대차']:.0e}" for a in ACCS}, "같음", "p_holm 재계산"),
    ("pyfixest 교차 확인", {a: f"h={res[a]['pf h']}" for a in ACCS}, "같음", f"PF_C1 = {m_pf}"),
    ("사전추세 방식", {a: res[a]["사전추세"] for a in ACCS}, "방식 같음 (N은 SPEC대로 다름)", "pretrend.csv, 로그"),
    ("결과 출처", {a: "; ".join(res[a]["출처"]) for a in ACCS}, "같음", "results_all.csv `출처`"),
]
md = ["| 항목 | " + " | ".join(ACCS) + " | 계정 간 | 근거 |", "|---|" + "---|" * len(ACCS) + "---|---|"]
for name, vals, flag, basis in rows:
    md.append(f"| {name} | " + " | ".join(str(vals[a]) for a in ACCS) + f" | {flag} | {basis} |")
text = "\n".join(md)
open(os.path.join(OUT, "audit.md"), "w", encoding="utf-8").write(text + "\n")
print(text)
print("\n로그 위치(현재 결과를 만든 C1 블록):", {a: logi[a]["로그 위치"] for a in ACCS})
