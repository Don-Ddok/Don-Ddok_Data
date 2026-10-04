# -*- coding: utf-8 -*-
"""결과 정리 (SPEC §7·§9): results_all.csv, pretrend.csv, 판정표·요약표·업종별 표(tables.md), 그림
판정 규칙과 원래 결과 기준값은 SPEC.md(커밋 58502f3)에 고정된 그대로.
"""
import glob
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\test\outputs\harmonized"
PARTS = os.path.join(OUT, "parts")
ACCS = ["요구불", "운전자금", "거치식", "적립식"]
MODELS = ["C1", "C2", "E23", "EX", "POS", "HOLD", "RAW"]

# SPEC §7 원래 결과 기준값 (h=6, 10%p 하락 시 차이 %, 원래 p)
ORIG = {"요구불": (-3.56, 0.004), "운전자금": (2.34, 0.094), "거치식": None, "적립식": None}
ORIG_IND = {  # (계정, 업종, 모형): (환산, p)
    ("거치식", "1차 금속", "C1"): (61.1, 0.000), ("거치식", "기타 기계", "C1"): (22.9, 0.358), ("거치식", "도매·상품중개", "C1"): (-3.9, 0.679),
    ("적립식", "1차 금속", "C1"): (25.1, 0.065), ("적립식", "기타 기계", "C1"): (4.5, 0.725), ("적립식", "도매·상품중개", "C1"): (10.6, 0.189),
    ("거치식", "1차 금속", "HOLD"): (9.0, 0.000), ("거치식", "기타 기계", "HOLD"): (2.5, 0.567), ("거치식", "도매·상품중개", "HOLD"): (-1.2, 0.532),
    ("적립식", "1차 금속", "HOLD"): (5.9, 0.140), ("적립식", "기타 기계", "HOLD"): (3.1, 0.284), ("적립식", "도매·상품중개", "HOLD"): (2.1, 0.368)}


def vs_orig(orig, orig_p, c1, c1_p):
    if orig is None:
        return "원래 pooled 결과 없음"
    if np.sign(orig) != np.sign(c1):
        mark = "뒤집힘"
    elif abs(c1) >= abs(orig) / 2:
        mark = "유지"
    else:
        mark = "약해짐"
    if orig_p < 0.10 and c1_p >= 0.10:
        if mark in ("약해짐", "뒤집힘"):
            return f"{mark} · 착시 가능성"
        return "유지 · 크기 유지·유의성 상실"
    return mark


def f(x, d=2):
    return "–" if pd.isna(x) else f"{x:+.{d}f}"


def main():
    # 요구불 POS: did_demand_deposit A-양수 (SPEC §8 재사용, 재계산과 일치 확인 — check_pos_13.py)
    src = pd.read_csv(r"C:\test\outputs\did_demand_deposit\results_table.csv")
    pos = src[src["모형"] == "A-양수"].rename(columns={"10%p하락_차이%": "환산"}).copy()
    pos["단위"] = "%"; pos["모형"] = "POS"; pos.insert(0, "업종", np.nan); pos.insert(0, "계정", "요구불")
    pos["출처"] = "did_demand_deposit A-양수 (커밋 8884970)"
    pos.to_csv(os.path.join(PARTS, "res_요구불_POS.csv"), index=False, encoding="utf-8-sig")

    allr = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(PARTS, "res_*.csv"))], ignore_index=True)
    allr["출처"] = allr.get("출처", pd.Series(dtype=str)).fillna("harmonized run_common")
    allr["구분"] = np.where(allr["모형"].str.startswith("업종"), "사후 재추정 · 탐색적", "사후 재추정")
    allr.to_csv(os.path.join(OUT, "results_all.csv"), index=False, encoding="utf-8-sig")
    pre = pd.concat([pd.read_csv(os.path.join(PARTS, f"pre_{a}.csv")) for a in ACCS], ignore_index=True)
    chk = pd.concat([pd.read_csv(os.path.join(PARTS, f"prechk_{a}.csv")).assign(계정=a) for a in ACCS], ignore_index=True)
    pre.to_csv(os.path.join(OUT, "pretrend.csv"), index=False, encoding="utf-8-sig")
    chk.to_csv(os.path.join(OUT, "pretrend_check.csv"), index=False, encoding="utf-8-sig")

    main_rows, lines = [], []
    for a in ACCS:
        c = allr[(allr["계정"] == a) & (allr["모형"] == "C1")].set_index("h")
        pr = pre[pre["계정"] == a].iloc[0]
        b6, p6, e6 = c.loc[6, "β3"], c.loc[6, "p"], c.loc[6, "환산"]
        flag = (np.sign(pr["원본방식_β3pre"]) == np.sign(b6)) and abs(pr["원본방식_β3pre"]) >= abs(b6) / 2
        holm_min = c.loc[c.index.isin(range(1, 13)), "p_holm"].min()
        holm_sig = sorted(c.index[(c.index >= 1) & (c["p_holm"] < 0.05)].tolist())
        if holm_min < 0.05 and pr["p_chi2"] >= 0.10 and not flag:
            grade = "확정"
        elif p6 < 0.10:
            grade = "약한 증거"
        else:
            grade = "근거 없음"
        o = ORIG[a]
        mark = vs_orig(o[0] if o else None, o[1] if o else None, e6, p6)
        n6 = f"법인 {int(c.loc[6, '노출법인'] + c.loc[6, '비노출법인']):,} (노출 {int(c.loc[6, '노출법인']):,}) · 법인×월 {int(c.loc[6, 'N']):,}"
        main_rows.append({"계정": a, "원래(h=6,%)": "원래 pooled 결과 없음" if o is None else f"{o[0]:+.2f} (p {o[1]:.3f})",
                          "C1(h=6,%)": f"{e6:+.2f}", "β3": f"{b6:+.3f}", "이중군집 p": f"{p6:.3f}",
                          "Holm 최소 p": f"{holm_min:.3f}", "Holm<0.05 h": ",".join(map(str, holm_sig)) or "없음",
                          "사전추세 LP p": f"{pr['p_chi2']:.3f}",
                          "원본방식 β3pre": f"{pr['원본방식_β3pre']:+.3f} (p {pr['원본방식_p']:.3f})",
                          "연장 표시": "사전 추세 연장 가능성" if flag else "–", "등급": grade, "원래 대비": mark, "표본(h=6)": n6})
    mt = pd.DataFrame(main_rows)
    mt.to_csv(os.path.join(OUT, "judgment.csv"), index=False, encoding="utf-8-sig")
    lines.append("## 판정표\n\n" + mt.to_markdown(index=False))

    # 보고용 모형 요약 (h=6)
    rows = []
    for a in ACCS:
        c1 = allr[(allr["계정"] == a) & (allr["모형"] == "C1") & (allr["h"] == 6)].iloc[0]
        r = {"계정": a}
        for m in MODELS:
            x = allr[(allr["계정"] == a) & (allr["모형"] == m) & (allr["h"] == 6)]
            if len(x):
                x = x.iloc[0]
                opp = " ⚠부호 반대" if (m != "C1" and m != "HOLD" and np.sign(x["β3"]) != np.sign(c1["β3"])) else ""
                if m == "HOLD" and np.sign(x["β3"]) != np.sign(c1["β3"]):
                    opp = " ⚠부호 반대"
                r[m] = f"{x['환산']:+.2f}{x['단위']} (p {x['p']:.3f}){opp}"
        rows.append(r)
    rt = pd.DataFrame(rows)
    rt.to_csv(os.path.join(OUT, "robust_summary.csv"), index=False, encoding="utf-8-sig")
    lines.append("## 보고용 모형 요약 (h=6, 10%p 하락 시 노출−비노출 차이)\n\n" + rt.to_markdown(index=False))

    # 업종별
    rows = []
    for a in ["거치식", "적립식"]:
        for m in ["C1", "HOLD"]:
            g = allr[(allr["계정"] == a) & (allr["모형"] == f"업종_{m}")]
            for ind in ["1차 금속", "기타 기계", "도매·상품중개"]:
                x = g[(g["업종"] == ind) & (g["h"] == 6)]
                if not len(x):
                    continue
                x = x.iloc[0]
                o, op = ORIG_IND[(a, ind, m)]
                unit = "%p" if m == "HOLD" else "%"
                note = []
                if m == "C1" and abs(o) > 30:
                    note.append("원래: 크기 해석 불가, HOLD 참조")
                if m == "C1" and abs(x["환산"]) > 30:
                    note.append("C1: 크기 해석 불가, HOLD 참조")
                rows.append({"계정": a, "모형": m, "업종": ind, "법인(노출/비노출)": f"{int(x['노출법인'])}/{int(x['비노출법인'])}",
                             "원래 h=6": f"{o:+.1f}{unit} (p {op:.3f})", "C1 h=6": f"{x['환산']:+.1f}{unit}",
                             "p": f"{x['p']:.3f}", "Holm(36) p": f"{x['p_holm_36']:.3f}",
                             "원래 대비": vs_orig(o, op, x["환산"], x["p"]), "주석": "; ".join(note) or "–"})
    it = pd.DataFrame(rows)
    it.to_csv(os.path.join(OUT, "industry_summary.csv"), index=False, encoding="utf-8-sig")
    lines.append("## 업종별 (탐색적)\n\n" + it.to_markdown(index=False))
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(lines) + "\n")
    print("\n\n".join(lines))

    # 그림
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    col = {"요구불": "#00A38B", "운전자금": "#2D5FC4", "거치식": "#D26FC2", "적립식": "#A8903A"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=False)
    for ax, a in zip(axes, ACCS):
        c = allr[(allr["계정"] == a) & (allr["모형"] == "C1")].sort_values("h")
        c = pd.concat([c, pd.DataFrame([{"h": -1, "β3": 0.0, "CI_lo": 0.0, "CI_hi": 0.0}])]).sort_values("h")
        cv = lambda b: (np.exp(-0.1 * b) - 1) * 100
        ax.fill_between(c["h"], cv(c["CI_hi"]), cv(c["CI_lo"]), color=col[a], alpha=0.15, linewidth=0)
        ax.plot(c["h"], cv(c["β3"]), color=col[a], lw=2, marker="o", ms=3.5)
        ax.axhline(0, color="#62625d", lw=0.8); ax.axvline(-1, color="#9d9d97", lw=0.8, ls=":")
        ax.set_title(a, fontsize=11); ax.set_xticks(range(-6, 13, 3)); ax.set_xlabel("h (개월)")
        ax.grid(axis="y", color="#efefeb")
    axes[0].set_ylabel("10%p 하락 시 노출-비노출 차이 (%)")
    fig.suptitle("사후 재추정 · 공통 사양 C1 (법인 FE + 지역×연월 FE + 노출×영업일수차, 보정YoY, 윈저 1/99, 이중 군집 95% CI)", fontsize=10)
    fig.text(0.5, 0.005, "주: 세로축은 모두 같은 단위(%). 계정마다 크기가 달라 세로축 범위는 따로 둔다. h<0은 이전 - 나중이라 사후 효과와 같은 방향의 사전 추세는 반대 부호로 나온다.",
             ha="center", fontsize=8, color="#62625d")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(os.path.join(OUT, "irf_c1.png"), dpi=150); plt.close(fig)

    mc = {"C1": "#2f2f2c", "C2": "#2D5FC4", "E23": "#D26FC2", "EX": "#00A38B", "POS": "#A8903A", "RAW": "#9d9d97"}
    for a in ACCS:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2, 1]})
        for m, cc in mc.items():
            c = allr[(allr["계정"] == a) & (allr["모형"] == m)].sort_values("h")
            if not len(c):
                continue
            ax.plot(c["h"], c["환산"], color=cc, lw=2.2 if m == "C1" else 1.3, marker="o" if m == "C1" else None, ms=3,
                    ls="--" if m == "RAW" else "-", label=m + (" (판정)" if m == "C1" else ""))
        ax.axhline(0, color="#62625d", lw=0.8); ax.axvline(-1, color="#9d9d97", lw=0.8, ls=":")
        ax.set_xticks(range(-6, 13, 2)); ax.set_xlabel("h (개월)"); ax.set_ylabel("10%p 하락 시 차이 (%)")
        ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", color="#efefeb"); ax.set_title(f"{a}: C1과 보고용 모형 (잔액)", fontsize=11)
        c = allr[(allr["계정"] == a) & (allr["모형"] == "HOLD")].sort_values("h")
        ax2.plot(c["h"], c["환산"], color="#62625d", lw=1.8, marker="o", ms=3)
        ax2.fill_between(c["h"], -0.1 * c["CI_hi"] * 100, -0.1 * c["CI_lo"] * 100, color="#62625d", alpha=0.12, linewidth=0)
        ax2.axhline(0, color="#62625d", lw=0.8); ax2.set_xticks(range(-6, 13, 3)); ax2.set_xlabel("h (개월)")
        ax2.set_ylabel("보유 확률 차이 (%p)"); ax2.set_title("HOLD (보유 여부)", fontsize=11); ax2.grid(axis="y", color="#efefeb")
        fig.suptitle(f"사후 재추정 · {a}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, f"irf_robust_{a}.png"), dpi=150); plt.close(fig)
    print("그림 저장")


if __name__ == "__main__":
    main()
