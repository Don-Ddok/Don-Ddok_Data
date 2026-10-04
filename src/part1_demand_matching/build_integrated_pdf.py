# -*- coding: utf-8 -*-
"""요구불 신호(민영) × 여신·업종(파트3) 통합 정리 PDF — 집계만 사용

입력: 분석결과\\통합분석\\integrated.json (integrated_analysis.py), 분석결과\\비교_파트3\\combo_compare.json,
      분석결과\\민영_β합검정.csv, 파트3 수치는 팀원 문서(파트3_분석전체_정리.pdf)·저장소 결과표
출력: 분석결과\\통합분석\\요구불x여신_통합정리.pdf
실행: py -3.11 build_integrated_pdf.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, r"C:\test\code\matching")
import build_summary_pdf as bsp  # noqa: E402
from build_summary_pdf import BLUE, BLUE_100, FONT, GRID, INK, INK2, MUTED, SURFACE, ST  # noqa: E402

# 맑은 고딕에 U+2212(−) 글리프가 없어 일반 하이픈으로 바꿔 그린다
_P0, _B0 = bsp.P, bsp.bullets
def P(t, s="p"):
    return _P0(str(t).replace("−", "-"), s)
def bullets(items):
    return _B0([str(t).replace("−", "-") for t in items])
bsp.P = P                      # table() 안에서도 같은 치환이 되도록
table = bsp.table

BASE = r"C:\test\분석결과"
OUT_DIR = os.path.join(BASE, "통합분석")
OUT = os.path.join(OUT_DIR, "요구불x여신_통합정리.pdf")
ORANGE = "#d9822b"
B13, B1, B3 = "β<sub>1</sub>+β<sub>3</sub>", "β<sub>1</sub>", "β<sub>3</sub>"

J = json.load(open(os.path.join(OUT_DIR, "integrated.json"), encoding="utf-8"))
CMP = json.load(open(os.path.join(BASE, "비교_파트3", "combo_compare.json"), encoding="utf-8"))
LP = pd.DataFrame(J["lp"])
CORR, PV = pd.DataFrame(J["corr"]), pd.DataFrame(J["p"])
FIRM = pd.DataFrame(J["firm"])
SAV = pd.read_csv(os.path.join(OUT_DIR, "savings_lp.csv"))


def sv(ind, acc, kind, h, col="차이_효과"):
    return float(SAV[(SAV["업종"] == ind) & (SAV["계정"] == acc) & (SAV["종속"] == kind) & (SAV["h"] == h)][col].iloc[0])

# 파트3 수치 (팀원 문서 파트3_분석전체_정리.pdf, 저장소 outputs/tables)
P3 = [("주 결과 (6개월, 매칭 1:3)", -0.231, "0.094"), ("+ 환율 (8-1)", -0.230, "0.095"), ("+ 영업일수 (8-6)", -0.306, "0.073"),
      ("+ 기준금리 (8-8)", -0.257, "0.056"), ("+ 달력월 더미 (8-9)", -0.227, "–"), ("2023-01 관측 회사만 (8-4)", -0.194, "0.140"),
      ("매칭 없이 회귀 보정 (4단계)", -0.149, "0.25"), ("IPW (8-2)", -0.165, "0.21"), ("수출 회사만 (9-3)", -0.084, "0.597"),
      ("균형 패널 (9-4)", -0.081, "0.695"), ("충격 전 6개월 (8-7, 사전추세)", -0.282, "0.095")]


def setup():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def fig_lp(path):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), facecolor=SURFACE)
    for ax, k in zip(axes, ["요구불", "운전자금", "할인어음"]):
        g = LP[LP["계정"] == k].sort_values("h")
        lo, hi = -g["ci13_hi"] * 100, -g["ci13_lo"] * 100
        style(ax)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.fill_between(g["h"], lo, hi, color=BLUE, alpha=.15, lw=0)
        ax.plot(g["h"], g["비노출_효과%"], color=MUTED, lw=1.6, ls="--")
        ax.plot(g["h"], g["노출_효과%"], color=BLUE, lw=2)
        sig = g["p13_holm"] < .05
        ax.scatter(g["h"][sig], g["노출_효과%"][sig], s=14, color=BLUE, zorder=3)
        ax.set_title(f"{k} (법인 {int(g['법인'].iloc[6]):,})", loc="left", fontsize=9, color=INK, fontweight="bold")
        ax.set_xticks([0, 3, 6, 9, 12])
    axes[0].set_ylabel("누적 변화 (%)", fontsize=8, color=INK2)
    axes[1].set_xlabel("수출 YoY 10%p 하락 후 경과 개월 (h)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_corr(path):
    cols = ["원YoY", "보정YoY", "영업일수차", "노출_요구불", "비노출_요구불", "노출_운전자금", "비노출_운전자금", "노출_할인어음", "비노출_할인어음"]
    lab = ["원 수출YoY", "보정 수출YoY", "영업일수차", "노출·요구불", "비노출·요구불", "노출·운전자금", "비노출·운전자금", "노출·할인어음", "비노출·할인어음"]
    c = CORR.loc[cols, cols].values
    fig, ax = plt.subplots(figsize=(6.6, 5.2), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f1f0ec", ORANGE])
    ax.imshow(c, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            p = PV.loc[cols[i], cols[j]]
            ax.text(j, i, f"{c[i, j]:.2f}{'*' if i != j and p < .05 else ''}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c[i, j]) > .6 else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(lab, fontsize=7.5, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_calendar(path):
    rows = [r for r in CMP if r["window"] == 3]
    labs, ys = [], []
    fig, ax = plt.subplots(figsize=(7.0, 2.6), facecolor=SURFACE)
    style(ax); ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
    y = 0
    for samp, treat in [("할인어음 보유 법인", "수출노출"), ("할인어음 보유 법인", "외환노출"), ("전체 법인(참고)", "수출노출")]:
        for kind, col in [("원", MUTED), ("보정", BLUE)]:
            r = next(x for x in rows if x["sample"] == samp and x["treat"] == treat and x["down_def"] == kind)
            ax.plot(r["didCI"], [y, y], color=col, lw=3, solid_capstyle="round")
            ax.scatter([r["did"]], [y], color=col, s=22, zorder=3)
            labs.append(f"{samp.replace('(참고)', '')} · {treat} · {kind} YoY"); ys.append(y); y += 1
        y += .6
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("조건 ② 초과분 차이 (감소 달 - 증가 달, %p) · 95% 구간", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_p3(path):
    fig, ax = plt.subplots(figsize=(7.0, 3.0), facecolor=SURFACE)
    style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
    names = [x[0] for x in P3][::-1]; vals = [x[1] for x in P3][::-1]
    cols = [ORANGE if "사전추세" in n else (BLUE if n.startswith("주 결과") else "#b9b8b2") for n in names]
    ax.barh(range(len(names)), vals, color=cols, height=.6)
    for i, (v, n) in enumerate(zip(vals, names)):
        ax.text(v - .005, i, f"{v:+.3f}", va="center", ha="right", fontsize=7, color=INK)
    ax.axvline(0, color=INK, lw=.8); ax.axvline(-0.1155, color=MUTED, lw=.8, ls=":")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(-0.36, 0.02)
    ax.set_xlabel("파트3 β₃ (운전자금 6개월, 노출-비노출; 음수 = 수출 하락기에 노출 법인이 상대적으로 대출 유지)", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_sav(path):
    fig, ax = plt.subplots(figsize=(7.0, 2.6), facecolor=SURFACE)
    style(ax)
    hs, w = [3, 6, 12], 0.26
    cols = {"요구불": BLUE, "거치식": ORANGE, "적립식": "#8a6fc2"}
    for i, a in enumerate(["요구불", "거치식", "적립식"]):
        v = [sv("전체", a, "로그", h) for h in hs]
        pp_ = [sv("전체", a, "로그", h, "p_차이") for h in hs]
        bars = ax.bar([x + (i - 1) * w for x in range(3)], v, w * .92, color=cols[a], label=a)
        for b, vv, pp in zip(bars, v, pp_):
            ax.text(b.get_x() + b.get_width() / 2, vv + (0.6 if vv >= 0 else -1.8), f"{vv:+.1f}{'*' if pp < .05 else ''}".replace("\u2212", "-"),
                    ha="center", fontsize=7, color=INK)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"h={h}" for h in hs], fontsize=8)
    ax.set_ylabel("노출-비노출 차이 (%)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Malgun", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "돈독 프로젝트 · 요구불 × 여신·업종 × 저축성 예금 통합 정리 · 2026-09-29")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def lpv(k, h, col="노출_효과%"):
    return float(LP[(LP["계정"] == k) & (LP["h"] == h)][col].iloc[0])


def build():
    setup()
    f = {k: os.path.join(OUT_DIR, f"_{k}.png") for k in ["lp", "corr", "cal", "p3", "sav"]}
    fig_lp(f["lp"]); fig_corr(f["corr"]); fig_calendar(f["cal"]); fig_p3(f["p3"]); fig_sav(f["sav"])
    s = []
    r_dep_biz = CORR.loc["영업일수차", "노출_요구불"]; p_dep_biz = PV.loc["영업일수차", "노출_요구불"]
    r_dep_raw = CORR.loc["원YoY", "노출_요구불"]; r_dep_adj = CORR.loc["보정YoY", "노출_요구불"]
    main_cmp = {r["down_def"]: r for r in CMP if r["window"] == 3 and r["treat"] == "수출노출" and r["sample"] == "할인어음 보유 법인"}

    # 표지
    s += [Spacer(1, 50 * mm), P("수출이 꺾일 때, 법인 통장과 대출은", "title"), P("어떻게 함께 움직이나", "title"),
          P("요구불 신호(민영) × 여신·업종(파트3, 이성진·전유환) × 거치식·적립식 예금 업종별 분석 통합 정리", "sub"), Spacer(1, 6 * mm),
          P("돈독 프로젝트 · 정리 민영 · 2026-09-29 · 대구·경북 법인 11,018곳, 2023-01~2025-12", "sub"), Spacer(1, 36 * mm),
          P("두 파트의 결과를 나란히 놓고, 같은 데이터에서 관계를 새로 계산했다. 새 계산은 기존 사양을 그대로 쓴 탐색적 분석이다. "
            "집계만 담았고 법인 ID·개별 잔액은 없다. 파트3 수치는 팀원 문서와 저장소 결과표를 옮겼고, 팀원 코드는 수정 없이 재실행해 확인했다.", "note"),
          PageBreak()]

    # 1. 한 장 요약
    s += [P("1. 한 장 요약", "h1"),
          table([["", "요구불 신호 (민영)", "여신·업종 (파트3)"],
                 ["질문", "수출이 꺾이면 외환노출 법인의 통장(요구불예금)이 줄어드나", "수출이 꺾일 때 노출·비노출 법인의 운전자금 대출이 다르게 움직이나"],
                 ["핵심 수치", f"수출 YoY 10%p 하락 시 노출 법인 요구불 6개월 {lpv('요구불', 6):+.1f}%, 12개월 {lpv('요구불', 12):+.1f}% (Holm 후 h=2~12 유의)",
                  "노출−비노출 차이 β₃ −0.231 (10%p 하락당 약 2.3%p), p 0.094 (약한 증거), 사전추세 p 0.095"],
                 ["방법", "전체 표본 국소투영, 법인 FE + 달력월 더미, 법인·월 이중 군집, 사전등록·Holm", "성향점수 매칭 1:3 후 회귀, 법인 FE + 연도 더미, 이중 군집, 8·9단계 강건성"],
                 ["은행 신호로 선별", "지역×월 평균 신호의 하락기 포착률 51.1% (신호 켜짐 비율 53.0%와 비슷)", "대시보드 신호 ① 선별력 부족, ② 부분 지지"]],
                [26, 74, 74]), Spacer(1, 2 * mm),
          table([["", "거치식·적립식 예금 업종별 LP (팀원, 수정본)"],
                 ["질문", "수출 충격 때 외환노출 법인의 저축성 예금이 비노출 법인과 다르게 움직이나 (업종별)"],
                 ["핵심 수치", "1차 금속 거치식 β₃ −0.021/−0.048/−0.094 (h=3/6/12, 수출 1%p당, 모두 p&lt;0.05) = 수출이 나쁠 때 노출 법인이 상대적으로 늘림. 효과는 계좌 개설·유지(보유 확률 12개월 약 +17%p)에서 나옴"],
                 ["방법", "전체 법인 패널, 법인 FE + 연월 FE + 외환노출×월 더미, 법인 군집(본)·이중 군집(강건성), 업종 3개는 사후 선택 → 탐색적"]],
                [26, 148]), Spacer(1, 4 * mm), P("두 결과를 이어 보면", "h2")]
    s += bullets([
        f"수출이 꺾이면 <b>노출 법인의 통장 잔고는 줄고</b>(민영), 비슷한 비노출 법인과 비교해 <b>노출 법인은 운전자금 대출을 줄이지 않는 쪽</b>(파트3)이다. "
        "통장은 빠지는데 대출은 유지되는 모양으로, 노출 법인이 유동성 부족을 대출로 메우는 그림과 맞는다. 다만 파트3는 약한 증거·사전추세 의심이라 인과로 말하지 않는다.",
        f"같은 식(민영 사양)에 종속변수만 바꾸면, 운전자금의 노출−비노출 차이는 6개월 {lpv('운전자금', 6, '차이_효과%'):+.2f}%p로 <b>파트3와 같은 방향</b>이지만 작고 유의하지 않다. "
        "파트3의 '매칭 없이 회귀 보정만 하면 −0.149, p 0.25'와 같은 모습이다.",
        f"월별 지역 평균으로 보면 노출 법인 요구불 변화는 원 수출 YoY(r {r_dep_raw:.2f})보다 <b>영업일수 차이(r {r_dep_biz:.2f})</b>와 더 같이 움직이고, 보정 YoY와는 r {r_dep_adj:.2f}다. "
        "같은 달 신호는 달력 영향이 크다.",
        f"저축성 예금은 반대로 움직인다. 같은 식으로 보면 수출 YoY 10%p 하락 시 노출 법인의 적립식은 비노출보다 6개월 {sv('전체', '적립식', '로그', 6):+.1f}%, 거치식은 12개월 {sv('전체', '거치식', '로그', 12):+.1f}% 더 늘어난다. "
        "<b>요구불은 줄고 저축성 예금은 느는</b> 재배치 모양이고, 팀원의 1차 금속 결과도 같은 사양에서 크기까지 비슷하게 재현된다.",
        f"개별 법인 신호는 두 파트 모두 '가려내는 힘이 약하다'는 쪽이다. 파트3 신호 ②의 조건 ②는 원 YoY에서 성립하고(95% 구간 [{main_cmp['원']['didCI'][0]:+.1f}, {main_cmp['원']['didCI'][1]:+.1f}]), "
        f"보정 YoY에서는 0을 포함한다([{main_cmp['보정']['didCI'][0]:+.1f}, {main_cmp['보정']['didCI'][1]:+.1f}])."])
    s.append(PageBreak())

    # 2. 설계 비교
    s += [P("2. 두 파트는 무엇이 같고 무엇이 다른가", "h1"),
          P("같은 은행 데이터와 같은 모집단에서 출발하지만, 종속변수·비교 방식·달력 처리가 다르다. 결과를 읽을 때 이 차이를 함께 봐야 한다."),
          table([["항목", "요구불 신호 (민영)", "여신·업종 (파트3)", "관계"],
                 ["모집단", "대구·경북, 금융·보험업 제외 11,018곳 (β 재검정은 11,036곳)", "같음 (11,018곳, 팀원 코드 재실행으로 확인)", "같음"],
                 ["노출 정의", "외환 수출·수입 실적 36개월 중 1회 이상 1,032곳", "같음 (외환노출 1,032곳, 수출노출 523곳)", "같음"],
                 ["충격", "지역 수출 YoY ÷ 10, 영업일수 성분 제거(보정)", "지역 수출 YoY (원), 8-6에서 영업일수 통제항 추가", "달력 처리 방식 다름"],
                 ["종속변수", "ln(요구불+1) 누적 변화 t−1→t+h (h=0~12)", "운전자금대출 로그 변화 t−1→t+6 (주), t+3 (참고)", "계정·기간 다름"],
                 ["비교 방식", "전체 표본, 법인 FE + 달력월 더미", "성향점수 매칭 1:3, 법인 FE + 연도 더미", "파트3가 대조군을 더 좁게 맞춤"],
                 ["표준오차", "법인·월 이중 군집, t(G월−1)", "법인·월 이중 군집", "같음"],
                 ["다중검정", "시차 13개 Holm (사전등록)", "9단계 4개 Holm, 8-5 검정 횟수 정리", "둘 다 사전 고정"],
                 ["계수 부호", f"{B13} = 노출 법인 전체 반응", f"{B3} &lt; 0 = 노출 법인이 상대적으로 유지", "해석 방향 확인 필요"]],
                [22, 56, 56, 40])]
    s.append(PageBreak())

    # 3. 민영 파트
    s += [P("3. 요구불 신호 분석 정리 (민영)", "h1"),
          table([["분석", "무엇을 봤나", "결과"],
                 ["요구불 반응 (β 재검정)", "수출 YoY 10%p 하락 시 노출·비노출 법인 요구불 누적 변화, h=0~12",
                  f"노출 6개월 {lpv('요구불', 6):+.1f}%, 12개월 {lpv('요구불', 12):+.1f}%. Holm 후 h=2~12 유의, 사전추세 없음. 노출−비노출 차이는 h=6~8에서 유의"],
                 ["포착률 (달력 보정)", "요구불↓+할인어음↑ 지역×월 평균 신호가 수출 하락기에 켜지는가", "보정 후 포착률 51.1%, 신호 켜짐 비율 53.0%, 위약 p 0.736 — 무작위 수준과 비슷"],
                 ["하락기·증가기 비교", "깊은 하락기와 증가기의 요구불 반응 크기 차이", "직접 검정 18번 모두 p ≥ 0.05, Holm 후 유의 칸 없음"],
                 ["전자부품 업종 운전자금 (표7)", "전자부품 운전자금이 수출·생산 충격에 반응하는가", "두 충격 모두 같은 방향으로 유의, 위약 검정 미충족 → 탐색적"],
                 ["매칭 모델", "공식 수출 통계로 단계 판정 + 은행 데이터로 기업 분류 → 상품 제안", "대구·경북 36개월 실데이터 적용, 추천 화면 구축 (마케팅 제안 시점용, 조기경보 아님)"]],
                [36, 62, 76]), Spacer(1, 3 * mm), P("요구불 반응 (주 사양)", "h2"),
          table([["h", "노출 (%)", "비노출 (%)", "차이 (%p)", "노출 Holm p", "차이 Holm p"]] +
                [[h, f"{lpv('요구불', h):+.2f}", f"{lpv('요구불', h, '비노출_효과%'):+.2f}", f"{lpv('요구불', h, '차이_효과%'):+.2f}",
                  f"{float(LP[(LP['계정'] == '요구불') & (LP['h'] == h)]['p13_holm'].iloc[0]):.3f}",
                  f"{float(LP[(LP['계정'] == '요구불') & (LP['h'] == h)]['p3_holm'].iloc[0]):.3f}"] for h in [0, 2, 3, 6, 9, 12]],
                [16, 30, 30, 30, 34, 34]),
          P("수치 = 수출 YoY 10%p 하락 시 h개월 뒤까지 누적 로그 변화 ×100. 출처: 민영_β합검정.md, 민영_포착률보정.md, 민영_비대칭재확인.md, 민영_표7재확인.md", "note"),
          PageBreak()]

    # 4. 파트3
    s += [P("4. 여신·업종 분석 정리 (파트3, 이성진·전유환)", "h1"),
          P("비슷한 조건의 비노출 법인은 수출이 나쁜 시기에 운전자금을 줄이고, 노출 법인은 유지한다. 방향은 여러 통제에서 유지되지만 유의성은 경계선이고, 충격 이전에도 같은 차이가 보인다(팀원 결론)."),
          Image(f["p3"], width=172 * mm, height=74 * mm),
          P("점선 = 사전 판정 기준(주 결과의 절반). 주황 = 충격 전 6개월(사전추세). p는 법인·월 이중 군집.", "note"),
          table([["단계", "결과"],
                 ["준비 (1·2단계)", "11,018곳·271,460행 패널, 환율 파일 오류 발견·교체. 업종별로 부호가 상쇄돼 평균은 무반응"],
                 ["핵심 비교 (3·4단계)", "매칭 1:3 후 β₃ −0.231, p 0.094. 대조군 7가지·이중강건 모두 −0.23~−0.29. 매칭 없이 회귀 보정만 하면 −0.149, p 0.25"],
                 ["세부 확인 (5·6단계)", "대출 세부 항목 20개, 업종 분류표 모두 근거 없음 (검정력 부족, 최소검출효과 약 6배)"],
                 ["반론 점검 (8단계)", "환율·영업일수·기준금리·달력월·출발 시점으로 설명되지 않음. 검정력 39%. 사전추세 p 0.095"],
                 ["새 질문 (9단계)", "하락 국면·노출 강도·수출 회사만·균형 패널 모두 Holm 후 근거 없음"],
                 ["실무 연결", "대시보드 신호 ① (요구불+운전자금) 선별력 부족, ② (요구불+할인어음) 부분 지지"]],
                [34, 140]),
          P("출처: 파트3_분석전체_정리.pdf, Don-Ddok_Data outputs/tables/part3_loan_industry", "note"), PageBreak()]

    # 5. 관계 1
    s += [P("5. 관계 ① 같은 식으로 본 세 계정 — 요구불·운전자금·할인어음", "h1"),
          P("민영 사양(전체 표본, 법인 FE + 달력월 더미, 보정 충격, 이중 군집, Holm)에 종속변수만 바꿨다. 운전자금·할인어음은 그 계정을 한 번이라도 보유한 법인만 넣었다. "
            "결과를 본 뒤 식을 바꾸지 않았다(탐색적)."),
          Image(f["lp"], width=172 * mm, height=60 * mm),
          P("파란 선 = 외환노출 법인(β₁+β₃)과 95% 구간, 점 = Holm 보정 후 유의, 회색 점선 = 비노출 법인(β₁).", "note"),
          table([["계정", "h", "노출 (%)", "비노출 (%)", "차이 (%p)", "노출 Holm p", "법인"]] +
                [[k, h, f"{lpv(k, h):+.2f}", f"{lpv(k, h, '비노출_효과%'):+.2f}", f"{lpv(k, h, '차이_효과%'):+.2f}",
                  f"{float(LP[(LP['계정'] == k) & (LP['h'] == h)]['p13_holm'].iloc[0]):.3f}",
                  f"{int(LP[(LP['계정'] == k) & (LP['h'] == h)]['법인'].iloc[0]):,}"]
                 for k in ["요구불", "운전자금", "할인어음"] for h in [3, 6, 12]],
                [22, 12, 26, 26, 26, 30, 24]),
          P("읽는 법", "h2")]
    s += bullets([
        "요구불만 뚜렷하게 줄어든다. 운전자금은 노출·비노출 모두 거의 0 근처이고, 차이는 파트3와 같은 방향(노출이 상대적으로 유지)이지만 작고 유의하지 않다.",
        "파트3의 차이(약 2.3%p)는 매칭으로 비슷한 비노출 법인만 대조군으로 남겼을 때 드러난다. 전체 표본에서는 크기가 줄어드는 점이 두 파트에서 같다(파트3 회귀 보정 −0.149, 민영 사양 +0.8%p).",
        "할인어음은 보유 법인이 약 360곳뿐이라 구간이 넓다. 노출 법인에서 줄어드는 방향이지만 확정할 수 없다(파트3 5단계 '할인어음 선행 확인되지 않음'과 같은 결론)."])
    s.append(PageBreak())

    # 6. 관계 2
    s += [P("6. 관계 ② 지역×월 상관 — 수출·달력·계정 변화", "h1"),
          P(f"대구·경북 × 월({J['n_region_month']}개)마다 노출·비노출 법인의 3개월 로그 변화 평균을 구해 수출 YoY, 영업일수 차이와의 상관을 봤다. "
            "* 표시는 p &lt; 0.05. 표본이 작아 상관 크기는 참고로만 본다."),
          Image(f["corr"], width=150 * mm, height=118 * mm), P("읽는 법", "h2")]
    s += bullets([
        f"원 수출 YoY와 영업일수 차이의 상관은 {CORR.loc['원YoY', '영업일수차']:.2f}다. 보정 YoY는 영업일수와 거의 무관하다({CORR.loc['보정YoY', '영업일수차']:.2f}).",
        f"노출 법인 요구불 변화는 영업일수 차이와 {r_dep_biz:.2f}(p {p_dep_biz:.3f}), 원 YoY와 {r_dep_raw:.2f}, 보정 YoY와 {r_dep_adj:.2f}다. "
        "같은 달 수출과 통장이 함께 움직이는 부분의 상당수가 영업일수 차이에서 온다.",
        f"운전자금 변화는 수출·영업일수와 모두 상관이 작다(|r| ≤ {max(abs(CORR.loc[['원YoY', '보정YoY', '영업일수차'], ['노출_운전자금', '비노출_운전자금']].values.ravel())):.2f}). "
        "월 단위 평균에서는 대출 반응이 보이지 않고, 파트3처럼 6개월 누적·매칭 비교에서만 드러난다.",
        f"월 평균으로 보면 요구불과 운전자금은 뚜렷하게 반대로 움직인다(노출 {CORR.loc['노출_요구불', '노출_운전자금']:.2f}, 비노출 {CORR.loc['비노출_요구불', '비노출_운전자금']:.2f}). "
        "통장이 빠지는 달에 대출이 느는 대체 관계가 지역 전체 수준에서 보인다. 다만 수출·영업일수와는 상관이 작아, 수출과 무관한 공통 월 변동(결제일·분기말 등)일 수 있다.",
        "그래서 같은 달 신호로 둔화를 판정하는 것보다 3·6개월 누적과 보정 통계를 쓰는 편이 달력 영향을 덜 받는다(파트3 대시보드 주석과 같은 방향)."])
    s.append(PageBreak())

    # 7. 관계 3·4
    s += [P("7. 관계 ③ 같은 법인 안에서 통장과 대출", "h1"),
          P("운전자금 보유 이력 법인의 6개월 로그 변화로 Δ요구불과 Δ운전자금의 순위상관(Spearman)을 봤다. 수출 감소 달은 보정 YoY 기준."),
          table([["집단", "국면", "법인-월", "법인", "Spearman", "요구불 −10% 이상 + 대출 유지 비율"]] +
                [[r["집단"], r["국면"], f"{r['법인월']:,}", f"{r['법인']:,}", f"{r['Spearman']:+.3f}", f"{r['요구불↓10%+·대출유지 비율(%)']:.1f}%"] for r in J["firm"]],
                [22, 30, 26, 20, 26, 50])]
    s += bullets([
        "통장과 대출은 같은 법인 안에서 약하게 반대로 움직인다(−0.04~−0.07). 통장이 줄 때 대출이 조금 느는 모양이지만, 노출·비노출, 감소·증가 달 사이 차이는 거의 없다.",
        "파트3 대시보드 신호 ①의 모양(통장 −10% 이상 + 대출 유지)은 어느 집단·국면에서도 31~35%로 비슷하게 나타난다. 파트3의 '선별력 부족'과 같은 결과다."])
    s += [Spacer(1, 4 * mm), P("8. 관계 ④ '수출 감소 달' 기준에 따른 신호 결과", "h1"),
          P("파트3 조합 신호 점검(요구불 3개월 −10% + 할인어음 증가)을 팀원 코드·통과 기준 그대로 쓰고, 수출 감소 달 정의만 원 YoY와 보정 YoY로 바꿨다. "
            "두 기준은 72개 지역×월 중 8개 달에서 판정이 반대다."),
          Image(f["cal"], width=172 * mm, height=64 * mm)]
    s += bullets([
        "조건 ②(수출 감소 달에 수출 거래처의 초과분이 더 크다)는 원 YoY에서 세 표본 모두 성립하고, 보정 YoY에서는 95% 구간이 0을 포함한다. 조건 ①(1.5배)은 두 기준 모두 미달.",
        "대시보드 기준월 2025-01은 경북이 원 −16.2%, 보정 +1.6%인 달(설 연휴로 영업일 4일 적음)이다. 보정 기준이면 이 달 충족 수출 거래처가 106곳 → 38곳.",
        "파트3 회귀(8-6)는 영업일수를 통제항으로 넣어 주 결과가 유지됐다. 영업일수 영향은 대시보드·신호 점검처럼 '같은 달 판정'에서 크게 나타난다."])
    s.append(PageBreak())

    # 9. 저축성 예금 업종별 분석 정리
    s += [P("9. 거치식·적립식 예금 업종별 분석 정리 (팀원, 수정본)", "h1"),
          P("리뷰를 반영해 대조군을 넣고(전체 법인 패널, β₃ = 노출−비노출 격차), 팀 표준 노출 정의·달력 통제(외환노출×월 더미)를 적용한 수정본이다. "
            "세 업종은 원본 결과를 본 뒤 고른 것이라 모든 결과가 탐색적이다. 계수는 수출 증감률 1%p당, β₃ &lt; 0 = 수출이 나쁠 때 노출 법인 예금이 상대적으로 늘어남."),
          table([["계정", "업종", "처치/대조", "h=3", "h=6", "h=12"],
                 ["거치식", "기타 기계·장비", "19 / 21", "+0.0107 (0.525)", "−0.0206 (0.358)", "−0.0821** (0.036)"],
                 ["거치식", "도매·상품 중개", "20 / 85", "+0.0012 (0.843)", "+0.0040 (0.679)", "−0.0001 (0.996)"],
                 ["거치식", "1차 금속", "19 / 11", "−0.0209** (0.019)", "−0.0477** (0.000)", "−0.0941** (0.000)"],
                 ["적립식", "기타 기계·장비", "12 / 22", "−0.0040 (0.675)", "−0.0044 (0.725)", "−0.0164 (0.449)"],
                 ["적립식", "도매·상품 중개", "25 / 103", "−0.0035 (0.410)", "−0.0101 (0.189)", "−0.0084 (0.437)"],
                 ["적립식", "1차 금속", "11 / 14", "−0.0135 (0.121)", "−0.0224* (0.065)", "−0.0320 (0.171)"]],
                [20, 34, 24, 32, 32, 32]),
          P("괄호 = p값 (법인 군집). 출처: 예금 분석이랑 변수정리.pdf (수정본)", "note")]
    s += bullets([
        "1차 금속 거치식은 이중 군집·지역×연월 FE·달력 통제 제외·leave-one-out에서 유지되고, 사전추세(30개 중 0개 유의)도 없다. 2023 기준 처치에서는 약해진다.",
        "잔액이 양수인 관측만 쓰면 거치식 효과가 0 근처로 사라진다. 보유 여부 LP에서는 h=3/6/12 모두 유의(−0.0043/−0.0090/−0.0169) → 거치식은 '계좌를 여느냐·유지하느냐', 적립식은 '잔액 규모'의 차이로 해석.",
        "원본의 세 유형 분류(기계=자금 소진형, 도매=방어형, 1차 금속=구조적 무반응형)는 대조군을 넣은 β₃ 기준으로 지지되지 않는다(팀원 결론)."])
    s.append(PageBreak())

    # 10. 관계 ⑤
    s += [P("10. 관계 ⑤ 요구불은 줄고 저축성 예금은 는다", "h1"),
          P("민영 요구불 사양(법인 FE + 달력월 더미, 보정 충격, 이중 군집)에 종속변수만 거치식·적립식으로 바꿨다. 계정마다 한 번이라도 잔액이 있는 법인만. "
            "막대 = 수출 YoY 10%p 하락 시 노출−비노출 누적 차이(%), * = p &lt; 0.05 (보정 전). 탐색적."),
          Image(f["sav"], width=172 * mm, height=64 * mm),
          P("팀원 결과를 민영 사양으로 다시 보면 (1차 금속 거치식)", "h2"),
          table([["", "h=3", "h=6", "h=12"],
                 ["팀원 (원 충격, 연월 FE, 1%p당 β₃ ×10 → 10%p 하락 시 %)", "+20.9", "+47.7", "+94.1"],
                 ["민영 사양 (보정 충격, 달력월 더미, 이중 군집)",
                  f"{sv('1차 금속', '거치식', '로그', 3):+.1f} (p {sv('1차 금속', '거치식', '로그', 3, 'p_차이'):.3f})",
                  f"{sv('1차 금속', '거치식', '로그', 6):+.1f} (p {sv('1차 금속', '거치식', '로그', 6, 'p_차이'):.3f})",
                  f"{sv('1차 금속', '거치식', '로그', 12):+.1f} (p {sv('1차 금속', '거치식', '로그', 12, 'p_차이'):.3f})"],
                 ["보유 확률 차이 (%p) — 팀원 / 민영 사양", f"+4.3 / {sv('1차 금속', '거치식', '보유', 3) * 100:+.1f}",
                  f"+9.0 / {sv('1차 금속', '거치식', '보유', 6) * 100:+.1f}", f"+16.9 / {sv('1차 금속', '거치식', '보유', 12) * 100:+.1f}"]],
                [74, 33, 33, 34]), P("읽는 법", "h2")]
    s += bullets([
        f"전체 업종에서 노출 법인의 <b>요구불은 줄고</b>({sv('전체', '요구불', '로그', 6):+.1f}%, 6개월) <b>적립식은 늘어난다</b>({sv('전체', '적립식', '로그', 6):+.1f}%, p {sv('전체', '적립식', '로그', 6, 'p_차이'):.3f}). "
        f"거치식도 12개월에 {sv('전체', '거치식', '로그', 12):+.1f}%(p {sv('전체', '거치식', '로그', 12, 'p_차이'):.3f})로 같은 방향이다. 적립식은 보유 확률도 늘어난다.",
        "요구불 → 저축성 예금 재배치로 볼 수 있다. 쓸 돈(요구불)을 줄이고 남은 자금을 묶어 두면서, 대출은 유지하는(파트3) 유동성 관리 모양과 맞는다. 인과로 말하지 않는다.",
        "1차 금속 거치식은 다른 충격 정의·군집 방식에서도 크기와 방향이 비슷하게 나온다(10%p 하락 시 12개월 약 +94%). 다만 법인 30곳뿐이고, 로그 크기는 계좌 개설(0↔양수)에서 커지므로 보유 확률로 읽는 편이 맞다(팀원 해석과 같음).",
        "업종별 차이가 크다. 기타 기계는 요구불이 크게 줄고(6개월 약 −9%) 거치식 보유가 늘며, 도매는 뚜렷한 움직임이 없다. 업종 선택이 사후라 경향으로만 본다.",
        "요구불 계정 자체의 보유 여부는 변하지 않는다. 요구불 반응은 계좌 해지가 아니라 잔액 감소다."])
    s.append(PageBreak())

    # 11. 통합 해석
    s += [P("11. 통합 해석 — 세 분석이 함께 가리키는 것", "h1"),
          table([["흐름", "근거", "강도"],
                 ["① 수출 둔화 판정", "공식 수출 통계(영업일수 보정). 같은 달 원 YoY는 달력 영향이 큼 (관계 ②·④)", "두 파트 모두 달력 영향 인지"],
                 ["② 노출 법인 통장 감소", "요구불 6개월 −3.9%, 12개월 −5.0%, Holm 후 유의 (민영)", "유의"],
                 ["③ 노출 법인 대출 유지", "매칭 비교 β₃ −0.231, p 0.094, 사전추세 p 0.095 (파트3). 전체 표본에서는 같은 방향·작음 (관계 ①)", "약한 증거"],
                 ["③' 노출 법인 저축성 예금 증가", "적립식 6개월 +8%(p 0.03), 거치식 12개월 +17%(p 0.05). 1차 금속 거치식은 보유 확률로 뚜렷 (팀원·관계 ⑤)", "탐색적"],
                 ["④ 통장↓ + 대출 유지 = 유동성 보전", "②와 ③을 합친 해석. 월 평균에서는 요구불과 운전자금이 반대로 움직이지만(관계 ②), 같은 법인 안의 동행은 약함 (관계 ③)", "해석 (인과 아님)"],
                 ["⑤ 개별 법인 선별", "포착률 무작위 수준(민영), 신호 ① 선별력 부족·② 부분 지지(파트3), 보정 기준에서 ② 조건 미충족 (관계 ④)", "약함"]],
                [44, 96, 34]), Spacer(1, 4 * mm), P("함께 쓸 수 있는 표현과 조심할 표현", "h2"),
          table([["함께 쓸 수 있는 표현", "조심할 표현"],
                 ["수출이 꺾이면 외환노출 법인의 요구불예금이 줄어든다 (6개월 약 −3.9%)", "은행 데이터가 수출 둔화를 예측한다"],
                 ["비슷한 비노출 법인에 비해 노출 법인은 운전자금을 유지하는 경향 (약한 증거)", "노출 법인이 자금 압박으로 대출을 늘렸다 · 수출 충격이 대출을 바꿨다"],
                 ["통장은 빠지고 대출은 유지되는 모양이라, 운전자금·한도 상담 수요와 맞닿는다", "대시보드가 위험 거래처를 찾아낸다"],
                 ["같은 달 신호는 영업일수 영향이 커서 3·6개월 누적·보정 통계를 쓴다", "특정 달(예: 2025-01) 수치를 경기 판단 근거로 단독 제시"],
                 ["개별 법인 신호는 연락 순서를 정하는 참고 자료다", "효과가 없다 (검정력 부족과 구별)"]],
                [87, 87]), PageBreak()]

    # 12. 남은 일
    s += [P("12. 함께 정할 것과 남은 확인", "h1")]
    s += bullets([
        "<b>수출 감소 기준 통일:</b> 대시보드·신호 점검의 '수출 감소 달'을 원 YoY로 둘지, 보정 YoY 또는 3개월 누적으로 바꿀지. 어느 쪽이든 화면과 보고서에 기준을 적는다.",
        "<b>대시보드 세 번째 조건:</b> 화면 설명은 '할인어음 3개월 증가', internal_demo_summary.py는 '운전자금 6개월 ≥ −5%'. 실제 규칙 확인.",
        "<b>계수 부호 말 맞추기:</b> 파트3 β₃ &lt; 0 = '노출 유지, 비노출 감소'. 민영 차이(β₃)는 요구불에서 음수 = '노출이 더 감소'. 같은 기호라도 계정별 해석 방향이 다르다.",
        "<b>예금 분석과 사양 맞추기:</b> 예금 업종별 분석은 연월 FE·원 충격·법인 군집(본 추정), 요구불 분석은 달력월 더미·보정 충격·이중 군집. 1차 금속 결과는 두 사양에서 비슷했지만, 보고서에서는 한 사양으로 통일하는 편이 좋다.",
        "<b>β₃ 단위·부호:</b> 예금 분석 β₃ &lt; 0 = 수출이 나쁠 때 노출 법인 예금이 늘어남(충격 1%p당). 요구불 보고 수치는 '10%p 하락 시 변화'로 부호를 뒤집어 쓴다. 한 표에 모을 때 단위와 방향을 맞춘다.",
        "<b>시연 기준월:</b> 2025-01은 영업일 효과가 큰 달. 두 기준에서 모두 감소인 달로 바꾸는 방안.",
        "<b>보고 일정:</b> 분석 동결 10/2, 보고서·PPT 10/6 18:00 (파트3 문서 기준)."])
    s += [Spacer(1, 4 * mm), P("파일", "h2"),
          table([["구분", "경로", "내용"],
                 ["통합 분석 코드", "code/integrated_analysis.py", "관계 ①~③ (같은 식 3계정, 지역×월 상관, 법인 동행)"],
                 ["저축성 예금", "code/integrated_savings.py, 통합분석/savings_lp.csv", "관계 ⑤ (요구불·거치식·적립식, 업종 3개, 로그·보유)"],
                 ["달력 기준 비교", "code/compare_part3.py, 분석결과/비교_파트3/", "팀원 함수 그대로, 수출 감소 달 정의만 교체"],
                 ["팀원 코드", "external/Don-Ddok_Data (원본), external/part3_work (재실행)", "수정 없이 재실행, 은행 파생 파일은 로컬 전용"],
                 ["결과표", "분석결과/통합분석/*.csv, integrated.json", "집계만"],
                 ["이 문서", "code/build_integrated_pdf.py", "재생성: integrated_analysis.py → integrated_savings.py → build_integrated_pdf.py"]],
                [30, 76, 68])]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title="요구불 신호 × 여신·업종 × 저축성 예금 통합 정리", author="민영 (돈독 프로젝트)")
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("저장:", OUT)


if __name__ == "__main__":
    build()
