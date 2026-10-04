# -*- coding: utf-8 -*-
"""돈독 · 민영 파트 최종 정리 PDF — 집계만 사용 (법인 ID·개별 잔액 없음)

숫자는 각 폴더에 커밋된 결과 파일에서 읽는다:
  harmonized (judgment.csv, robust_summary.csv, industry_summary.csv, mde.csv, results_all.csv)
  did_demand_deposit, did_loan_robust (results_table.csv, pretrend*.csv)
  mechanism_inflow (results.csv, pretrend.csv, mde.csv, judgment.csv)
  exploratory (tests_log.csv)
  industry_shock (stage1_results.csv, stage1_import_results.csv), firm_shock (judgment.csv)
  trade_extra_mde (mde_prospect.csv, mde_by_h.csv), trade_finance_mde (mde_prospect.csv)
그 밖의 앞선 재검증 수치(포착률·비대칭·표7·β합·파트3 비교)는 분석결과\\*.md에 적힌 값을 옮겼다.
출력: C:\\test\\outputs\\FINAL\\돈독_민영파트_최종정리.pdf
실행: py -3.11 build_final_pdf.py
"""
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

O = r"C:\test\outputs"
OUT = os.path.join(O, "FINAL", "돈독_민영파트_최종정리.pdf")
pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunB", r"C:\Windows\Fonts\malgunbd.ttf"))
registerFontFamily("Malgun", normal="Malgun", bold="MalgunB", italic="Malgun", boldItalic="MalgunB")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
ST = {
    "title": ParagraphStyle("title", fontName="MalgunB", fontSize=22, leading=30, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="Malgun", fontSize=11, leading=17, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="MalgunB", fontSize=15, leading=21, textColor=INK, spaceBefore=2, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="MalgunB", fontSize=11, leading=16, textColor=INK, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Malgun", fontSize=9.3, leading=14.5, textColor=INK, spaceAfter=4),
    "b": ParagraphStyle("b", fontName="Malgun", fontSize=9.3, leading=14.5, textColor=INK, leftIndent=10, bulletIndent=0, spaceAfter=2),
    "note": ParagraphStyle("note", fontName="Malgun", fontSize=7.8, leading=11.5, textColor=INK2, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Malgun", fontSize=7.8, leading=10.5, textColor=INK, alignment=TA_LEFT),
    "cellh": ParagraphStyle("cellh", fontName="MalgunB", fontSize=7.8, leading=10.5, textColor=INK),
}


def fix(t):
    return str(t).replace("−", "-").replace("⚠", "※")     # 맑은 고딕에 U+2212·⚠ 없음


def P(t, s="p"):
    return Paragraph(fix(t), ST[s])


def B(items):
    return [Paragraph(fix(t), ST["b"], bulletText="•") for t in items]


def T(rows, widths):
    data = [[P(c, "cellh" if i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    for i in range(2, len(rows), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f8f6")))
    t.setStyle(TableStyle(st))
    return t


def IMG(path, w_mm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=w_mm * mm, height=w_mm * mm * ih / iw)


def footer(c, d):
    c.saveState(); c.setFont("Malgun", 7.5); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(18 * mm, 10 * mm, "돈독 · 민영 파트 최종 정리 · 2026-09-30 · 집계만 수록")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page)); c.restoreState()


def f2(x, d=2):
    return f"{x:+.{d}f}"


def appendix_h(TH):
    """부록: 무역 추가 A·B·C의 시차별 MDE (k 근사, 두 방향 중 큰 쪽)"""
    k = TH[TH["방식"] == "근사(k)"]
    hs = sorted(k["h"].unique())
    order = [("A 결제채널", "할인어음"), ("A 결제채널", "외상매출채권담보"), ("A 결제채널", "기업구매자금"),
             ("B 수출중단", "요구불"), ("B 수출중단", "운전자금"), ("B 수출중단", "적립식"),
             ("B 수출시작", "요구불"), ("B 수출시작", "운전자금"), ("B 수출시작", "적립식"), ("C 순노출×환율", "요구불")]
    rows = [["분석", "계정", "단위"] + [f"h={h}" for h in hs]]
    for a, acct in order:
        x = k[(k["분석"] == a) & (k["계정"] == acct)].set_index("h")
        rows.append([a, acct, x["단위"].iloc[0].replace("% (", "").rstrip(")")] + [f"{x.loc[h, 'MDE_큰쪽']:.1f}%" for h in hs])
    n = k.drop_duplicates(["분석", "h"]).set_index(["분석", "h"])["월"]
    months = " / ".join(f"{a.split()[0]} {int(n.loc[(a, hs[0])])}→{int(n.loc[(a, hs[-1])])}개월" for a in ["A 결제채널", "B 수출중단", "C 순노출×환율"])
    return [PageBreak(), P("부록. 시차별 MDE (무역 추가 A·B·C, k 근사, 진단용)", "h1"),
            P("회귀 없이 MDE만 계산했다. 이 표는 진단용이며 판정 시차(h=6)를 바꾸는 근거로 쓰지 않는다. 10% 아래로 내려가는 칸은 없다.", "p"),
            T(rows, [26, 26, 34] + [17.6] * len(hs)),
            P(f"월 수(h=1→12): {months}. B 사건 수(요구불, h=1→12): 중단 168→98, 시작 178→119.", "note"),
            P("<b>가정:</b> k_h는 h마다 harmonized 요구불 C1의 같은 h 실제 SE에 맞췄다(h=1, 3, 6, 9, 12: 1.256, 1.142, 1.375, 1.662, 1.836). "
              "A는 C1과 같은 설계라 그대로 맞지만, <b>설계가 다른 B(법인별 사건 더미)와 C(순노출 × 전국 공통 환율 시계열)에 같은 k_h를 쓴 것은 가정이다.</b> "
              "군집 구조가 달라 실제 SE와 어긋날 수 있다. 보조로 계산한 샌드위치 값은 trade_extra_mde\\prospect_by_h.md에 있다.", "note"),
            P("<b>처리 기록:</b> 외상매출채권담보 h=1의 샌드위치 SE는 이중 군집 분산이 음수(준정부호 아님)로 나와 처음에 0으로 잘리면서 MDE 0.0%로 출력됐다. "
              "실제 값이 아니므로 \"계산 불가\"로 바로잡았다(k 근사 값 25.6%는 영향 없음).", "note")]


def main():
    H = os.path.join(O, "harmonized")
    J = pd.read_csv(os.path.join(H, "judgment.csv"))
    R = pd.read_csv(os.path.join(H, "results_all.csv"))
    RS = pd.read_csv(os.path.join(H, "robust_summary.csv"))
    IS = pd.read_csv(os.path.join(H, "industry_summary.csv"))
    MD = pd.read_csv(os.path.join(H, "mde.csv"))
    MR = pd.read_csv(os.path.join(O, "mechanism_inflow", "results.csv"))
    MP = pd.read_csv(os.path.join(O, "mechanism_inflow", "pretrend.csv"))
    MM = pd.read_csv(os.path.join(O, "mechanism_inflow", "mde.csv"))
    FR = pd.read_csv(os.path.join(O, "mechanism_inflow", "flow_ratio.csv"))
    EX = pd.read_csv(os.path.join(O, "exploratory", "tests_log.csv"))
    DD = pd.read_csv(os.path.join(O, "did_demand_deposit", "results_table.csv"))
    DDp = pd.read_csv(os.path.join(O, "did_demand_deposit", "pretrend_joint.csv"), index_col=0)
    DL = pd.read_csv(os.path.join(O, "did_loan_robust", "results_table.csv"))
    DLo = pd.read_csv(os.path.join(O, "did_loan_robust", "pretrend_original_method.csv"))
    IX = pd.read_csv(os.path.join(O, "industry_shock", "stage1_results.csv"))
    IM = pd.read_csv(os.path.join(O, "industry_shock", "stage1_import_results.csv"))
    FJ = pd.read_csv(os.path.join(O, "firm_shock", "judgment.csv"))
    TE = pd.read_csv(os.path.join(O, "trade_extra_mde", "mde_prospect.csv"))
    TH = pd.read_csv(os.path.join(O, "trade_extra_mde", "mde_by_h.csv"))
    TFM = pd.read_csv(os.path.join(O, "trade_finance_mde", "mde_prospect.csv"))
    g = lambda df, **kw: df.loc[pd.concat([df[k] == v for k, v in kw.items()], axis=1).all(axis=1)].iloc[0]

    s = []
    # ── 표지·요약
    s += [Spacer(1, 18 * mm), P("돈독 · 민영 파트 최종 정리", "title"),
          P("수출 충격과 법인 은행 거래: 요구불 신호 재검증에서 세 계정 공통 사양 재추정까지", "sub"), Spacer(1, 4 * mm),
          P("iM뱅크 디지털뱅커아카데미 3조(돈독) · 2026-09-30 · 대구·경북 법인 11,036곳 × 2023-01~2025-12 (36개월) · 모든 수치는 집계", "note"),
          Spacer(1, 8 * mm), P("한 장 요약", "h1")]
    s += B(["<b>요구불:</b> 지역 수출이 꺾이면 외환노출 법인의 요구불예금이 비노출 법인보다 상대적으로 줄어든다. 수출 YoY 10%p 하락 시 "
            "h=6 기준 −3.6%, h=6·7·8·11에서 Holm 후 유의. 통제를 강화하고(지역×연월 FE, 노출×영업일수차) 매칭·원YoY·양수 관측만 써도 남는다. 공통 사양 등급 <b>확정</b>(사전 기준을 모두 통과, 인과 확정 아님).",
            "<b>운전자금 대출:</b> 파트3의 \"노출 법인 대출 상대 증가\"(+2.34%)는 매칭 표본에서만 보이고 공통 사양에서는 +0.6%(p 0.59). "
            "사전추세 연장 가능성까지 있어 <b>근거 없음 · 착시 가능성</b>. 다만 원래 크기를 잡을 검정력이 50%라 효과가 없다는 뜻은 아니다.",
            "<b>저축성 예금:</b> 적립식 보유 확률 +2.5%p(p 0.015, <b>약한 증거</b>), 잔액 차이는 양수 잔액 상태의 진입·유지에서 나옴(계좌 자체의 개설·해지는 확인 못함, 잔액 +8.5%, p 0.035). 거치식 <b>근거 없음</b>.",
            "<b>경로:</b> 요구불 감소는 입금 감소로 확인되지 않았고(입금 h=6 −0.3%, p 0.74), 지역 수출 YoY와 개별 기업 수출 실적의 연결도 약하다. "
            "수입만 하는 법인(509곳)에서도 요구불이 −4.2%로 줄어, 수출 기업만의 반응이 아니다. "
            "업종 수출·수입 충격, 법인 자신의 외환 실적 충격으로도 경로가 확인되지 않았다(8·9절). "
            "그래서 요구불 결과는 <b>지역 수출 경기와의 동행 관계</b>로 표현한다.",
            "<b>영업일수:</b> 기존 은행 신호의 포착률과 파트3 조합 신호는 영업일수 보정 뒤 무작위 수준이 되거나 구간이 0을 포함했다(<b>착시로 판명</b>). "
            "공식 수출 YoY는 영업일 1일당 +4.46%p 움직인다.",
            "<b>방식:</b> 모든 재검정은 결과를 보기 전에 사양·판정 규칙을 문서로 고정하고 git에 커밋했다. 결과를 본 뒤 바꾼 것은 에러 수정과 속도 개선이며 모두 기록했다. "
            "결과를 본 뒤 정한 해석 기준 2개(요구불 E23의 SE 기준, 거치식 반대 부호 사전추세)는 '사후'로 표시했다. 탐색 분석은 판정표와 별도다."])
    s += [Spacer(1, 4 * mm), P("결과 요약 (증거 사다리)", "h2"),
          T([["결과", "수치 (10%p 하락 시 노출−비노출)", "강도"],
             ["요구불", "h=6 기준 −3.6%, h=6·7·8·11에서 Holm 후 유의", "가장 강함 (확정)"],
             ["적립식", "h=6 +8.48%, p 0.035 / 보유 확률 +2.5%p, p 0.015", "약한 증거 · Holm 후 비유의 · 추정치가 MDE보다 작아 크기 과대추정 가능"],
             ["운전자금", "C1 +0.6%, p 0.59 / 매칭(C2)에서만 +2.7%", "근거 없음 · 착시 가능성 (사전추세 연장 가능성, 원래 크기 검정력 50%)"],
             ["거치식", "C1 +7.8%, p 0.14", "근거 없음"],
             ["영업일수 보정의 영향", "기존 포착률·조합 신호 ② 구간이 보정 뒤 무작위 수준/0 포함", "착시로 판명 (방법론 발견)"]],
            [30, 72, 72]), PageBreak()]

    # ── 1. 데이터와 공통 설계
    s += [P("1. 데이터와 공통 설계", "h1")]
    s += [T([["항목", "내용"],
             ["원자료", "(iM뱅크) 2026 교육용 법인 익명데이터.xlsx — 365,988행 × 70열, 월말 잔액·월 흐름·구간값(좌수·건수). 금액은 유효숫자 2자리 정도로 반올림. 대출 실행·상환액은 없음"],
             ["분석 패널", "df_ready.csv (원자료 파생, 로컬 전용) 중 대구·경북, 2023-01~2025-12: 법인 11,036곳, 법인×월 271,915행"],
             ["노출", "36개월 중 수출 또는 수입 금액 > 0인 달이 1회 이상 = 1,034곳 (금액 또는 건수 기준과 같은 수). 수출 실적 있음 525곳, 수입만 509곳"],
             ["충격", "지역(대구·경북) 월별 수출액 전년동월비. 보정YoY = 원YoY − 4.458 × 영업일수 전년동월차 (72개 지역×월, r 0.45)"],
             ["영업일수", "workdays_2021_2025.csv (팀 배포 파일, 파트3와 동일)"],
             ["공통 사양 C1", "ln(y_{t+h}+1) − ln(y_{t−1}+1) = 법인 FE + 지역×연월 FE + β3(노출×X) + γ(노출×영업일수차). 윈저 1/99, 법인·연월 이중 군집, p = t(G_min−1), Holm h=1~12, h=−6~12"],
             ["보고 단위", "수출 YoY 10%p 하락 시 노출−비노출 차이(%) = (exp(−0.1·β3) − 1)×100 (β3는 100%p당). 보유 여부는 %p"],
             ["판정 등급", "확정(Holm 후 유의 h 있음 + 사전추세 결합 p ≥ 0.10 + 사전추세 연장 없음) / 약한 증거(h=6 p < 0.10) / 근거 없음. \"확정\"은 사전 기준 통과이며 인과 확정이 아님"]],
            [30, 144])]
    s += [P("원래 파트별 사양은 서로 달랐다 (요구불: 달력월 더미·보정YoY·10%p당 / 운전자금: 매칭 1:3·연도 더미·원YoY·h=6 단일 / 예금: 업종 3개·연월 FE·원YoY·법인 군집·1%p당). "
            "그래서 세 계정을 C1 하나로 다시 쟀다(harmonized). 운전자금 표본 9,541곳 vs 파트3 9,528곳의 차이 13곳은 모두 금융 및 보험업(파트3가 제외).", "note"), PageBreak()]

    # ── 2. 작업 흐름
    s += [P("2. 작업 흐름", "h1"),
          T([["단계", "질문", "결과", "위치"],
             ["포착률 달력 보정", "은행 신호가 수출 하락 국면을 잡나", "착시로 판명: 포착률 51.1% < 신호 켜짐 53.0%, 위약 p 0.736 (원본도 무작위 수준)", "분석결과\\민영_포착률보정.md"],
             ["비대칭 재확인", "요구불 하방 비대칭이 있나", "없음 유지: 직접 검정 18번 모두 p ≥ 0.05", "분석결과\\민영_비대칭재확인.md"],
             ["표7 재확인", "전자부품 운전자금 셀", "두 셀 모두 탐색적 (업종생산 위약 p 0.129 / 0.060)", "분석결과\\민영_표7재확인.md"],
             ["β1+β3 재검정", "노출 법인 요구불 반응", "6개월 −3.9%, 12개월 −5.0% (Holm 후 h=2~12). 차이 β3는 h=6~8만", "분석결과\\민영_β합검정.md"],
             ["파트3 비교", "영업일수 보정이 조합 신호를 바꾸나", "조건 ② 구간 [+2.2, +11.6] → [−0.9, +7.8]. 2025-01 충족 106 → 38곳", "분석결과\\비교_파트3"],
             ["통합 분석", "요구불·대출·저축성의 관계", "요구불↓, 대출 상대 유지, 저축성↑(탐색적). 통합 PDF 12쪽 (이후 공통 사양 재추정으로 갱신)", "분석결과\\통합분석"],
             ["대출 강건성", "같은 통제에서 −0.231 유지되나", "유지(−0.304)하나 매칭 표본에서만, 사전추세 −0.380 → 약한 증거", "outputs\\did_loan_robust"],
             ["요구불 DID", "통제 강화 뒤 β3 유지되나", "유지: h=6 +0.361 (p 0.003), 사전추세 p 0.907, 매칭 B +0.440", "outputs\\did_demand_deposit"],
             ["공통 사양 재추정", "세 계정을 같은 잣대로", "요구불 확정 / 운전자금 근거 없음·착시 가능성 / 적립식 약한 증거 / 거치식 근거 없음", "outputs\\harmonized"],
             ["MDE", "안 보인 게 효과가 없어서인가", "운전자금 원래 크기 검정력 50%, 요구불 C1 87%", "harmonized 부록 B"],
             ["메커니즘", "요구불 감소가 입금 감소인가", "근거 없음 (입금 −0.3%, p 0.74). 충격 변수-개별 수출 연결 약함", "outputs\\mechanism_inflow"],
             ["탐색 분석", "다른 변수에도 신호가 있나", "48개 검정, q < 0.10 6개 중 5개는 요구불 하위 집단, 요구불 밖은 체크카드 1개", "outputs\\exploratory"],
             ["업종 충격", "업종 수출·수입이 기업 실적에 닿나", "수출·수입 모두 1단계 연결 약함 → 2단계 미실행", "outputs\\industry_shock"],
             ["기업별 외환 실적 충격", "법인 자신의 무역 변화에 계정이 반응하나", "h=6 확정 없음, 적립식 약한 증거 (인과 아님)", "outputs\\firm_shock"],
             ["사전 MDE 판단", "추가 분석이 검정 가능한 크기인가", "무역 추가 3종 전부 10% 초과, 결제 금융은 구조 1 초과 · 구조 2 단위 주의", "outputs\\trade_extra_mde, trade_finance_mde"]],
            [26, 38, 74, 36]), PageBreak()]

    # ── 3. 공통 사양 판정표
    s += [P("3. 공통 사양 판정표 (harmonized C1, h=6)", "h1")]
    rows = [["계정", "원래 결과", "C1 결과", "이중 군집 p", "Holm 최소 p", "사전추세 LP p", "원본방식 β3pre", "등급", "원래 대비"]]
    for _, x in J.iterrows():
        rows.append([x["계정"], str(x["원래(h=6,%)"]).replace(" (p", "% (p"), f"{float(x['C1(h=6,%)']):+.2f}%", x["이중군집 p"], x["Holm 최소 p"], x["사전추세 LP p"],
                     x["원본방식 β3pre"] + (" ⚠연장" if x["연장 표시"] != "–" else ""), x["등급"], x["원래 대비"]])
    s += [T(rows, [16, 24, 16, 15, 16, 16, 26, 17, 28]),
          P("요구불 원래 결과는 기존 β3(h=6, 10%p당 +0.0362). 기존에 보고한 −3.9%/−5.0%는 β1+β3(노출 법인 전체 반응)라 비교 대상이 아니다. "
            "거치식은 원본 방식 사전추세가 반대 부호이면서 유의(β3pre +1.813, p 0.022)해 평균 회귀 가능성이 있다(규칙 밖 사후 관찰).", "note"),
          IMG(os.path.join(H, "irf_c1.png"), 174),
          P("보고용 모형 (h=6, 판정에 쓰지 않음)", "h2")]
    rr = [["계정"] + [c for c in RS.columns if c != "계정"]]
    for _, x in RS.iterrows():
        rr.append([x["계정"]] + [str(x[c]).replace("⚠부호 반대", "⚠") if pd.notna(x[c]) else "–" for c in RS.columns if c != "계정"])
    s += [T(rr, [16, 22, 22, 22, 24, 24, 22, 22]),
          P("C2 매칭 1:3 · E23 2023년 기준 노출(2024년 이후) · EX 수출 실적만 · POS t−1·t+h 잔액 양수만 · HOLD 보유 여부(%p) · RAW 원YoY. ⚠ = C1과 부호 반대. "
            "RAW = C1(노출×영업일수차가 있으면 원YoY와 보정YoY의 β3가 같음). 요구불 E23은 SE가 1.91배로 커져 확인·부정 불가(검정력 손실 가능성).", "note"),
          PageBreak()]

    # ── 4. 요구불 · 대출 강건성 그림
    a6, b6 = g(DD, 모형="A", h=6), g(DD, 모형="B", h=6)
    l1 = g(DL, 모형="L1", h=6); l2 = g(DL, 모형="L2", h=6)
    s += [P("4. 요구불 DID와 대출 강건성 (통제 강화판)", "h1"),
          T([["분석", "h=6 β3", "10%p 하락 시 차이", "p", "사전추세", "판정"],
             ["요구불 A (전체)", f2(a6['β3'], 3), f"{a6['10%p하락_차이%']:+.2f}%", f"{a6['p']:.3f}", f"LP {DDp.loc['A', 'p_chi2']:.3f}", "유지"],
             ["요구불 B (매칭 1:3)", f2(b6['β3'], 3), f"{b6['10%p하락_차이%']:+.2f}%", f"{b6['p']:.3f}", f"LP {DDp.loc['B', 'p_chi2']:.3f}", "강건성 (같은 부호)"],
             ["대출 L1 (매칭, 판정)", f2(l1['β3'], 3), f"{l1['10%p하락_차이%']:+.2f}%", f"{l1['p']:.3f}", f"원본 방식 {DLo.iloc[0]['β3']:+.3f} (p {DLo.iloc[0]['p']:.3f})", "유지 → 해석: 약한 증거"],
             ["대출 L2 (매칭 없음)", f2(l2['β3'], 3), f"{l2['10%p하락_차이%']:+.2f}%", f"{l2['p']:.3f}", "", "참고"]],
            [36, 18, 26, 14, 44, 36]),
          P("대출 L1 판정은 대출 강건성 SPEC(매칭 표본) 기준이다. 최종 등급은 3절 공통 사양(C1) 기준 '근거 없음 · 착시 가능성'이다. "
            "대출 L1은 규칙상 \"유지\"지만, 매칭 표본에서만 나타나고(L2 p 0.622) 충격 전 6개월 차이(−0.380)가 사후 6개월 차이(−0.304)보다 커서 수출 충격의 효과로 해석하기 어렵다.", "note"),
          IMG(os.path.join(O, "did_demand_deposit", "irf_plot.png"), 170), IMG(os.path.join(O, "did_loan_robust", "irf_plot.png"), 170), PageBreak()]

    # ── 5. MDE
    s += [P("5. 최소검출효과 (MDE) — 안 보인 것은 효과가 없어서인가", "h1")]
    mr = [["대상 (h=6)", "MDE 주 값", "추정치", "원래 크기 검정력", "원래 크기 ≥ MDE"]]
    for _, x in MD.iterrows():
        u = "%p" if x["단위"] == "%p" else "%"
        mr.append([f"{x['계정']} {x['모형']}", f"{x['MDE 주 값']:+.2f}{u}", f"{x['추정치(환산)']:+.2f}{u}",
                   "–" if pd.isna(x.get("원래 크기 검정력")) else f"{x['원래 크기 검정력'] * 100:.0f}%",
                   "–" if pd.isna(x.get("원래 크기 ≥ MDE")) else ("예" if x["원래 크기 ≥ MDE"] in (True, "True") else "아니오")])
    s += [T(mr, [36, 30, 30, 36, 36])]
    s += B(["요구불 C1: 원래 크기(−3.56%)를 87% 확률로 잡는 설계였고 실제로 잡혔다.",
            "운전자금 C1: 원래 크기(+2.34%)의 검정력 50%. \"근거 없음\"은 효과가 없다는 뜻이 아니다. 매칭 C2도 35%로, 원래 결과 p 0.094가 경계선인 것과 맞는다.",
            "요구불 E23 · 운전자금 C1은 95% 신뢰구간이 0과 원래 크기를 함께 포함한다. E23은 보고용이라 등급에 영향이 없고, 운전자금 C1은 주 사양 + 매칭 의존 + 사전추세 연장으로 착시 가능성.",
            "거치식·적립식 잔액 MDE +15%·+11%(0↔양수 전환 포함, 크기 해석 주의). 적립식은 추정치가 MDE보다 작아, 유의하더라도 크기는 과대추정됐을 수 있다."])
    s += [PageBreak()]

    # ── 6. 메커니즘
    i6, o6 = g(MR, 분석="입금", h=6), g(MR, 분석="출금", h=6)
    ip = MP[MP["분석"] == "입금"].iloc[0]
    s += [P("6. 메커니즘 — 요구불 감소는 입금 감소인가", "h1"),
          T([["h=6, 10%p 하락 시 차이", "환산", "p", "MDE (감소 방향)"],
             ["요구불 잔액 (C1, 참고)", "−3.57%", "0.003", "−3.24%"],
             ["요구불 입금 (판정)", f"{i6['환산']:+.2f}%", f"{i6['p']:.3f}", f"{MM.iloc[0]['MDE_감소방향%']:+.2f}%"],
             ["요구불 출금 (보고만)", f"{o6['환산']:+.2f}%", f"{o6['p']:.3f}", f"{MM.iloc[1]['MDE_감소방향%']:+.2f}%"]],
            [60, 36, 30, 40]),
          P(f"판정 <b>근거 없음</b> — \"입금 감소로는 확인되지 않아, 요구불 감소의 경로는 이 데이터로 특정하지 못했다\". 입금 LP 사전추세 p {ip['p_chi2']:.3f}, 원본 방식 β3pre {ip['원본방식_β3pre']:+.3f} (p {ip['원본방식_p']:.3f}).", "p"),
          P("입금 %와 잔액 %는 기준이 달라 직접 비교할 수 없다. 노출 법인의 월 흐름 ÷ 잔액(기술 통계, 회귀 없음):", "p"),
          T([["비율", "p25", "중앙값", "p75"],
             ["월 요구불 입금 / 잔액", f"{FR.iloc[0]['p25']:.2f}", f"{FR.iloc[0]['중앙값']:.2f}", f"{FR.iloc[0]['p75']:.2f}"],
             ["월 요구불 출금 / 잔액", f"{FR.iloc[1]['p25']:.2f}", f"{FR.iloc[1]['중앙값']:.2f}", f"{FR.iloc[1]['p75']:.2f}"]], [60, 30, 30, 30]),
          P(f"참고값: h=6 입금 차이 {i6['환산']:+.2f}% × 입금/잔액 중앙값 {FR.iloc[0]['중앙값']:.2f} → 잔액의 약 {i6['환산'] * FR.iloc[0]['중앙값']:+.1f}% "
            f"(p25~p75 비율로 {i6['환산'] * FR.iloc[0]['p25']:+.1f}% ~ {i6['환산'] * FR.iloc[0]['p75']:+.1f}%). 입금 추정치의 신뢰구간이 넓어 이 근사도 불확실하다. "
            "잔액·입출금 금액의 단위와 정의(월말 잔액 / 월 합계 여부)는 데이터 설명서를 찾지 못해 <b>확인 필요</b>.", "note"),
          P("충격 변수 점검(EX 노출 525곳, 3개월 합계 수출 실적): h=0~3 Holm 후 p 모두 1.000 → \"지역 수출 YoY와 개별 기업 수출의 연결이 이 데이터에서 약하다. 충격 변수 해석에 한계\". "
            "월 단위 참고 모형은 h=1·2가 유의하나 크기가 −0.11~−0.16%로 매우 작다. "
            "가능한 설명(검정 아님): 지역 수출 통계는 대형 수출 기업 비중이 커서, 은행 거래 중소 법인의 수출과 연결이 약할 수 있다.", "p"),
          IMG(os.path.join(O, "mechanism_inflow", "irf_inflow_outflow.png"), 170),
          P("팀원 주장 점검 요약: ① 제조업 재현 — 탐색 분석에서 요구불·제조업 −5.1%(q 0.090)로 보이나 집단 간 차이는 유의하지 않음. "
            "② 요구불·거치식의 다른 반응 — 같은 충격에서 방향이 다름(요구불↓, 거치식↑ 비유의). ③ 결제 계좌 해석 — 입금액 변수로는 요구불 감소의 경로를 확인하지 못했다. "
            "다만 입출금 변수가 잔액 변화를 온전히 담는지 불확실해(잔액 변화와 입금−출금 상관 0.589), 결제 계좌 해석을 부정하는 근거로 쓰지는 않는다. "
            "④ \"요구불↓·거치식↓·대출 유지 동시 불성립\" — 결론은 맞지만 막히는 곳은 거치식(방향이 증가).", "note"),
          PageBreak()]

    # ── 7. 탐색 분석
    f6 = EX[EX["FDR묶음"]]
    disc = f6[f6["q_BH"] < 0.10].sort_values("q_BH")
    dr = [["묶음", "결과변수 · 표본", "h=6", "p", "q", "사전추세"]]
    for _, x in disc.iterrows():
        lpv = "계산 불가" if pd.isna(x["LP결합p"]) else f"LP {x['LP결합p']:.3f}"
        flag = " ⚠반대 부호 유의" if x["사전추세_반대부호유의"] else ""
        dr.append([x["묶음"], f"{x['결과변수']} · {x['표본']}", f"{x['환산']:+.2f}%", f"{x['p']:.3f}", f"{x['q_BH']:.3f}", lpv + flag])
    s += [P("7. 탐색 분석 (별도, 판정표에 영향 없음)", "h1"),
          P(f"harmonized C1 사양 그대로 결과변수·표본만 바꿔 h=6 검정 {len(f6)}개에 BH-FDR. 돌린 회귀 786행을 모두 기록했다. q < 0.10 {len(disc)}개, p < 0.05 {int((f6['p'] < 0.05).sum())}개. q < 0.10 기준이라 발견 중 약 10%는 잘못된 발견일 수 있다.", "p"),
          T(dr, [12, 70, 18, 14, 14, 46])]
    s += B(["발견 6개 중 5개는 확정된 요구불 결과가 큰 하위 집단(제조업, 전담 Y, 규모 하위, 고객등급 우수)과 수입만 하는 법인에서 보이는 것이다. 집단 간 차이 7개는 모두 p ≥ 0.10 → 이질성 근거 없음.",
            "수입만 하는 법인(509곳)에서도 요구불 −4.20%로 수출 실적 법인(−2.84%)보다 작지 않다. 요구불 반응이 수출 쪽에서만 나오지 않는다(메커니즘 결과와 같은 방향).",
            "요구불 밖의 유일한 발견은 체크카드 사용금액(−3.69%, q 0.090, 노출 90곳, 사전추세 계산 불가) — 다음 사전등록 후보.",
            "여신 8개는 p < 0.05도 없다. 14개 칸은 LP 결합검정 공분산이 양정치가 아니어서 \"계산 불가\"로 표시했다(사전등록 분석들은 모두 정상)."])
    s += [PageBreak()]

    # ── 8. 외부 통계 연결 시도
    def link_row(df, model, name, target):
        m = df[df["모형"] == model]
        sg = "모두 +" if (m["β3"] > 0).all() else ("모두 −" if (m["β3"] < 0).all() else "섞임 (0 근처)")
        return [name, target, sg, f"{m['p_holm_h0_3'].min():.3f}", "연결 약함 → 2단계 미실행"]
    mx = MR[MR["분석"] == "수출점검_3개월합"]
    s += [P("8. 외부 통계 연결 시도 — 공식 수출입 통계가 개별 법인에 닿나", "h1"),
          P("요구불 결과의 경로를 찾으려고, 공식 통계의 수출입 변동이 은행 거래 법인의 실제 외환 실적에 반영되는지 먼저 확인했다(1단계). "
            "연결이 확인되면 계정 반응(2단계)을 보기로 SPEC에 미리 고정했다. 종속변수는 법인 외환 실적 3개월 합계의 로그 변화, h=0~3, Holm(4개).", "p"),
          T([["충격", "1단계 대상", "h=0~3 방향", "최소 Holm p", "판정"],
             ["지역 수출 YoY (6절)", "수출 노출 법인 525곳", "작고 비유의", f"{mx['p_holm_h0_3'].min():.3f}", "연결 약함"],
             link_row(IX, "주_업종충격", "업종 수출 YoY (15개 업종, HS4→KSIC)", "수출 실적 법인 422곳"),
             link_row(IM, "주_업종수입충격", "업종 수입 YoY (13개 업종)", "수입 실적 법인 478곳")],
            [44, 36, 30, 22, 42]),
          P("업종 충격: 관세청 HS4 품목 수출입(대구·경북, 2022~2025)을 UNSD HS→CPC→ISIC→KSIC 연결표로 업종에 배분했다. "
            "업종 수출은 네 시차 모두 β > 0이지만 크기가 10%p 하락 시 −0.1~−0.2%로 작고 유의하지 않다. 업종 수입은 부호도 섞인다. "
            "KSIC 26(전자부품)을 빼도 결론이 같다. 업종 수입은 제품 기준이라 원자재를 수입하는 법인과 어긋날 수 있다(한계).", "note"),
          P("→ 지역·업종, 수출·수입 어느 충격으로도 공식 통계와 은행 거래 법인의 실제 외환 실적 사이의 연결이 확인되지 않았다.", "p"),
          Spacer(1, 6 * mm)]

    # ── 9. 기업별 외환 실적 충격
    fr = [["계정 (h=6)", "외환 실적 10% 감소 시", "MDE", "p", "Holm(6개)", "LP 사전 p (h=−6..−3)", "등급 · 표시"]]
    for _, x in FJ.iterrows():
        u = x["단위"]
        lp = "계산 불가" if pd.isna(x["LP_p_chi2_h6_3"]) else f"{x['LP_p_chi2_h6_3']:.3f}"
        fr.append([x["계정"].replace("_HOLD", " HOLD"), f"{x['환산']:+.2f}{u}", f"{x['MDE_환산']:.2f}{u}", f"{x['p']:.3f}", f"{x['p_holm6']:.3f}", lp,
                   f"{x['등급']} · {x['표시']}"])
    s += [P("9. 기업별 외환 실적 충격 — 법인 자신의 무역 변화에 계정이 반응하나", "h1"),
          P("외부 통계를 거치지 않고, 은행 데이터 안의 법인별 외환 실적(수출+수입) 변화를 충격으로 썼다. 노출 법인 안의 비교라 3절의 노출−비노출 β3와 다른 양이다. "
            "충격 (a′) = ln(100·S_t+1) − ln(100·S_{t−12}+1), S = t−2~t 외환 실적 합(0.01 단위라 100을 곱함). 강건성은 50% 급감 더미. "
            "충격이 2024-03부터 정의되어 h=6은 16개월(t(15))이다. MDE는 추정치를 보기 전에 먼저 계산했다.", "p"),
          T(fr, [22, 26, 18, 14, 16, 24, 54]),
          *B(["<b>h=6 확정 없음.</b> 적립식만 약한 증거(p 0.049)이나 Holm 후 비유의이고 추정치가 MDE보다 작다. 여섯 계정 모두 |추정치| < MDE.",
             "요구불은 외환 실적이 준 달 전후(h=0~2)에만 같이 줄고(h=0 p 0.009, h=2 p 0.033) 6개월 뒤에는 남지 않는다. 계정 안 Holm 최소 p 0.393.",
             "강건성(급감 더미)은 여섯 계정 모두 0 근처. \"역인과 의심\"(사전추세 p < 0.10) 표시는 주 충격에서 없다.",
             "<b>모든 결과는 인과가 아니다.</b> 자금 사정이 나빠져 무역이 줄었을 수 있다(급감 발생률: 요구불 하위 5분위 45.5% → 상위 5분위 36.0%)."]),
          PageBreak()]

    # ── 10. 검정 가능성 사전 판단
    def te(a, acct):
        x = TE[(TE["분석"] == a) & (TE["계정"] == acct)].set_index("방식")
        return x.loc["근사(k)", "MDE_큰쪽"], x.loc["샌드위치", "MDE_큰쪽"], x.loc["근사(k)", "판단"]
    pr = [["분석", "대상", "h=6 MDE (k 근사 / 샌드위치)", "단위", "판단"]]
    for acct in ["할인어음", "외상매출채권담보", "기업구매자금"]:
        k_, w_, j_ = te("A 결제채널", acct)
        pr.append(["무역 추가 A 결제채널", acct, f"{k_:.1f}% / {w_:.1f}%", "지역 수출 10%p당", j_])
    for acct in ["요구불", "운전자금", "적립식"]:
        k1, w1, j1 = te("B 수출중단", acct); k2, w2, j2 = te("B 수출시작", acct)
        pr.append(["무역 추가 B 수출 중단·시작", acct, f"중단 {k1:.0f}% / {w1:.0f}% · 시작 {k2:.0f}% / {w2:.0f}%", "사건 1회당", j1 if j1 == j2 else f"{j1} / {j2}"])
    k_, w_, j_ = te("C 순노출×환율", "요구불")
    pr.append(["무역 추가 C 순노출×환율", "요구불", f"{k_:.1f}% / {w_:.1f}%", "순노출 1, 환율 YoY 10%p당", j_])
    for st, lab, unit in [("1 harmonized C1", "결제 금융 구조 1 (노출 vs 비노출)", "지역 수출 10%p당"), ("2 firm_shock", "결제 금융 구조 2 (노출 안, 자기 실적)", "자기 외환 실적 10% 감소당")]:
        for code, nm in [("TF", "TF 무역금융"), ("PF", "PF 기업구매자금"), ("AR", "AR 외상매출채권담보"), ("SUM3", "SUM3 (주 대상)")]:
            x = TFM[(TFM["코드"] == code) & (TFM["구조"] == st)].set_index("방식")
            kk, ww = x.loc["근사(k)", "MDE_가설방향(증가)"], x.loc["샌드위치", "MDE_가설방향(증가)"]
            j = x.loc["근사(k)", "판단"]
            if st.startswith("2"):
                j = "기준 통과 (단위 주의, 주 a)"
            elif code == "TF":
                j += " (대조군 부족, 참고)"
            pr.append([lab, nm, f"{kk:.1f}% / {ww:.1f}%", unit, j])
    pr.append(["시차 진단 (h=1, 3, 6, 9, 12)", "무역 추가 A·B·C 전체", "부록 표", "–",
               "h를 1~12로 바꿔도 MDE가 10% 아래로 내려가는 분석·계정은 없었다. h가 커질수록 표본 감소, 누적 변화폭 증가, 군집 설계효과 증가로 MDE가 함께 커진다."])
    s += [P("10. 검정 가능성 사전 판단 — 추가 분석을 할 만한 크기인가", "h1"),
          P("본 분석 전에 MDE만 근사했다(회귀 없음, β 계산 안 함). 판단 기준(계산 전 고정): MDE ≤ 5% 진행 후보 / 5~10% 단일 가설만 / > 10% 진행 안 함. "
            "판단에는 두 방향 중 큰 쪽(결제 금융은 가설 방향인 잔액 증가, 이것이 큰 쪽)을 썼다. 판정 시차는 h=6으로 유지한다.", "p"),
          T(pr, [34, 30, 38, 26, 46]),
          P("근사: SE = k × SD(y 잔차) / (SD(X 잔차) × sqrt(N)). y 잔차·X 잔차는 처치 없는 귀무 모형(FE + 통제)의 잔차, k는 구조별로 요구불의 실제 SE에 맞춤(harmonized C1 1.375, firm_shock 1.275). "
            "보조로 귀무 잔차 샌드위치 SE. 기존 실제 SE 재현: k 근사 0.56~1.19배, 샌드위치 0.95~1.43배(N은 모두 일치).", "note"),
          P("주 a) 구조 2의 기준 단위(자기 외환 실적 10% 감소)는 잔차 충격 SD의 약 0.09~0.11배로 작은 충격이고, 구조 1의 10%p는 약 1.6~2.1배다. "
            "충격 1 SD당으로 맞추면 모든 코드에서 구조 1의 MDE가 더 작다(SUM3: 7.9~8.2% vs 20~23%). 5%/10% 기준은 구조 1의 단위를 전제로 만든 것이라, "
            "구조 2의 기준 통과는 숫자상으로만 그렇다. 상세는 trade_finance_mde\\prospect.md.", "note"),
          PageBreak()]

    # ── 11. 한계와 다음 단계
    s += [P("11. 한계와 다음 단계", "h1"), P("한계", "h2")]
    s += B(["충격 변동이 2개 지역 × 36개월뿐이다. 월 군집 23~35개라 이중 군집 SE가 과소추정될 수 있다.",
            "노출을 결과 기간과 같은 36개월로 정했다. 2023년 기준 노출로는 검정력이 부족해 확인·부정할 수 없다.",
            "β3는 연속형 충격의 기울기이지 특정 사건의 효과가 아니다. 비노출 법인이 수출 기업 협력사면 차이가 작게 잡힌다(파급효과).",
            "표본을 결과변수 보유 이력으로 정했다(특히 거치식 843곳, 적립식 666곳). 금액 반올림과 월 흐름 잡음으로 입금·출금 분석의 정밀도가 낮다.",
            "평균 반응은 확인했지만 개별 기업 선별력은 부족하다(기존 신호 ① 선별력 부족 판정). h=6은 충격 뒤의 반응이라 선행 신호가 아니다."])
    s += [P("다음 단계", "h2")]
    s += B(["연체·한도 소진 등 결과 데이터가 있어야 개별 기업 선별력과 은행 업무 가치를 검증할 수 있다.",
            "노출 강도(수출 비중)에 따른 용량-반응, 체크카드 신호는 각각 사전등록 후 검정한다. 판정 시차를 h=6에서 바꾸려면 별도 SPEC에 사유와 함께 미리 고정한다.",
            "발표 표현: \"지역 수출 경기와 함께 외환노출 법인 요구불이 상대적으로 준다(동행)\". 인과·조기경보 표현은 쓰지 않는다."])
    s += [P("재현과 기록", "h2")]
    s += B(["각 분석 폴더(outputs\\did_loan_robust, did_demand_deposit, harmonized, mechanism_inflow, exploratory, industry_shock, firm_shock, trade_extra_mde, trade_finance_mde)는 SPEC.md를 먼저 커밋한 git 저장소이고, CHANGES.md에 에러 수정·속도 개선을 기록했다.",
            "코드·데이터 지도와 재현 순서는 outputs\\FINAL\\INDEX.md에 있다. 원자료와 파생 패널은 로컬 전용이며 이 문서에는 집계만 실었다."])
    s += appendix_h(TH)

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="돈독 민영 파트 최종 정리", author="민영 (돈독)")
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("saved", OUT)


if __name__ == "__main__":
    main()
