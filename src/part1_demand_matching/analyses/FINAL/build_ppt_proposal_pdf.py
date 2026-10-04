# -*- coding: utf-8 -*-
"""돈독_PPT_재구성_제안.md → PDF. build_project_report_pdf.py와 같은 파서·스타일 재사용."""
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

O = r"C:\test\outputs\FINAL"
MD = os.path.join(O, "돈독_PPT_재구성_제안.md")
OUT = os.path.join(O, "돈독_PPT_재구성_제안.pdf")
pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunB", r"C:\Windows\Fonts\malgunbd.ttf"))
registerFontFamily("Malgun", normal="Malgun", bold="MalgunB", italic="Malgun", boldItalic="MalgunB")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
ST = {
    "title": ParagraphStyle("title", fontName="MalgunB", fontSize=20, leading=27, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Malgun", fontSize=10.5, leading=16, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="MalgunB", fontSize=14.5, leading=20, textColor=INK, spaceBefore=4, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="MalgunB", fontSize=11, leading=15.5, textColor=INK, spaceBefore=9, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Malgun", fontSize=9.3, leading=14.5, textColor=INK, spaceAfter=6),
    "b": ParagraphStyle("b", fontName="Malgun", fontSize=9.3, leading=14, textColor=INK, leftIndent=11, spaceAfter=3),
    "note": ParagraphStyle("note", fontName="Malgun", fontSize=8.6, leading=13, textColor=INK2, spaceAfter=6,
                           leftIndent=8, borderColor=colors.HexColor(GRID), borderWidth=0.6, borderPadding=6,
                           backColor=colors.HexColor("#f7f7f5")),
    "cell": ParagraphStyle("cell", fontName="Malgun", fontSize=7.9, leading=11, textColor=INK, alignment=TA_LEFT),
    "cellh": ParagraphStyle("cellh", fontName="MalgunB", fontSize=7.9, leading=11, textColor=INK),
}


def fix(t):
    t = t.replace("−", "-").replace("⚠", "※")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', t)
    return t


def P(t, s="p"):
    return Paragraph(fix(t), ST[s])


def parse_table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if not re.match(r"^\s*\|?\s*:?-{2,}", ln.split("|")[1] if "|" in ln else "")]
    return rows


def make_table(rows):
    ncol = len(rows[0])
    avail = 174.0
    data = [[Paragraph(fix(c), ST["cellh"] if i == 0 else ST["cell"]) for c in r] for i, r in enumerate(rows)]

    def clen(txt):
        return len(re.sub(r"</?[a-z][^>]*>", "", txt))
    raw_w = [max(clen(r[c]) for r in rows) for c in range(ncol)]
    raw_w = [max(w, 6) for w in raw_w]
    scale = avail / sum(raw_w)
    col_widths = [w * scale * mm for w in raw_w]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f8f6")))
    t.setStyle(TableStyle(style))
    return t


def footer(c, d):
    c.saveState(); c.setFont("Malgun", 7.3); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(18 * mm, 10 * mm, "돈독 · PPT 재구성 제안 · 2026-10-01 · 팀 내부 검토용")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page)); c.restoreState()


def build():
    raw = open(MD, encoding="utf-8").read().splitlines()
    s = []
    i = 0
    bullet_buf = []
    first_h1 = True
    just_after_title = False

    def flush_bullets():
        nonlocal bullet_buf
        for b in bullet_buf:
            s.append(Paragraph("• " + fix(b), ST["b"]))
        if bullet_buf:
            s.append(Spacer(1, 3))
        bullet_buf = []

    while i < len(raw):
        ln = raw[i]
        stripped = ln.strip()
        if not stripped:
            flush_bullets(); i += 1; continue
        if stripped == "---":
            flush_bullets(); i += 1; continue
        if stripped.startswith("# "):
            flush_bullets()
            s += [Spacer(1, 8 * mm), P(stripped[2:], "title")]
            just_after_title = True
        elif stripped.startswith("## "):
            flush_bullets()
            if just_after_title:
                s += [P(stripped[3:], "sub"), Spacer(1, 6 * mm)]
            else:
                if not first_h1:
                    s.append(PageBreak())
                first_h1 = False
                s.append(P(stripped[3:], "h1"))
            just_after_title = False
        elif stripped.startswith("### "):
            flush_bullets(); s.append(P(stripped[4:], "h2"))
        elif stripped.startswith(">"):
            flush_bullets(); s.append(P(stripped.lstrip("> ").strip(), "note"))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_buf.append(stripped[2:])
        elif stripped.startswith("|"):
            flush_bullets()
            block = []
            while i < len(raw) and raw[i].strip().startswith("|"):
                block.append(raw[i]); i += 1
            i -= 1
            rows = parse_table(block)
            if rows:
                s.append(Spacer(1, 2)); s.append(make_table(rows)); s.append(Spacer(1, 6))
        elif re.match(r"^\d+\.\s", stripped):
            flush_bullets(); s.append(Paragraph(fix(stripped), ST["b"]))
        else:
            flush_bullets(); s.append(P(stripped, "p"))
        i += 1
    flush_bullets()

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="돈독 PPT 재구성 제안", author="돈독팀")
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("saved", OUT)


if __name__ == "__main__":
    build()
