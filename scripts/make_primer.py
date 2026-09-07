# -*- coding: utf-8 -*-
"""Typeset the primer that this project's questions asked for.

Every section is anchored to a moment in the project where the concept actually
bit, because a definition read cold is forgotten and a definition attached to a
number you spent a week chasing is not. Content lives in BLOCKS below; the code
under it only lays the page out.

Usage:
    python scripts/make_primer.py --output "C:/Users/HuangSiQi/Desktop/xxx.pdf"
"""

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

FONTS = {
    "Han": "C:/Windows/Fonts/msyh.ttc",
    "HanBold": "C:/Windows/Fonts/msyhbd.ttc",
    "Mono": "C:/Windows/Fonts/consola.ttf",
    "MonoBold": "C:/Windows/Fonts/consolab.ttf",
}

INK = colors.HexColor("#14171B")
MUTED = colors.HexColor("#4A5058")
FAINT = colors.HexColor("#8A929C")
RULE = colors.HexColor("#D8DDE3")
ACCENT = colors.HexColor("#1F5FAE")
WARM = colors.HexColor("#C1552A")
PANEL = colors.HexColor("#F4F6F8")


def register_fonts():
    for name, path in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily("Han", normal="Han", bold="HanBold",
                                  italic="Han", boldItalic="HanBold")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="MonoBold",
                                  italic="Mono", boldItalic="MonoBold")


def styles():
    base = dict(fontName="Han", wordWrap="CJK", textColor=INK)
    return {
        "title": ParagraphStyle("title", fontName="HanBold", fontSize=23, leading=32,
                                textColor=INK, spaceAfter=6, wordWrap="CJK"),
        "subtitle": ParagraphStyle("subtitle", fontSize=11.5, leading=19,
                                   textColor=MUTED, spaceAfter=3, **{k: v for k, v in base.items()
                                                                     if k != "textColor"}),
        "meta": ParagraphStyle("meta", fontName="Mono", fontSize=8.5, leading=14,
                               textColor=FAINT, spaceAfter=2),
        "h1": ParagraphStyle("h1", fontName="HanBold", fontSize=17, leading=25,
                             textColor=INK, spaceBefore=4, spaceAfter=9, wordWrap="CJK"),
        "h2": ParagraphStyle("h2", fontName="HanBold", fontSize=12.5, leading=19,
                             textColor=INK, spaceBefore=15, spaceAfter=5, wordWrap="CJK"),
        "h3": ParagraphStyle("h3", fontName="HanBold", fontSize=10.8, leading=17,
                             textColor=ACCENT, spaceBefore=11, spaceAfter=3, wordWrap="CJK"),
        "body": ParagraphStyle("body", fontSize=10.2, leading=17.6, alignment=TA_JUSTIFY,
                               spaceAfter=7, **base),
        "bullet": ParagraphStyle("bullet", fontSize=10.2, leading=17.2, leftIndent=13,
                                 bulletIndent=3, spaceAfter=4, **base),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=8.6, leading=13.4,
                               textColor=INK, leftIndent=9, spaceBefore=3, spaceAfter=8),
        "anchor": ParagraphStyle("anchor", fontSize=9.8, leading=16.4, textColor=MUTED,
                                 fontName="Han", wordWrap="CJK"),
        "cap": ParagraphStyle("cap", fontSize=9, leading=14.5, textColor=MUTED,
                              fontName="Han", wordWrap="CJK", spaceBefore=3, spaceAfter=9),
        "tcell": ParagraphStyle("tcell", fontSize=9.1, leading=14, fontName="Han",
                                wordWrap="CJK", textColor=INK),
        "thead": ParagraphStyle("thead", fontSize=8.7, leading=13, fontName="HanBold",
                                wordWrap="CJK", textColor=MUTED),
    }


S = None
WIDTH = A4[0] - 42 * mm


def P(text, kind="body"):
    return Paragraph(text, S[kind])


def BUL(items):
    return [Paragraph("• " + item, S["bullet"]) for item in items]


def CODE(lines):
    body = "<br/>".join(line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
                        for line in lines)
    table = Table([[Paragraph(body, S["code"])]], colWidths=[WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBEFORE", (0, 0), (0, -1), 2, RULE),
    ]))
    return table


def ANCHOR(text):
    """The moment in the project where this concept actually mattered."""
    inner = Paragraph("<b>你在项目里遇到它的时候</b><br/>" + text, S["anchor"])
    table = Table([[inner]], colWidths=[WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBF3EE")),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, WARM),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [table, Spacer(1, 9)]


def TBL(header, rows, widths=None):
    data = [[Paragraph(c, S["thead"]) for c in header]]
    data += [[Paragraph(c, S["tcell"]) for c in row] for row in rows]
    if widths is None:
        widths = [WIDTH / len(header)] * len(header)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.1, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 1.1, INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [table, Spacer(1, 10)]


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Mono", 7.6)
    canvas.setFillColor(FAINT)
    if doc.page > 1:
        canvas.drawString(21 * mm, 13 * mm, "深度学习与视频取证基础")
        canvas.drawRightString(A4[0] - 21 * mm, 13 * mm, str(doc.page))
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(21 * mm, 17 * mm, A4[0] - 21 * mm, 17 * mm)
    canvas.restoreState()


def build(destination, blocks):
    doc = BaseDocTemplate(str(destination), pagesize=A4,
                          leftMargin=21 * mm, rightMargin=21 * mm,
                          topMargin=20 * mm, bottomMargin=22 * mm,
                          title="深度学习与视频取证基础", author="")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=decorate)])
    doc.build(list(blocks))


def main():
    global S
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    register_fonts()
    S = styles()
    from primer_content import blocks  # noqa: E402
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    build(destination, blocks(P, BUL, CODE, ANCHOR, TBL, Spacer, PageBreak, WIDTH))
    size = destination.stat().st_size
    print("wrote {}  ({:,} bytes)".format(destination, size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
