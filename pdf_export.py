"""
pdf_export.py
-------------
Renders a downloadable, print-ready Investment Memo PDF combining the same
data shown on the dashboard: company overview, key financial metrics,
valuation summary, DCF output, comps analysis, and the AI-generated
narrative sections (Investment Thesis, Key Strengths, Risks, Industry
Outlook, Recommendation).

Built with reportlab's Platypus framework (flowables laid out top-to-bottom
on a page) and kaleido (via Plotly's `fig.to_image`) for chart images. PDFs
are printed on white paper, so charts get a light-themed clone of the
dashboard's dark Plotly figures rather than reusing them directly.

Nothing here imports Streamlit -- callers pass in already-computed domain
objects (CompanyData, ValuationSummary, DCFResult, CompsResult) and the raw
markdown memo string, and get back PDF bytes ready for `st.download_button`.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from comps import CompsResult
from dcf import DCFResult
from financials import CompanyData
from utils import format_currency, format_multiple_nm, format_percent, format_ratio
from valuation import ValuationSummary

# --------------------------------------------------------------------------- #
# Print-friendly color scheme (distinct from the dashboard's dark theme --
# a white page with dark bars/lines reads far better on paper/PDF viewers).
# --------------------------------------------------------------------------- #

NAVY = colors.HexColor("#0f1f3d")
GOLD = colors.HexColor("#b8860b")
BODY_GRAY = colors.HexColor("#33383f")
LIGHT_GRID = colors.HexColor("#e2e5ea")
ROW_ALT = colors.HexColor("#f4f6f9")

_CHART_NAVY = "#0f1f3d"
_CHART_GOLD = "#b8860b"
_CHART_TEAL = "#0f766e"
_CHART_BLUE = "#1d4ed8"
_CHART_RED = "#b91c1c"


# --------------------------------------------------------------------------- #
# Paragraph styles
# --------------------------------------------------------------------------- #

def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MemoTitle", parent=base["Title"], fontSize=22, textColor=NAVY,
            spaceAfter=2, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "MemoSubtitle", parent=base["Normal"], fontSize=11, textColor=BODY_GRAY,
            alignment=TA_CENTER, spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "SectionHeader", parent=base["Heading2"], fontSize=13, textColor=NAVY,
            spaceBefore=16, spaceAfter=6, borderColor=GOLD, borderWidth=0,
        ),
        "body": ParagraphStyle(
            "BodyText2", parent=base["BodyText"], fontSize=9.5, textColor=BODY_GRAY,
            leading=14, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletText", parent=base["BodyText"], fontSize=9.5, textColor=BODY_GRAY,
            leading=13, alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontSize=8, textColor=BODY_GRAY,
            alignment=TA_CENTER, spaceBefore=2, spaceAfter=10,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"], fontSize=7.5, textColor=colors.gray,
            alignment=TA_CENTER,
        ),
    }


# --------------------------------------------------------------------------- #
# Markdown (AI memo) -> reportlab flowables
# --------------------------------------------------------------------------- #

def _md_inline(text: str) -> str:
    """Convert inline **bold** / *italic* markdown to reportlab's HTML-subset markup."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _parse_memo_sections(memo_markdown: str) -> dict[str, str]:
    """
    Split the AI memo's markdown into {section_title: raw_body_text} using
    the '### Header' boundaries that ai.py's prompt instructs Claude to use.
    Falls back gracefully if a section is missing -- callers should check
    with `.get()`.
    """
    sections: dict[str, str] = {}
    current_title: Optional[str] = None
    buffer: list[str] = []

    for line in memo_markdown.splitlines():
        header_match = re.match(r"^#{1,3}\s+(.*)", line.strip())
        if header_match:
            if current_title is not None:
                sections[current_title] = "\n".join(buffer).strip()
            current_title = header_match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(buffer).strip()
    return sections


def _markdown_to_flowables(text: str, styles: dict) -> list:
    """Render a section's raw markdown body as a mix of Paragraphs and bullet ListFlowables."""
    flowables = []
    bullet_buffer: list[str] = []

    def _flush_bullets():
        if bullet_buffer:
            items = [ListItem(Paragraph(_md_inline(b), styles["bullet"]), leftIndent=6) for b in bullet_buffer]
            flowables.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=14, spaceAfter=6))
            bullet_buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet_match = re.match(r"^[-*•]\s+(.*)", line)
        if bullet_match:
            bullet_buffer.append(bullet_match.group(1))
            continue
        _flush_bullets()
        flowables.append(Paragraph(_md_inline(line), styles["body"]))

    _flush_bullets()
    return flowables


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def _make_table(data: list[list[str]], col_widths: Optional[list[float]] = None) -> Table:
    """Build a consistently-styled table: navy header row, light alternating row bands."""
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT_GRID),
        ("GRID", (0, 0), (-1, -1), 0.25, LIGHT_GRID),
    ]
    for row_idx in range(1, len(data)):
        if row_idx % 2 == 0:
            style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), ROW_ALT))
    table.setStyle(TableStyle(style))
    return table


# --------------------------------------------------------------------------- #
# Print-friendly chart rendering
# --------------------------------------------------------------------------- #

def _print_friendly(fig: go.Figure) -> go.Figure:
    """
    Return a light-themed clone of a dashboard chart suitable for a white
    PDF page. Re-themes colors/background rather than mutating trace data,
    so the original dark figure used by the live dashboard is untouched.
    """
    fig = go.Figure(fig)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#33383f", family="Helvetica, Arial, sans-serif", size=11),
        title=dict(font=dict(color="#0f1f3d", size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=45, b=40),
    )
    fig.update_xaxes(gridcolor="#e2e5ea", zeroline=False, linecolor="#c7ccd4")
    fig.update_yaxes(gridcolor="#e2e5ea", zeroline=False, linecolor="#c7ccd4")

    recolor = {"#f0b90b": _CHART_GOLD, "#2dd4bf": _CHART_TEAL, "#3b82f6": _CHART_BLUE,
               "#a78bfa": "#6d28d9", "#22c55e": "#15803d", "#ef4444": _CHART_RED}
    for trace in fig.data:
        if "marker" in trace and getattr(trace.marker, "color", None) in recolor:
            trace.marker.color = recolor[trace.marker.color]
        if "line" in trace and getattr(trace.line, "color", None) in recolor:
            trace.line.color = recolor[trace.line.color]
    return fig


def _fig_to_image(fig: go.Figure, draw_width: float = 6.3 * inch) -> Image:
    """Render a Plotly figure to PNG bytes via kaleido and wrap it as a reportlab Image."""
    png_bytes = fig.to_image(format="png", width=1000, height=460, scale=2)
    aspect = 460 / 1000
    return Image(io.BytesIO(png_bytes), width=draw_width, height=draw_width * aspect)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def generate_memo_pdf(
    company: CompanyData,
    valuation: ValuationSummary,
    memo_markdown: str,
    dcf_result: Optional[DCFResult] = None,
    comps_result: Optional[CompsResult] = None,
) -> bytes:
    """
    Assemble the full Investment Memo PDF and return it as raw bytes, ready
    to be handed to `st.download_button(data=..., mime="application/pdf")`.

    Args:
        company: CompanyData for the subject ticker (overview + statements).
        valuation: ValuationSummary for the subject company.
        memo_markdown: The AI-generated memo text from ai.generate_investment_memo,
            expected to contain '### ' headers for Business Overview,
            Investment Thesis, Key Strengths, Risks, Industry Outlook,
            Valuation Summary, and Recommendation.
        dcf_result: Optional DCFResult -- adds a DCF Output table if provided.
        comps_result: Optional CompsResult -- adds a Comparable Company
            Analysis table (on its own page) if provided.

    Returns:
        The rendered PDF as bytes.
    """
    styles = _styles()
    sections = _parse_memo_sections(memo_markdown)
    overview = company.overview
    summary_df = company.annual_summary

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.65 * inch, rightMargin=0.65 * inch,
        title=f"{overview['name']} ({overview['ticker']}) Investment Memo",
    )
    story: list = []

    # --- Header ---
    story.append(Paragraph(f"{overview['name']} ({overview['ticker']})", styles["title"]))
    story.append(Paragraph(
        f"{overview['sector']} &middot; {overview['industry']} &middot; "
        f"Prepared {datetime.now().strftime('%B %d, %Y')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=GOLD, spaceAfter=12))

    # --- Business Overview ---
    if "Business Overview" in sections:
        story.append(Paragraph("Business Overview", styles["section"]))
        story.extend(_markdown_to_flowables(sections["Business Overview"], styles))

    # --- Key Financial Metrics ---
    story.append(Paragraph("Key Financial Metrics", styles["section"]))
    metrics_data = [["Metric", "Value"], ["Market Cap", format_currency(valuation.market_cap)],
                    ["Enterprise Value", format_currency(valuation.enterprise_value)],
                    ["Revenue (LTM)", format_currency(valuation.revenue)],
                    ["EBITDA (LTM)", format_currency(valuation.ebitda)],
                    ["Net Income (LTM)", format_currency(valuation.net_income)],
                    ["Diluted EPS", format_currency(valuation.diluted_eps, decimals=2) if valuation.diluted_eps else "N/A"],
                    ["Current Share Price", format_currency(valuation.share_price, decimals=2)]]
    story.append(_make_table(metrics_data, col_widths=[3.2 * inch, 3.1 * inch]))
    story.append(Spacer(1, 10))

    if not summary_df.empty:
        try:
            from charts import income_statement_chart
            story.append(_fig_to_image(_print_friendly(income_statement_chart(summary_df))))
            story.append(Paragraph("Revenue, EBITDA, and Net Income by fiscal year.", styles["caption"]))
        except Exception:
            pass  # kaleido/chrome unavailable in this environment -- omit chart, keep the rest of the PDF

    # --- Valuation Summary ---
    story.append(Paragraph("Valuation Summary", styles["section"]))
    val_data = [["Multiple", "Value"], ["EV / Revenue", format_multiple_nm(valuation.ev_to_revenue)],
                ["EV / EBITDA", format_multiple_nm(valuation.ev_to_ebitda)],
                ["P / E", format_multiple_nm(valuation.price_to_earnings)],
                ["P / B", format_multiple_nm(valuation.price_to_book)]]
    story.append(_make_table(val_data, col_widths=[3.2 * inch, 3.1 * inch]))
    if "Valuation Summary" in sections:
        story.append(Spacer(1, 6))
        story.extend(_markdown_to_flowables(sections["Valuation Summary"], styles))

    # --- DCF Output ---
    if dcf_result is not None:
        story.append(Paragraph("Discounted Cash Flow Output", styles["section"]))
        dcf_data = [
            ["Metric", "Gordon Growth", "Exit Multiple"],
            ["Enterprise Value", format_currency(dcf_result.enterprise_value_gordon),
             format_currency(dcf_result.enterprise_value_exit_multiple)],
            ["Equity Value", format_currency(dcf_result.equity_value_gordon),
             format_currency(dcf_result.equity_value_exit_multiple)],
            ["Implied Share Price", format_currency(dcf_result.implied_share_price_gordon, decimals=2),
             format_currency(dcf_result.implied_share_price_exit_multiple, decimals=2)],
        ]
        story.append(_make_table(dcf_data, col_widths=[2.4 * inch, 2.0 * inch, 2.0 * inch]))
        if dcf_result.warning:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<i>Note: {dcf_result.warning}</i>", styles["caption"]))

    # --- Comparable Company Analysis (own page: often the widest table) ---
    if comps_result is not None and not comps_result.table.empty:
        story.append(PageBreak())
        story.append(Paragraph("Comparable Company Analysis", styles["section"]))
        header = ["Ticker", "EV/Rev", "EV/EBITDA", "P/E", "P/B"]
        comps_rows = [header]
        for ticker, row in comps_result.table.iterrows():
            comps_rows.append([
                ticker, format_multiple_nm(row["ev_to_revenue"]), format_multiple_nm(row["ev_to_ebitda"]),
                format_multiple_nm(row["price_to_earnings"]), format_multiple_nm(row["price_to_book"]),
            ])
        story.append(_make_table(comps_rows, col_widths=[1.3 * inch] + [1.35 * inch] * 4))
        if comps_result.failed_tickers:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<i>Could not load: {', '.join(comps_result.failed_tickers)}</i>", styles["caption"]
            ))

    # --- Narrative sections carried over from the AI memo ---
    for title in ("Investment Thesis", "Key Strengths", "Risks", "Industry Outlook", "Recommendation"):
        if title in sections:
            story.append(Paragraph(title, styles["section"]))
            story.extend(_markdown_to_flowables(sections[title], styles))

    # --- Footer / disclaimer ---
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRID, spaceAfter=8))
    story.append(Paragraph(
        "This document was generated by an AI-assisted investment banking dashboard for "
        "educational and portfolio purposes only. It is not a research report, investment "
        "advice, or a solicitation to buy or sell any security. Figures are sourced from "
        "public market data (Yahoo Finance) and may contain errors or omissions. "
        f"Prepared by Keira Hirst on {datetime.now().strftime('%B %d, %Y')}.",
        styles["footer"],
    ))

    doc.build(story)
    return buffer.getvalue()
