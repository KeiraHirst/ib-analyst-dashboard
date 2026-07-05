"""
pitchbook.py
------------
One-page "pitch book" -- the single-slide equity research summary an
analyst hands a portfolio manager: business snapshot, price/financial
trend charts, valuation multiples, DCF-implied price, comps-implied
range, and data-driven investment highlights / key risks.

Two entry points:
  - render_pitchbook_tab(...)   Streamlit tab -- interactive, in-app view.
  - generate_pitchbook_pdf(...) A single printable PDF page, built with the
                                same reportlab/kaleido machinery as
                                pdf_export.py (imported and reused rather
                                than duplicated).

Highlights and risks are derived directly from the company's own reported
financials and (if available) its peer comps -- no AI call is required,
so this feature works even without an ANTHROPIC_API_KEY configured. If an
AI investment memo has already been generated for this ticker in the
current session, its "Key Strengths" / "Risks" bullets are used instead,
since those are more nuanced than the rule-based fallback.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import charts
from comps import CompsResult, implied_valuation_range
from dcf import DCFResult
from financials import CompanyData
from pdf_export import GOLD, LIGHT_GRID, NAVY, _fig_to_image, _print_friendly
from utils import format_currency, format_multiple_nm, format_percent, format_ratio, safe_divide
from valuation import ValuationSummary

# --------------------------------------------------------------------------- #
# Data-driven highlights & risks
# --------------------------------------------------------------------------- #

def _peer_median(comps_result: Optional[CompsResult], column: str) -> Optional[float]:
    if comps_result is None or comps_result.peer_stats.empty or column not in comps_result.peer_stats.columns:
        return None
    value = comps_result.peer_stats.loc["median", column]
    return float(value) if pd.notna(value) else None


def derive_highlights_and_risks(
    company: CompanyData, valuation: ValuationSummary, comps_result: Optional[CompsResult] = None
) -> tuple[list[str], list[str]]:
    """
    Generate investment highlight and risk bullets straight from the
    company's reported financials and (optionally) its peer comps --
    e.g. growth trajectory, margin trend, leverage, cash position, and
    relative valuation vs. peers. Pure rule-based logic, no AI call.
    """
    df = company.annual_summary
    highlights: list[str] = []
    risks: list[str] = []

    if df.empty or len(df) < 2:
        return (
            ["Insufficient multi-year financial history to derive data-driven highlights."],
            ["Insufficient multi-year financial history to derive data-driven risk flags."],
        )

    latest, prior = df.iloc[-1], df.iloc[-2]
    cagrs = company.cagr_summary(years=min(3, len(df) - 1))
    peer_ev_ebitda = _peer_median(comps_result, "ev_to_ebitda")

    # --- Highlights ---
    if cagrs.get("revenue_cagr") is not None and cagrs["revenue_cagr"] > 0.05:
        highlights.append(
            f"Revenue has compounded at a {format_percent(cagrs['revenue_cagr'])} CAGR over the last "
            f"{cagrs['years']} fiscal years."
        )
    if pd.notna(latest.get("ebitda_margin")) and latest["ebitda_margin"] > 0.20:
        highlights.append(f"Attractive profitability profile, with a trailing EBITDA margin of {format_percent(latest['ebitda_margin'])}.")
    if pd.notna(latest.get("free_cash_flow")) and latest["free_cash_flow"] > 0:
        highlights.append(
            f"Generates positive free cash flow of {format_currency(latest['free_cash_flow'])}, "
            "supporting reinvestment and capital return capacity."
        )
    if valuation.total_cash is not None and valuation.total_debt is not None and valuation.total_cash > valuation.total_debt:
        highlights.append(
            f"Net cash balance sheet ({format_currency(valuation.total_cash)} cash vs. "
            f"{format_currency(valuation.total_debt)} debt) provides financial flexibility."
        )
    if peer_ev_ebitda is not None and valuation.ev_to_ebitda is not None and 0 < valuation.ev_to_ebitda < peer_ev_ebitda:
        highlights.append(
            f"Trades at a discount to peer median EV/EBITDA ({format_ratio(valuation.ev_to_ebitda)} vs. "
            f"{format_ratio(peer_ev_ebitda)}), which may indicate relative value."
        )
    if not highlights:
        highlights.append("No standout data-driven highlights identified from reported financials alone.")

    # --- Risks ---
    if pd.notna(latest.get("revenue_growth")) and pd.notna(prior.get("revenue_growth")) and latest["revenue_growth"] < prior["revenue_growth"]:
        risks.append(
            f"Revenue growth decelerated to {format_percent(latest['revenue_growth'])} in the most recent "
            f"fiscal year, from {format_percent(prior['revenue_growth'])} previously."
        )
    if pd.notna(latest.get("ebitda_margin")) and pd.notna(prior.get("ebitda_margin")) and latest["ebitda_margin"] < prior["ebitda_margin"]:
        risks.append(f"EBITDA margin compressed year-over-year to {format_percent(latest['ebitda_margin'])}.")
    debt_to_ebitda = safe_divide(valuation.total_debt, valuation.ebitda)
    if debt_to_ebitda is not None and debt_to_ebitda > 3:
        risks.append(f"Elevated leverage at {debt_to_ebitda:.1f}x Debt/EBITDA.")
    if peer_ev_ebitda is not None and valuation.ev_to_ebitda is not None and valuation.ev_to_ebitda > peer_ev_ebitda * 1.25:
        risks.append(
            f"Trades at a premium to peer median EV/EBITDA ({format_ratio(valuation.ev_to_ebitda)} vs. "
            f"{format_ratio(peer_ev_ebitda)}), leaving limited room for multiple expansion."
        )
    if pd.notna(latest.get("free_cash_flow")) and latest["free_cash_flow"] < 0:
        risks.append("Negative free cash flow in the most recent fiscal year.")
    if not risks:
        risks.append("No specific red flags detected in reported financials; standard market and macro risks still apply.")

    return highlights, risks


def _dcf_avg_price(dcf_result: Optional[DCFResult]) -> Optional[float]:
    if dcf_result is None:
        return None
    prices = [p for p in (dcf_result.implied_share_price_gordon, dcf_result.implied_share_price_exit_multiple) if p is not None]
    return sum(prices) / len(prices) if prices else None


def _comps_price_range(company: CompanyData, comps_result: Optional[CompsResult]) -> Optional[tuple[float, float]]:
    if comps_result is None or comps_result.peer_stats.empty:
        return None
    implied = implied_valuation_range(company, comps_result)
    lows = [v for v in (implied.get("share_price_from_revenue_min"), implied.get("share_price_from_ebitda_min")) if v is not None]
    highs = [v for v in (implied.get("share_price_from_revenue_max"), implied.get("share_price_from_ebitda_max")) if v is not None]
    if not lows or not highs:
        return None
    return min(lows), max(highs)


# --------------------------------------------------------------------------- #
# Streamlit tab
# --------------------------------------------------------------------------- #

def render_pitchbook_tab(
    company: CompanyData,
    valuation: ValuationSummary,
    dcf_result: Optional[DCFResult] = None,
    comps_result: Optional[CompsResult] = None,
) -> None:
    overview = company.overview
    df = company.annual_summary

    st.markdown(f"### {overview['name']} ({overview['ticker']})")
    st.caption(f"{overview['sector']} · {overview['industry']} · {overview['exchange']} · One-page investment summary")
    st.divider()

    highlights, risks = derive_highlights_and_risks(company, valuation, comps_result)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Business Overview")
        description = overview["description"]
        st.write(description[:450] + ("…" if len(description) > 450 else ""))

        st.markdown("##### Investment Highlights")
        for item in highlights[:4]:
            st.markdown(f"- {item}")

        st.markdown("##### Key Risks")
        for item in risks[:4]:
            st.markdown(f"- {item}")

    with right:
        st.markdown("##### Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Market Cap", format_currency(valuation.market_cap))
        m2.metric("Enterprise Value", format_currency(valuation.enterprise_value))
        m3, m4 = st.columns(2)
        m3.metric("EV / EBITDA", format_multiple_nm(valuation.ev_to_ebitda))
        m4.metric("P / E", format_multiple_nm(valuation.price_to_earnings))

        dcf_price = _dcf_avg_price(dcf_result)
        m5, m6 = st.columns(2)
        m5.metric("DCF Implied Price", format_currency(dcf_price) if dcf_price else "N/A")
        m6.metric("Current Price", format_currency(overview["current_price"]))

        comps_range = _comps_price_range(company, comps_result)
        if comps_range:
            st.metric("Comps Implied Range", f"{format_currency(comps_range[0])} – {format_currency(comps_range[1])}")

    st.divider()
    chart_cols = st.columns(2)
    with chart_cols[0]:
        prices = company.price_history("1y")
        if not prices.empty:
            st.plotly_chart(charts.price_history_chart(prices, company.ticker), use_container_width=True)
    with chart_cols[1]:
        if not df.empty:
            st.plotly_chart(charts.income_statement_chart(df), use_container_width=True)

    st.divider()
    if st.button("📑 Prepare One-Page Pitch Book PDF"):
        with st.spinner("Rendering pitch book..."):
            try:
                pdf_bytes = generate_pitchbook_pdf(company, valuation, dcf_result, comps_result)
                st.session_state[f"pitchbook_pdf::{company.ticker}"] = pdf_bytes
            except Exception as exc:  # kaleido/reportlab issues shouldn't crash the app
                st.error(f"Could not generate pitch book PDF: {exc}")

    pdf_bytes = st.session_state.get(f"pitchbook_pdf::{company.ticker}")
    if pdf_bytes:
        st.download_button(
            "⬇️ Download Pitch Book (PDF)", pdf_bytes,
            file_name=f"{company.ticker}_pitchbook.pdf", mime="application/pdf",
        )


# --------------------------------------------------------------------------- #
# One-page PDF export
# --------------------------------------------------------------------------- #

def _pitch_styles() -> dict:
    base = getSampleStyleSheet()
    body_gray = colors.HexColor("#33383f")
    return {
        "title": ParagraphStyle("PitchTitle", parent=base["Title"], fontSize=18, textColor=NAVY, alignment=TA_CENTER, spaceAfter=2),
        "subtitle": ParagraphStyle("PitchSubtitle", parent=base["Normal"], fontSize=9, textColor=body_gray, alignment=TA_CENTER, spaceAfter=8),
        "heading": ParagraphStyle("PitchHeading", parent=base["Heading3"], fontSize=10.5, textColor=NAVY, spaceBefore=4, spaceAfter=3),
        "body": ParagraphStyle("PitchBody", parent=base["BodyText"], fontSize=8, leading=10.5, textColor=body_gray, alignment=TA_JUSTIFY),
        "bullet": ParagraphStyle("PitchBullet", parent=base["BodyText"], fontSize=7.8, leading=10, textColor=body_gray, alignment=TA_JUSTIFY),
        "kpi_label": ParagraphStyle("KPILabel", parent=base["Normal"], fontSize=7, textColor=colors.white, alignment=TA_CENTER),
        "kpi_value": ParagraphStyle("KPIValue", parent=base["Normal"], fontSize=10.5, textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "footer": ParagraphStyle("PitchFooter", parent=base["Normal"], fontSize=6.5, textColor=colors.gray, alignment=TA_CENTER),
    }


def _bullets(items: list[str], style) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(text, style), leftIndent=4) for text in items],
        bulletType="bullet", start="circle", leftIndent=10, spaceAfter=2,
    )


def _kpi_strip(pairs: list[tuple[str, str]], styles: dict) -> Table:
    """A single-row 'tear sheet' strip: label on top, value below, per column."""
    labels = [Paragraph(label, styles["kpi_label"]) for label, _ in pairs]
    values = [Paragraph(value, styles["kpi_value"]) for _, value in pairs]
    col_width = 6.9 * inch / len(pairs)
    table = Table([labels, values], colWidths=[col_width] * len(pairs))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.HexColor("#2a3a5c")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def generate_pitchbook_pdf(
    company: CompanyData,
    valuation: ValuationSummary,
    dcf_result: Optional[DCFResult] = None,
    comps_result: Optional[CompsResult] = None,
) -> bytes:
    """
    Render a single-page pitch book PDF: header, KPI strip (market cap, EV,
    multiples, DCF-implied price, comps range), a two-column business
    overview / highlights / risks block, and one supporting chart.

    Returns raw PDF bytes, ready for st.download_button.
    """
    styles = _pitch_styles()
    overview = company.overview
    df = company.annual_summary
    highlights, risks = derive_highlights_and_risks(company, valuation, comps_result)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        title=f"{overview['name']} ({overview['ticker']}) Pitch Book",
    )
    story: list = []

    story.append(Paragraph(f"{overview['name']} ({overview['ticker']})", styles["title"]))
    story.append(Paragraph(
        f"{overview['sector']} &middot; {overview['industry']} &middot; One-Page Investment Summary &middot; "
        f"{datetime.now().strftime('%B %d, %Y')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=GOLD, spaceAfter=8))

    dcf_price = _dcf_avg_price(dcf_result)
    comps_range = _comps_price_range(company, comps_result)
    kpi_pairs = [
        ("MARKET CAP", format_currency(valuation.market_cap)),
        ("ENTERPRISE VALUE", format_currency(valuation.enterprise_value)),
        ("EV/EBITDA", format_multiple_nm(valuation.ev_to_ebitda)),
        ("P/E", format_multiple_nm(valuation.price_to_earnings)),
        ("DCF IMPLIED PRICE", format_currency(dcf_price) if dcf_price else "N/A"),
        ("COMPS RANGE", f"{format_currency(comps_range[0])}-{format_currency(comps_range[1])}" if comps_range else "N/A"),
    ]
    story.append(_kpi_strip(kpi_pairs, styles))
    story.append(Spacer(1, 8))

    description = overview["description"]
    left_col = [
        Paragraph("Business Overview", styles["heading"]),
        Paragraph(description[:320] + ("…" if len(description) > 320 else ""), styles["body"]),
        Spacer(1, 4),
        Paragraph("Investment Highlights", styles["heading"]),
        _bullets(highlights[:4], styles["bullet"]),
    ]
    right_col = [
        Paragraph("Key Risks", styles["heading"]),
        _bullets(risks[:4], styles["bullet"]),
        Spacer(1, 4),
        Paragraph("Current Share Price", styles["heading"]),
        Paragraph(format_currency(overview["current_price"], decimals=2), styles["body"]),
    ]
    two_col = Table([[left_col, right_col]], colWidths=[3.85 * inch, 3.05 * inch])
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (1, 0), (1, 0), 12),
        ("LINEBEFORE", (1, 0), (1, 0), 0.5, LIGHT_GRID),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 8))

    if not df.empty:
        try:
            fig = _print_friendly(charts.income_statement_chart(df))
            fig.update_layout(height=300)
            story.append(_fig_to_image(fig, draw_width=6.9 * inch))
        except Exception:
            pass  # kaleido/Chrome unavailable -- omit chart, one-pager still renders

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRID, spaceAfter=4))
    story.append(Paragraph(
        "For informational and educational purposes only -- not investment advice or a solicitation to buy or "
        f"sell any security. Prepared by Keira Hirst on {datetime.now().strftime('%B %d, %Y')}.",
        styles["footer"],
    ))

    doc.build(story)
    return buffer.getvalue()
