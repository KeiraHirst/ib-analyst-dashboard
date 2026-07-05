"""
app.py
------
Streamlit entry point. Wires financials.py, valuation.py, comps.py,
dcf.py, charts.py, and ai.py into a single interactive dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import charts
from ai import AIError, generate_investment_memo, summarize_news_article
from comps import build_comps_table, get_default_peers, implied_valuation_range
from dcf import DCFAssumptions, DCFModel
from financials import CompanyData, TickerNotFoundError
from pdf_export import generate_memo_pdf
from pitchbook import render_pitchbook_tab
from utils import format_currency, format_multiple_nm, format_number, format_percent, format_ratio
from valuation import build_valuation_summary

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="IB Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def inject_custom_css() -> None:
    """
    Full visual theme layer on top of Streamlit's dark base theme (set in
    .streamlit/config.toml). Everything here is additive CSS -- no component
    behavior changes -- aimed at a clean, modern "fintech terminal" look:
    Inter typeface, softly-elevated cards with hover lift, pill-style tabs
    with a gold active indicator, gradient primary buttons, and consistent
    rounded corners/spacing throughout.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ---------------------------------------------------------- */
        /* Global background: subtle radial glow instead of flat black */
        /* ---------------------------------------------------------- */
        [data-testid="stAppViewContainer"] > .main {
            background:
                radial-gradient(circle at 15% 0%, rgba(240, 185, 11, 0.05), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(45, 212, 191, 0.04), transparent 40%),
                #0e1117;
        }
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

        /* ---------------------------------------------------------- */
        /* Sidebar */
        /* ---------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12151d 0%, #0e1117 100%);
            border-right: 1px solid rgba(240, 185, 11, 0.12);
        }
        section[data-testid="stSidebar"] h1 {
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #f0b90b, #ffd766);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        section[data-testid="stSidebar"] .stTextInput input {
            border-radius: 8px;
            border: 1px solid rgba(240, 185, 11, 0.25);
            background-color: #161a23;
        }
        section[data-testid="stSidebar"] .stTextInput input:focus {
            border-color: #f0b90b;
            box-shadow: 0 0 0 1px rgba(240, 185, 11, 0.35);
        }
        section[data-testid="stSidebar"] .streamlit-expanderHeader {
            border-radius: 8px;
            font-weight: 600;
        }

        /* ---------------------------------------------------------- */
        /* KPI metric cards */
        /* ---------------------------------------------------------- */
        div[data-testid="stMetric"] {
            background: linear-gradient(160deg, #171b25 0%, #12151d 100%);
            border: 1px solid rgba(240, 185, 11, 0.14);
            border-radius: 12px;
            padding: 16px 18px 12px 18px;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(240, 185, 11, 0.4);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.6;
        }
        div[data-testid="stMetricValue"] {
            color: #f0b90b;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        /* ---------------------------------------------------------- */
        /* Tabs -- pill style with gold underline on the active tab */
        /* ---------------------------------------------------------- */
        div[data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        button[data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            font-size: 0.92rem;
            color: rgba(230, 230, 230, 0.6);
            padding: 8px 18px;
            transition: color 0.15s ease, background-color 0.15s ease;
        }
        button[data-baseweb="tab"]:hover {
            color: #e6e6e6;
            background-color: rgba(240, 185, 11, 0.06);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #f0b90b;
            background-color: rgba(240, 185, 11, 0.08);
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #f0b90b;
            height: 2.5px;
        }

        /* ---------------------------------------------------------- */
        /* Buttons */
        /* ---------------------------------------------------------- */
        .stButton button, .stDownloadButton button {
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid rgba(240, 185, 11, 0.3);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(240, 185, 11, 0.15);
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(90deg, #f0b90b, #d99e0a);
            border: none;
            color: #14151a;
        }

        /* ---------------------------------------------------------- */
        /* Containers, expanders, dataframes -- consistent rounding */
        /* ---------------------------------------------------------- */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            overflow: hidden;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        hr { border-color: rgba(255, 255, 255, 0.08); }

        /* ---------------------------------------------------------- */
        /* Alerts -- left accent bar instead of flat fill */
        /* ---------------------------------------------------------- */
        div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentWarning"],
        div[data-testid="stAlertContentError"], div[data-testid="stAlertContentSuccess"] {
            border-radius: 8px;
        }

        /* ---------------------------------------------------------- */
        /* Scrollbar */
        /* ---------------------------------------------------------- */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: #0e1117; }
        ::-webkit-scrollbar-thumb { background: rgba(240, 185, 11, 0.25); border-radius: 6px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(240, 185, 11, 0.45); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(company: CompanyData, valuation) -> None:
    """
    Hero banner shown above the KPI row: company name/ticker, a sector/
    industry/exchange badge line, and a live price with a color-coded
    day-over-day change -- the "masthead" of the dashboard.
    """
    overview = company.overview
    price = overview.get("current_price")
    prev_close = overview.get("previous_close")

    price_html = "N/A"
    if price is not None:
        price_html = f"${price:,.2f}"
        if prev_close:
            change = price - prev_close
            pct = (change / prev_close) * 100 if prev_close else 0
            color = "#22c55e" if change >= 0 else "#ef4444"
            arrow = "▲" if change >= 0 else "▼"
            price_html += (
                f' <span style="color:{color}; font-size:0.95rem; font-weight:600;">'
                f'{arrow} {abs(change):,.2f} ({abs(pct):.2f}%)</span>'
            )

    st.markdown(
        f"""
        <div style="
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 12px;
            padding: 18px 24px; margin-bottom: 4px; border-radius: 14px;
            background: linear-gradient(120deg, rgba(240,185,11,0.09), rgba(45,212,191,0.05));
            border: 1px solid rgba(240, 185, 11, 0.18);
        ">
            <div>
                <div style="font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; color: #f5f5f5;">
                    {overview['name']}
                    <span style="color:#f0b90b; font-weight:700;">({overview['ticker']})</span>
                </div>
                <div style="font-size: 0.85rem; opacity: 0.65; margin-top: 2px;">
                    {overview['sector']} &nbsp;·&nbsp; {overview['industry']} &nbsp;·&nbsp; {overview['exchange']}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #f5f5f5;">{price_html}</div>
                <div style="font-size: 0.75rem; opacity: 0.55;">Last price</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(ttl=600, show_spinner="Fetching company data...")
def get_company(ticker: str) -> CompanyData:
    """Cache one CompanyData instance per ticker for 10 minutes (network + lazy-computed data)."""
    return CompanyData(ticker)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> tuple[str, list[str]]:
    """Render sidebar controls. Returns (ticker, peer_tickers) -- DCF sliders are
    read directly from st.session_state by render_dcf_tab via their widget keys."""
    st.sidebar.title("📊 IB Dashboard")
    st.sidebar.caption("AI-Powered Investment Banking & Equity Research")
    ticker = st.sidebar.text_input("Company Ticker", value="AAPL").strip().upper()
    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='position: fixed; bottom: 1.5rem; font-size: 0.8rem; opacity: 0.6;'>Created by Keira Hirst</div>",
        unsafe_allow_html=True,
    )
    return ticker


def render_dcf_sidebar_controls(company: CompanyData) -> DCFAssumptions:
    """
    Render the DCF assumption sliders and return the resulting DCFAssumptions.

    Defaults are seeded once per ticker from DCFAssumptions.from_company()
    (historically-derived); after that, each slider's own Streamlit widget
    state (keyed by ticker) persists the user's edits across reruns.
    """
    ticker = company.ticker
    defaults_key = f"dcf_defaults::{ticker}"
    if defaults_key not in st.session_state:
        st.session_state[defaults_key] = DCFAssumptions.from_company(company)
    d = st.session_state[defaults_key]

    with st.sidebar.expander("💰 DCF Assumptions", expanded=False):
        projection_years = st.slider("Projection Years", 3, 10, value=d.projection_years, key=f"years::{ticker}")
        revenue_growth = st.slider("Revenue Growth (%)", -20.0, 50.0, value=round(d.revenue_growth * 100, 1), step=0.5, key=f"rg::{ticker}")
        ebitda_margin = st.slider("EBITDA Margin (%)", 0.0, 70.0, value=round(d.ebitda_margin * 100, 1), step=0.5, key=f"em::{ticker}")
        tax_rate = st.slider("Tax Rate (%)", 0.0, 40.0, value=round(d.tax_rate * 100, 1), step=0.5, key=f"tax::{ticker}")
        capex_pct = st.slider("CapEx (% of Revenue)", 0.0, 20.0, value=round(d.capex_pct_revenue * 100, 1), step=0.5, key=f"capex::{ticker}")
        nwc_pct = st.slider("Δ Working Capital (% of Δ Revenue)", 0.0, 30.0, value=round(d.nwc_pct_revenue_change * 100, 1), step=1.0, key=f"nwc::{ticker}")
        wacc = st.slider("WACC (%)", 4.0, 20.0, value=round(d.wacc * 100, 2), step=0.25, key=f"wacc::{ticker}")
        terminal_growth = st.slider("Terminal Growth (%)", 0.0, 5.0, value=round(d.terminal_growth * 100, 1), step=0.1, key=f"tg::{ticker}")
        exit_multiple = st.slider("Exit Multiple (EV/EBITDA)", 4.0, 25.0, value=d.exit_multiple, step=0.5, key=f"exit::{ticker}")

    return DCFAssumptions(
        projection_years=int(projection_years),
        revenue_growth=revenue_growth / 100,
        ebitda_margin=ebitda_margin / 100,
        da_pct_revenue=d.da_pct_revenue,  # smart default, not user-editable (see dcf.py)
        tax_rate=tax_rate / 100,
        capex_pct_revenue=capex_pct / 100,
        nwc_pct_revenue_change=nwc_pct / 100,
        wacc=wacc / 100,
        terminal_growth=terminal_growth / 100,
        exit_multiple=exit_multiple,
    )


def render_peer_sidebar_controls(company: CompanyData) -> list[str]:
    """Render the peer ticker editor, seeded with a curated/sector-based default list."""
    default_peers = get_default_peers(company.ticker, company.overview["sector"])
    with st.sidebar.expander("🏢 Comparable Companies", expanded=False):
        raw = st.text_area("Peer Tickers (comma-separated)", value=", ".join(default_peers), key=f"peers::{company.ticker}")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #

def render_kpi_cards(company: CompanyData, valuation) -> None:
    df = company.annual_summary
    latest = df.iloc[-1] if not df.empty else None

    row1 = st.columns(4)
    row1[0].metric("Market Cap", format_currency(valuation.market_cap))
    row1[1].metric("Enterprise Value", format_currency(valuation.enterprise_value))
    row1[2].metric("Revenue (FY)", format_currency(valuation.revenue),
                    format_percent(latest["revenue_growth"]) if latest is not None else None)
    row1[3].metric("EBITDA (FY)", format_currency(valuation.ebitda),
                    format_percent(latest["ebitda_margin"]) + " margin" if latest is not None else None)

    row2 = st.columns(4)
    row2[0].metric("Net Income (FY)", format_currency(valuation.net_income))
    row2[1].metric("Diluted EPS", format_number(valuation.diluted_eps))
    row2[2].metric("Free Cash Flow", format_currency(latest["free_cash_flow"]) if latest is not None else "N/A")
    row2[3].metric("EV / EBITDA", format_multiple_nm(valuation.ev_to_ebitda))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def render_overview_tab(company: CompanyData) -> None:
    overview = company.overview
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"{overview['name']} ({overview['ticker']})")
        st.caption(f"{overview['sector']} · {overview['industry']} · {overview['exchange']}")
        st.write(overview["description"])
        facts = st.columns(2)
        facts[0].markdown(
            f"**Employees:** {overview['employees']:,}\n\n**Beta:** {format_number(overview['beta'])}\n\n"
            f"**52-Week High:** {format_currency(overview['fifty_two_week_high'])}"
            if overview["employees"] else f"**Beta:** {format_number(overview['beta'])}"
        )
        facts[1].markdown(
            f"**Shares Outstanding:** {format_currency(overview['shares_outstanding'])}\n\n"
            f"**Website:** {overview['website']}\n\n"
            f"**52-Week Low:** {format_currency(overview['fifty_two_week_low'])}"
        )

    with right:
        period = st.select_slider("Price History Range", options=["1mo", "6mo", "1y", "5y", "max"], value="1y")
        prices = company.price_history(period)
        if prices.empty:
            st.warning("No price history available.")
        else:
            st.plotly_chart(charts.price_history_chart(prices, overview["ticker"]), use_container_width=True)


def render_financials_tab(company: CompanyData) -> None:
    df = company.annual_summary
    if df.empty:
        st.warning(f"No historical financial statement data is available for {company.ticker}.")
        return

    cagrs = company.cagr_summary(years=min(3, len(df) - 1))
    if cagrs:
        cols = st.columns(4)
        cols[0].metric(f"Revenue CAGR ({cagrs['years']}y)", format_percent(cagrs["revenue_cagr"]))
        cols[1].metric(f"EBITDA CAGR ({cagrs['years']}y)", format_percent(cagrs["ebitda_cagr"]))
        cols[2].metric(f"Net Income CAGR ({cagrs['years']}y)", format_percent(cagrs["net_income_cagr"]))
        cols[3].metric(f"FCF CAGR ({cagrs['years']}y)", format_percent(cagrs["fcf_cagr"]))

    row1 = st.columns(2)
    row1[0].plotly_chart(charts.income_statement_chart(df), use_container_width=True)
    row1[1].plotly_chart(charts.revenue_growth_chart(df), use_container_width=True)

    row2 = st.columns(2)
    row2[0].plotly_chart(charts.balance_sheet_chart(df), use_container_width=True)
    row2[1].plotly_chart(charts.cash_flow_chart(df), use_container_width=True)

    st.plotly_chart(charts.margin_trend_chart(df), use_container_width=True)

    with st.expander("Raw annual financial data"):
        st.dataframe(df.drop(columns=["fiscal_year"]).style.format("{:,.2f}"), use_container_width=True)


def render_valuation_tab(company: CompanyData, valuation, peer_tickers: list[str]) -> None:
    st.subheader("Valuation Multiples")
    cols = st.columns(4)
    cols[0].metric("EV / Revenue", format_multiple_nm(valuation.ev_to_revenue))
    cols[1].metric("EV / EBITDA", format_multiple_nm(valuation.ev_to_ebitda))
    cols[2].metric("P / E", format_multiple_nm(valuation.price_to_earnings))
    cols[3].metric("P / B", format_multiple_nm(valuation.price_to_book))

    st.divider()
    st.subheader("Comparable Company Analysis")
    if not peer_tickers:
        st.info("Add peer tickers in the sidebar to run a comps analysis.")
        return

    with st.spinner("Loading peer company data..."):
        comps_result = build_comps_table(company, peer_tickers)

    if comps_result.failed_tickers:
        st.warning("Could not load: " + "; ".join(f"{t} ({msg})" for t, msg in comps_result.failed_tickers.items()))

    display_table = comps_result.table.copy()
    for col in ["ev_to_revenue", "ev_to_ebitda", "price_to_earnings", "price_to_book"]:
        display_table[col] = comps_result.table[col].apply(format_multiple_nm)
    for col in ["market_cap", "enterprise_value", "revenue", "ebitda", "net_income"]:
        display_table[col] = comps_result.table[col].apply(format_currency)
    display_table = display_table.rename(columns={
        "name": "Company", "market_cap": "Market Cap", "enterprise_value": "EV",
        "revenue": "Revenue", "ebitda": "EBITDA", "net_income": "Net Income",
        "ev_to_revenue": "EV/Revenue", "ev_to_ebitda": "EV/EBITDA",
        "price_to_earnings": "P/E", "price_to_book": "P/B",
    })
    st.dataframe(display_table, use_container_width=True)

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(
        charts.comps_multiples_chart(comps_result.table, company.ticker, "ev_to_ebitda", "EV/EBITDA"),
        use_container_width=True,
    )
    chart_cols[1].plotly_chart(
        charts.comps_multiples_chart(comps_result.table, company.ticker, "ev_to_revenue", "EV/Revenue"),
        use_container_width=True,
    )

    implied = implied_valuation_range(company, comps_result)
    if implied:
        st.subheader("Implied Valuation Range (from peer multiples)")
        cols = st.columns(3)
        current_price = company.overview["current_price"]
        cols[0].metric("Implied Price — Low", format_currency(min(
            implied["share_price_from_revenue_min"] or np.inf, implied["share_price_from_ebitda_min"] or np.inf
        )))
        cols[1].metric("Implied Price — Median", format_currency(implied["share_price_from_ebitda_median"]))
        cols[2].metric(
            "Current Price", format_currency(current_price),
            format_percent((current_price / implied["share_price_from_ebitda_median"] - 1))
            if current_price and implied.get("share_price_from_ebitda_median") else None,
        )
    st.session_state["comps_result"] = comps_result


def render_dcf_tab(company: CompanyData, assumptions: DCFAssumptions) -> None:
    try:
        result = DCFModel(company, assumptions).run()
    except ValueError as exc:
        st.error(str(exc))
        return

    if result.warning:
        st.warning(result.warning)

    current_price = company.overview["current_price"]
    cols = st.columns(4)
    cols[0].metric("EV (Gordon Growth)", format_currency(result.enterprise_value_gordon))
    cols[1].metric("EV (Exit Multiple)", format_currency(result.enterprise_value_exit_multiple))
    cols[2].metric("Implied Price (Gordon)", format_currency(result.implied_share_price_gordon))
    cols[3].metric("Implied Price (Exit Multiple)", format_currency(result.implied_share_price_exit_multiple))

    st.caption(f"Current market price: {format_currency(current_price)}")

    with st.expander("Year-by-year unlevered FCF projection"):
        st.dataframe(result.projections.style.format("{:,.1f}"), use_container_width=True)

    st.divider()
    st.subheader("Sensitivity Analysis")
    model = DCFModel(company, assumptions)
    wacc_range = [round(assumptions.wacc + step, 4) for step in np.arange(-0.02, 0.021, 0.01)]
    growth_range = [round(assumptions.terminal_growth + step, 4) for step in np.arange(-0.01, 0.011, 0.005)]
    multiple_range = [round(assumptions.exit_multiple + step, 1) for step in np.arange(-4, 4.1, 2)]

    sens_cols = st.columns(2)
    with sens_cols[0]:
        grid1 = model.sensitivity_wacc_growth(wacc_range, growth_range)
        st.plotly_chart(charts.sensitivity_heatmap(grid1, "WACC vs. Terminal Growth"), use_container_width=True)
    with sens_cols[1]:
        grid2 = model.sensitivity_wacc_exit_multiple(wacc_range, multiple_range)
        st.plotly_chart(charts.sensitivity_heatmap(grid2, "WACC vs. Exit Multiple"), use_container_width=True)

    st.session_state["dcf_result"] = result


def render_ai_memo_tab(company: CompanyData, valuation) -> None:
    st.subheader("AI-Generated Investment Memo")
    st.caption("Generated by Claude from the exact figures currently shown in this dashboard.")

    if st.button("🪄 Generate Investment Memo", type="primary"):
        with st.spinner("Drafting memo..."):
            try:
                memo = generate_investment_memo(
                    company, valuation,
                    dcf_result=st.session_state.get("dcf_result"),
                    comps_result=st.session_state.get("comps_result"),
                )
                st.session_state[f"memo::{company.ticker}"] = memo
            except AIError as exc:
                st.error(str(exc))

    memo = st.session_state.get(f"memo::{company.ticker}")
    if memo:
        st.markdown(memo)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button("Download memo (.md)", memo, file_name=f"{company.ticker}_investment_memo.md")
        with col2:
            if st.button("📄 Prepare PDF Memo"):
                with st.spinner("Rendering PDF..."):
                    try:
                        pdf_bytes = generate_memo_pdf(
                            company, valuation, memo,
                            dcf_result=st.session_state.get("dcf_result"),
                            comps_result=st.session_state.get("comps_result"),
                        )
                        st.session_state[f"memo_pdf::{company.ticker}"] = pdf_bytes
                    except Exception as exc:  # kaleido/reportlab rendering issues shouldn't crash the app
                        st.error(f"Could not generate PDF: {exc}")

        pdf_bytes = st.session_state.get(f"memo_pdf::{company.ticker}")
        if pdf_bytes:
            st.download_button(
                "⬇️ Download PDF Memo", pdf_bytes,
                file_name=f"{company.ticker}_investment_memo.pdf", mime="application/pdf",
            )
    else:
        st.info("Click the button above to generate a memo. Requires ANTHROPIC_API_KEY to be set in .env.")


def render_news_tab(company: CompanyData) -> None:
    st.subheader(f"Recent News — {company.ticker}")
    news = company.recent_news(limit=8)
    if not news:
        st.info("No recent news found for this ticker.")
        return

    for i, item in enumerate(news):
        with st.container(border=True):
            st.markdown(f"**[{item['title']}]({item['link']})**" if item["link"] else f"**{item['title']}**")
            meta = item["publisher"]
            if item["published"]:
                meta += f" · {item['published'].strftime('%Y-%m-%d')}"
            st.caption(meta)

            summary_key = f"news_summary::{company.ticker}::{i}"
            if st.session_state.get(summary_key):
                st.write(st.session_state[summary_key])
            elif st.button("Summarize with AI", key=f"btn::{summary_key}"):
                with st.spinner("Summarizing..."):
                    try:
                        st.session_state[summary_key] = summarize_news_article(item["title"], item.get("summary", ""))
                        st.rerun()
                    except AIError as exc:
                        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    inject_custom_css()
    ticker = render_sidebar()

    if not ticker:
        st.info("Enter a ticker in the sidebar to begin.")
        return

    try:
        company = get_company(ticker)
        _ = company.overview  # forces validation now, not on first tab access
    except TickerNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unexpected error loading '{ticker}': {exc}")
        return

    dcf_assumptions = render_dcf_sidebar_controls(company)
    peer_tickers = render_peer_sidebar_controls(company)

    valuation = build_valuation_summary(company)
    render_header(company, valuation)
    st.write("")
    render_kpi_cards(company, valuation)
    st.divider()

    tabs = st.tabs([
        "Overview", "Financials", "Valuation & Comps", "DCF Model",
        "Pitch Book", "AI Investment Memo", "News",
    ])
    with tabs[0]:
        render_overview_tab(company)
    with tabs[1]:
        render_financials_tab(company)
    with tabs[2]:
        render_valuation_tab(company, valuation, peer_tickers)
    with tabs[3]:
        render_dcf_tab(company, dcf_assumptions)
    with tabs[4]:
        render_pitchbook_tab(
            company, valuation,
            dcf_result=st.session_state.get("dcf_result"),
            comps_result=st.session_state.get("comps_result"),
        )
    with tabs[5]:
        render_ai_memo_tab(company, valuation)
    with tabs[6]:
        render_news_tab(company)


if __name__ == "__main__":
    main()
