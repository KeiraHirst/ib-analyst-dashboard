"""
charts.py
---------
Plotly figure builders, all styled for a single dark "Bloomberg terminal
meets fintech" theme. Every function is pure: DataFrame (from financials.py,
comps.py, or dcf.py) in, a ready-to-render `go.Figure` out. No Streamlit
imports here -- `st.plotly_chart(fig, use_container_width=True)` is the
caller's job.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from utils import format_large_number

# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #

BG_COLOR = "#0e1117"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_COLOR = "#e6e6e6"
ACCENT_GOLD = "#f0b90b"    # Bloomberg-style amber, used as the primary accent
ACCENT_TEAL = "#2dd4bf"
ACCENT_BLUE = "#3b82f6"
ACCENT_PURPLE = "#a78bfa"
POSITIVE = "#22c55e"
NEGATIVE = "#ef4444"

_PALETTE = [ACCENT_GOLD, ACCENT_TEAL, ACCENT_BLUE, ACCENT_PURPLE]


def _apply_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply the shared dark theme to a figure. Called by every builder below."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, -apple-system, sans-serif"),
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#1c1f26", font_color=TEXT_COLOR),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def _year_labels(df: pd.DataFrame) -> list[str]:
    """Use the pre-computed `fiscal_year` column if present, else the index."""
    if "fiscal_year" in df.columns:
        return [str(y) for y in df["fiscal_year"]]
    return [str(i) for i in df.index]


# --------------------------------------------------------------------------- #
# Financial statement charts
# --------------------------------------------------------------------------- #

def income_statement_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of Revenue, EBITDA, and Net Income by fiscal year."""
    years = _year_labels(df)
    fig = go.Figure()
    for col, label, color in [
        ("revenue", "Revenue", ACCENT_GOLD),
        ("ebitda", "EBITDA", ACCENT_TEAL),
        ("net_income", "Net Income", ACCENT_BLUE),
    ]:
        fig.add_bar(
            x=years, y=df[col], name=label, marker_color=color,
            text=[format_large_number(v, prefix="$") for v in df[col]],
            textposition="outside",
        )
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "Income Statement Overview")


def balance_sheet_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing Total Assets, Total Liabilities, and Total Equity by year."""
    years = _year_labels(df)
    fig = go.Figure()
    for col, label, color in [
        ("total_assets", "Total Assets", ACCENT_TEAL),
        ("total_liabilities", "Total Liabilities", NEGATIVE),
        ("total_equity", "Total Equity", ACCENT_GOLD),
    ]:
        fig.add_bar(
            x=years, y=df[col], name=label, marker_color=color,
            text=[format_large_number(v, prefix="$") for v in df[col]],
            textposition="outside",
        )
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "Balance Sheet Overview")


def cash_flow_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of Operating Cash Flow, CapEx, and resulting Free Cash Flow by year."""
    years = _year_labels(df)
    fig = go.Figure()
    fig.add_bar(x=years, y=df["operating_cash_flow"], name="Operating Cash Flow", marker_color=ACCENT_TEAL)
    fig.add_bar(x=years, y=df["capex"], name="CapEx", marker_color=NEGATIVE)
    fig.add_trace(go.Scatter(
        x=years, y=df["free_cash_flow"], name="Free Cash Flow", mode="lines+markers",
        line=dict(color=ACCENT_GOLD, width=3), marker=dict(size=8),
    ))
    fig.update_layout(barmode="relative")
    return _apply_theme(fig, "Cash Flow Overview")


def revenue_growth_chart(df: pd.DataFrame) -> go.Figure:
    """Revenue bars with a YoY growth-rate line on a secondary axis."""
    years = _year_labels(df)
    fig = go.Figure()
    fig.add_bar(x=years, y=df["revenue"], name="Revenue", marker_color=ACCENT_GOLD, opacity=0.85)
    fig.add_trace(go.Scatter(
        x=years, y=df["revenue_growth"] * 100, name="YoY Growth %", mode="lines+markers",
        line=dict(color=ACCENT_TEAL, width=3), marker=dict(size=8), yaxis="y2",
    ))
    fig.update_layout(
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="YoY Growth %", overlaying="y", side="right", showgrid=False),
    )
    return _apply_theme(fig, "Revenue & Growth Trend")


def margin_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of gross, EBIT, EBITDA, and net margins over time."""
    years = _year_labels(df)
    fig = go.Figure()
    for col, label, color in [
        ("gross_margin", "Gross Margin", ACCENT_PURPLE),
        ("ebit_margin", "EBIT Margin", ACCENT_BLUE),
        ("ebitda_margin", "EBITDA Margin", ACCENT_TEAL),
        ("net_margin", "Net Margin", ACCENT_GOLD),
    ]:
        fig.add_trace(go.Scatter(
            x=years, y=df[col] * 100, name=label, mode="lines+markers",
            line=dict(color=color, width=2.5), marker=dict(size=7),
        ))
    fig.update_yaxes(title="Margin %")
    return _apply_theme(fig, "Margin Trends")


# --------------------------------------------------------------------------- #
# Market data charts
# --------------------------------------------------------------------------- #

def price_history_chart(price_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Area/line chart of daily closing price. `price_df` comes from CompanyData.price_history()."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df["Close"], mode="lines", name=ticker,
        line=dict(color=ACCENT_GOLD, width=2), fill="tozeroy",
        fillcolor="rgba(240, 185, 11, 0.08)",
    ))
    fig.update_yaxes(title="Price ($)")
    return _apply_theme(fig, f"{ticker} Price Performance", height=320)


def multi_price_comparison_chart(price_series: dict[str, pd.Series]) -> go.Figure:
    """
    Normalized (rebased to 100 at the start of the window) price comparison
    across multiple tickers -- used by the two-company comparison feature.

    Args:
        price_series: dict of {ticker: Close price Series} with a datetime index.
    """
    fig = go.Figure()
    for (ticker, series), color in zip(price_series.items(), _PALETTE):
        rebased = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased, mode="lines", name=ticker, line=dict(color=color, width=2.5)))
    fig.update_yaxes(title="Rebased Price (start = 100)")
    return _apply_theme(fig, "Relative Price Performance", height=380)


# --------------------------------------------------------------------------- #
# Valuation / DCF charts
# --------------------------------------------------------------------------- #

def sensitivity_heatmap(grid: pd.DataFrame, title: str, value_prefix: str = "$") -> go.Figure:
    """
    Heatmap for a DCF sensitivity grid (e.g. WACC x Terminal Growth), with
    each cell annotated with its formatted value.
    """
    z = grid.values.astype(float)
    text = [[f"{value_prefix}{v:.0f}" if pd.notna(v) else "N/A" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{c:.1%}" if grid.columns.name == "Terminal Growth" else str(c) for c in grid.columns],
        y=[f"{r:.1%}" for r in grid.index],
        colorscale=[[0, "#1e293b"], [0.5, "#f0b90b"], [1, "#22c55e"]],
        text=text, texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="Share Price"),
    ))
    fig.update_layout(
        xaxis=dict(title=grid.columns.name, side="bottom"),
        yaxis=dict(title=grid.index.name),
    )
    return _apply_theme(fig, title, height=380)


def comps_multiples_chart(comps_table: pd.DataFrame, subject_ticker: str, multiple_col: str, label: str) -> go.Figure:
    """
    Horizontal bar chart comparing one valuation multiple across the
    subject company and its peers, with the subject bar highlighted.
    """
    data = comps_table[multiple_col].dropna().sort_values()
    colors = [ACCENT_GOLD if ticker == subject_ticker else ACCENT_TEAL for ticker in data.index]

    fig = go.Figure(go.Bar(
        x=data.values, y=data.index, orientation="h", marker_color=colors,
        text=[f"{v:.1f}x" for v in data.values], textposition="outside",
    ))
    return _apply_theme(fig, f"{label} — Peer Comparison", height=max(280, 60 + 40 * len(data)))
