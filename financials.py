"""
financials.py
-------------
Data-fetching layer. Wraps yfinance and exposes clean, analyst-ready data:
a company overview dict, raw financial statement DataFrames, and a single
tidy `annual_summary` DataFrame (revenue, EBITDA, net income, EPS, FCF,
margins, YoY growth) that every downstream module builds on.

Nothing here imports Streamlit -- this module is pure data plumbing so it
can be tested and reused independently of the UI layer.
"""

from __future__ import annotations

from datetime import datetime
from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from utils import calculate_cagr, get_nested


class TickerNotFoundError(Exception):
    """Raised when yfinance returns no usable data for a given ticker symbol."""


def _first_available_row(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:
    """
    Return the first row in `df` whose index label matches one of `candidates`
    (case-insensitive). yfinance renames statement line items across versions
    and tickers, so callers pass several known aliases for the same concept.

    Returns None if the DataFrame is empty or none of the candidates match,
    rather than raising -- missing line items are common and expected.
    """
    if df is None or df.empty:
        return None
    lower_index = {str(label).lower(): label for label in df.index}
    for candidate in candidates:
        match = lower_index.get(candidate.lower())
        if match is not None:
            return df.loc[match]
    return None


def _parse_published(value) -> Optional[datetime]:
    """Normalize a news item's publish timestamp (epoch int or ISO string) to a datetime."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(value)
        return pd.to_datetime(value).to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None


def _normalize_news_item(raw: dict) -> dict:
    """
    Normalize a single yfinance news item into a stable shape.

    yfinance has shipped two different news payload schemas: an older flat
    one (title/publisher/link at the top level) and a newer one that nests
    those fields under a "content" key. This handles both.
    """
    content = raw.get("content", raw) or {}
    title = content.get("title") or raw.get("title") or "Untitled"
    publisher = get_nested(content, ["provider", "displayName"]) or raw.get("publisher") or "Unknown source"
    link = get_nested(content, ["canonicalUrl", "url"]) or raw.get("link") or ""
    published = _parse_published(content.get("pubDate") or raw.get("providerPublishTime"))
    summary = content.get("summary") or raw.get("summary") or ""
    return {"title": title, "publisher": publisher, "link": link, "published": published, "summary": summary}


class CompanyData:
    """
    Fetches and normalizes fundamental data for a single public company.

    Usage:
        company = CompanyData("AAPL")
        overview = company.overview
        summary = company.annual_summary          # tidy DataFrame
        cagrs = company.cagr_summary(years=3)
        prices = company.price_history(period="1y")
        news = company.recent_news(limit=5)

    Data is fetched lazily and cached per-instance (via cached_property for
    parameter-free calls, and a manual dict cache for parameterized ones like
    `price_history`), so repeated access doesn't re-hit the network.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.strip().upper()
        self._yf_ticker = yf.Ticker(self.ticker)
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._news_cache: Optional[list[dict]] = None

    # ------------------------------------------------------------------ #
    # Raw data (cached, lazy)
    # ------------------------------------------------------------------ #

    @cached_property
    def info(self) -> dict:
        """Raw yfinance `.info` dict. Raises TickerNotFoundError if the ticker is invalid."""
        try:
            info = self._yf_ticker.info
        except Exception as exc:  # network/parsing failures from yfinance's internals
            raise TickerNotFoundError(f"Could not fetch data for '{self.ticker}': {exc}") from exc

        has_identity = any(info.get(key) for key in ("longName", "shortName", "symbol"))
        if not info or not has_identity:
            raise TickerNotFoundError(
                f"'{self.ticker}' does not appear to be a valid, listed ticker symbol."
            )
        return info

    @cached_property
    def income_statement(self) -> pd.DataFrame:
        """Raw annual income statement. Rows = line items, columns = fiscal period-end dates."""
        return self._yf_ticker.income_stmt if self._yf_ticker.income_stmt is not None else pd.DataFrame()

    @cached_property
    def balance_sheet(self) -> pd.DataFrame:
        """Raw annual balance sheet. Rows = line items, columns = fiscal period-end dates."""
        return self._yf_ticker.balance_sheet if self._yf_ticker.balance_sheet is not None else pd.DataFrame()

    @cached_property
    def cash_flow(self) -> pd.DataFrame:
        """Raw annual cash flow statement. Rows = line items, columns = fiscal period-end dates."""
        return self._yf_ticker.cashflow if self._yf_ticker.cashflow is not None else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Company overview
    # ------------------------------------------------------------------ #

    @property
    def overview(self) -> dict:
        """Clean, display-ready company overview fields."""
        info = self.info
        return {
            "ticker": self.ticker,
            "name": info.get("longName") or info.get("shortName") or self.ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", "No description available."),
            "website": info.get("website", ""),
            "exchange": info.get("exchange", "N/A"),
            "currency": info.get("currency", "USD"),
            "employees": info.get("fullTimeEmployees"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "trailing_pe": info.get("trailingPE"),
            "price_to_book": info.get("priceToBook"),
        }

    # ------------------------------------------------------------------ #
    # Tidy annual summary: the canonical table other modules consume
    # ------------------------------------------------------------------ #

    @cached_property
    def annual_summary(self) -> pd.DataFrame:
        """
        Build a tidy, chronologically-ascending DataFrame with one row per
        fiscal year and pre-computed metrics: revenue, EBITDA, net income,
        EPS, FCF, margins, and YoY growth.

        Returns an empty DataFrame if yfinance has no income statement data
        for this ticker (callers should check `.empty` before use).
        """
        income = self.income_statement
        if income.empty:
            return pd.DataFrame()

        balance = self.balance_sheet
        cash_flow = self.cash_flow

        rows = {
            "revenue": _first_available_row(income, ["Total Revenue", "TotalRevenue"]),
            "gross_profit": _first_available_row(income, ["Gross Profit"]),
            "ebit": _first_available_row(income, ["EBIT", "Operating Income"]),
            "net_income": _first_available_row(
                income, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]
            ),
            "eps_diluted": _first_available_row(income, ["Diluted EPS"]),
            "d_and_a": _first_available_row(
                cash_flow, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"]
            ),
            "operating_cash_flow": _first_available_row(
                cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]
            ),
            "capex": _first_available_row(cash_flow, ["Capital Expenditure", "Capital Expenditures"]),
            "total_assets": _first_available_row(balance, ["Total Assets"]),
            "total_liabilities": _first_available_row(
                balance, ["Total Liabilities Net Minority Interest", "Total Liab"]
            ),
            "total_equity": _first_available_row(
                balance, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Total Stockholder Equity"]
            ),
            "cash_and_equivalents": _first_available_row(
                balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"]
            ),
            "total_debt": _first_available_row(balance, ["Total Debt"]),
        }

        # pd.DataFrame aligns each Series on the union of their period-end
        # dates automatically; restrict to periods the income statement
        # actually reports, and drop any period yfinance included as a
        # column header but never populated a revenue figure for (this
        # happens occasionally and would otherwise render as a phantom
        # "fiscal year" with no data in every chart).
        df = pd.DataFrame(rows).reindex(income.columns).sort_index()
        df = df.dropna(subset=["revenue"])

        # --- Derived metrics ---
        df["ebitda"] = df["ebit"] + df["d_and_a"]
        df["free_cash_flow"] = df["operating_cash_flow"] + df["capex"]  # capex is already negative

        with np.errstate(divide="ignore", invalid="ignore"):
            df["gross_margin"] = df["gross_profit"] / df["revenue"]
            df["ebit_margin"] = df["ebit"] / df["revenue"]
            df["ebitda_margin"] = df["ebitda"] / df["revenue"]
            df["net_margin"] = df["net_income"] / df["revenue"]
            df["fcf_margin"] = df["free_cash_flow"] / df["revenue"]

        df = df.replace([np.inf, -np.inf], np.nan)

        df["revenue_growth"] = df["revenue"].pct_change()
        df["ebitda_growth"] = df["ebitda"].pct_change()
        df["net_income_growth"] = df["net_income"].pct_change()

        df.index.name = "fiscal_period_end"
        df["fiscal_year"] = df.index.year
        return df

    def cagr_summary(self, years: Optional[int] = None) -> dict:
        """
        Compute CAGR for revenue, EBITDA, net income, and FCF over the most
        recent `years` fiscal years (or the full available history if None).

        Returns an empty dict if there isn't enough history (fewer than 2
        fiscal years of data) to compute a growth rate.
        """
        df = self.annual_summary
        if df.empty or len(df) < 2:
            return {}

        span = len(df) - 1 if years is None else min(years, len(df) - 1)
        if span <= 0:
            return {}

        begin, end = df.iloc[-(span + 1)], df.iloc[-1]
        return {
            "revenue_cagr": calculate_cagr(begin["revenue"], end["revenue"], span),
            "ebitda_cagr": calculate_cagr(begin["ebitda"], end["ebitda"], span),
            "net_income_cagr": calculate_cagr(begin["net_income"], end["net_income"], span),
            "fcf_cagr": calculate_cagr(begin["free_cash_flow"], end["free_cash_flow"], span),
            "years": span,
        }

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    def price_history(self, period: str = "1y") -> pd.DataFrame:
        """
        Daily OHLCV price history for the given yfinance period string
        (e.g. "1mo", "6mo", "1y", "5y", "max"). Cached per period.
        """
        if period not in self._price_cache:
            history = self._yf_ticker.history(period=period, auto_adjust=True)
            self._price_cache[period] = history
        return self._price_cache[period]

    def recent_news(self, limit: int = 8) -> list[dict]:
        """Recent news headlines for this ticker, normalized to a stable schema."""
        if self._news_cache is None:
            try:
                raw_items = self._yf_ticker.news or []
            except Exception:
                raw_items = []
            self._news_cache = [_normalize_news_item(item) for item in raw_items]
        return self._news_cache[:limit]
