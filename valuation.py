"""
valuation.py
------------
Standard valuation identities and multiples: Enterprise Value, Equity Value,
EV/Revenue, EV/EBITDA, P/E, and Price/Book.

This module takes already-fetched data (from a `CompanyData` instance in
financials.py) and applies formulas -- it does not fetch anything itself.
That separation lets `comps.py` reuse the exact same multiple calculations
for peer companies without duplicating the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from financials import CompanyData
from utils import safe_divide


@dataclass
class ValuationSummary:
    """
    A single company's valuation snapshot: the building blocks (market cap,
    debt, cash) plus the derived enterprise/equity values and multiples.

    All fields are Optional because any underlying yfinance figure can be
    missing -- callers should format with utils.format_* helpers, which
    already handle None gracefully.
    """

    ticker: str
    market_cap: Optional[float]
    total_debt: Optional[float]
    total_cash: Optional[float]
    minority_interest: Optional[float]
    preferred_equity: Optional[float]
    enterprise_value: Optional[float]
    revenue: Optional[float]
    ebitda: Optional[float]
    net_income: Optional[float]
    book_value: Optional[float]  # total stockholders' equity
    diluted_eps: Optional[float]
    share_price: Optional[float]
    ev_to_revenue: Optional[float]
    ev_to_ebitda: Optional[float]
    price_to_earnings: Optional[float]
    price_to_book: Optional[float]


def calculate_enterprise_value(
    market_cap: Optional[float],
    total_debt: Optional[float],
    total_cash: Optional[float],
    minority_interest: Optional[float] = 0.0,
    preferred_equity: Optional[float] = 0.0,
) -> Optional[float]:
    """
    Enterprise Value = Market Cap + Total Debt + Minority Interest
                        + Preferred Equity - Cash & Equivalents.

    Returns None if market cap is unavailable (EV is meaningless without it);
    missing debt/cash components are treated as zero since a company can
    genuinely carry none of either.
    """
    if market_cap is None:
        return None
    return (
        market_cap
        + (total_debt or 0.0)
        + (minority_interest or 0.0)
        + (preferred_equity or 0.0)
        - (total_cash or 0.0)
    )


def calculate_equity_value(
    enterprise_value: Optional[float],
    total_debt: Optional[float],
    total_cash: Optional[float],
    minority_interest: Optional[float] = 0.0,
    preferred_equity: Optional[float] = 0.0,
) -> Optional[float]:
    """
    Equity Value = Enterprise Value - Total Debt - Minority Interest
                    - Preferred Equity + Cash & Equivalents.

    This is the inverse of calculate_enterprise_value, used in the DCF model
    where we derive EV first and need to back into implied equity value.
    """
    if enterprise_value is None:
        return None
    return (
        enterprise_value
        - (total_debt or 0.0)
        - (minority_interest or 0.0)
        - (preferred_equity or 0.0)
        + (total_cash or 0.0)
    )


def build_valuation_summary(company: CompanyData) -> ValuationSummary:
    """
    Compute a full ValuationSummary for a company using its most recent
    fiscal year of fundamentals plus current market data.

    Args:
        company: A CompanyData instance (see financials.py) for the ticker.

    Returns:
        A populated ValuationSummary. Individual fields may be None if the
        underlying data was unavailable -- this function never raises for
        missing data, only for a fundamentally invalid ticker (which would
        already have surfaced when `company.overview` was first accessed).
    """
    overview = company.overview
    summary_df = company.annual_summary

    latest = summary_df.iloc[-1] if not summary_df.empty else None
    revenue = latest["revenue"] if latest is not None else None
    ebitda = latest["ebitda"] if latest is not None else None
    net_income = latest["net_income"] if latest is not None else None
    book_value = latest["total_equity"] if latest is not None else None
    diluted_eps = latest["eps_diluted"] if latest is not None else None

    market_cap = overview["market_cap"]
    total_debt = overview["total_debt"]
    total_cash = overview["total_cash"]
    share_price = overview["current_price"]

    enterprise_value = calculate_enterprise_value(market_cap, total_debt, total_cash)

    return ValuationSummary(
        ticker=overview["ticker"],
        market_cap=market_cap,
        total_debt=total_debt,
        total_cash=total_cash,
        minority_interest=0.0,
        preferred_equity=0.0,
        enterprise_value=enterprise_value,
        revenue=revenue,
        ebitda=ebitda,
        net_income=net_income,
        book_value=book_value,
        diluted_eps=diluted_eps,
        share_price=share_price,
        ev_to_revenue=safe_divide(enterprise_value, revenue),
        ev_to_ebitda=safe_divide(enterprise_value, ebitda),
        price_to_earnings=safe_divide(share_price, diluted_eps),
        price_to_book=safe_divide(market_cap, book_value),
    )
