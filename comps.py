"""
comps.py
--------
Comparable company ("comps") analysis: builds a peer valuation table and
derives an implied valuation range for the subject company from peer
trading multiples.

There is no free, reliable API for "give me this company's true peer set,"
so this module ships a small curated map of sensible defaults for
well-known names and sectors. The UI is expected to let the user edit the
peer list -- the defaults are a starting point, not a source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financials import CompanyData, TickerNotFoundError
from utils import safe_divide
from valuation import build_valuation_summary

# Curated defaults for tickers explicitly called out as examples in the
# product spec, plus a few obvious mega-cap peer sets.
DEFAULT_PEER_GROUPS: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL", "DELL", "HPQ"],
    "MSFT": ["AAPL", "GOOGL", "ORCL", "IBM"],
    "GOOGL": ["META", "MSFT", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "AMZN": ["WMT", "TGT", "GOOGL"],
    "JPM": ["BAC", "WFC", "C", "GS"],
    "BAC": ["JPM", "WFC", "C"],
    "GS": ["MS", "JPM", "BAC"],
    "NVDA": ["AMD", "INTC", "AVGO"],
    "AMD": ["NVDA", "INTC", "QCOM"],
}

# Fallback peer sets keyed by yfinance's broad `sector` field, used when the
# subject ticker isn't in the curated map above.
SECTOR_PEER_GROUPS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "ORCL"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
    "Energy": ["XOM", "CVX", "COP"],
}


def get_default_peers(ticker: str, sector: str = "") -> list[str]:
    """
    Suggest a default peer list for `ticker`, preferring the curated map
    and falling back to a sector-level list. Always excludes the subject
    ticker itself. Returns an empty list if no default is known -- the
    caller should prompt the user to enter peers manually in that case.
    """
    ticker = ticker.upper()
    peers = DEFAULT_PEER_GROUPS.get(ticker) or SECTOR_PEER_GROUPS.get(sector, [])
    return [p for p in peers if p.upper() != ticker]


@dataclass
class CompsResult:
    """
    Output of a comparable company analysis run.

    Attributes:
        table: One row per company (subject + peers) indexed by ticker,
            with columns: name, market_cap, enterprise_value, revenue,
            ebitda, net_income, ev_to_revenue, ev_to_ebitda,
            price_to_earnings, price_to_book.
        peer_stats: min/median/mean/max of each multiple column, computed
            over peers only (the subject row is excluded so it can be
            benchmarked against this).
        failed_tickers: peer tickers that could not be fetched, mapped to
            an error message, so the UI can surface a warning without
            failing the whole analysis.
    """

    table: pd.DataFrame
    peer_stats: pd.DataFrame
    failed_tickers: dict[str, str] = field(default_factory=dict)


_TABLE_COLUMNS = [
    "name", "market_cap", "enterprise_value", "revenue", "ebitda", "net_income",
    "ev_to_revenue", "ev_to_ebitda", "price_to_earnings", "price_to_book",
]
_MULTIPLE_COLUMNS = ["ev_to_revenue", "ev_to_ebitda", "price_to_earnings", "price_to_book"]


def build_comps_table(subject_company: CompanyData, peer_tickers: list[str]) -> CompsResult:
    """
    Build a comparable company table for `subject_company` against
    `peer_tickers`. Peer fetch failures (bad ticker, no data) are collected
    and skipped rather than aborting the whole analysis.

    Args:
        subject_company: An already-constructed CompanyData for the
            company being analyzed (avoids re-fetching it).
        peer_tickers: List of peer ticker symbols to compare against.

    Returns:
        A CompsResult with the combined table, peer-only summary stats,
        and any tickers that failed to load.
    """
    rows: dict[str, dict] = {}
    failed: dict[str, str] = {}

    subject_valuation = build_valuation_summary(subject_company)
    rows[subject_company.ticker] = {
        "name": subject_company.overview["name"],
        "market_cap": subject_valuation.market_cap,
        "enterprise_value": subject_valuation.enterprise_value,
        "revenue": subject_valuation.revenue,
        "ebitda": subject_valuation.ebitda,
        "net_income": subject_valuation.net_income,
        "ev_to_revenue": subject_valuation.ev_to_revenue,
        "ev_to_ebitda": subject_valuation.ev_to_ebitda,
        "price_to_earnings": subject_valuation.price_to_earnings,
        "price_to_book": subject_valuation.price_to_book,
    }

    for peer_ticker in peer_tickers:
        peer_ticker = peer_ticker.strip().upper()
        if not peer_ticker or peer_ticker == subject_company.ticker:
            continue
        try:
            peer_company = CompanyData(peer_ticker)
            peer_valuation = build_valuation_summary(peer_company)
            rows[peer_ticker] = {
                "name": peer_company.overview["name"],
                "market_cap": peer_valuation.market_cap,
                "enterprise_value": peer_valuation.enterprise_value,
                "revenue": peer_valuation.revenue,
                "ebitda": peer_valuation.ebitda,
                "net_income": peer_valuation.net_income,
                "ev_to_revenue": peer_valuation.ev_to_revenue,
                "ev_to_ebitda": peer_valuation.ev_to_ebitda,
                "price_to_earnings": peer_valuation.price_to_earnings,
                "price_to_book": peer_valuation.price_to_book,
            }
        except TickerNotFoundError as exc:
            failed[peer_ticker] = str(exc)
        except Exception as exc:  # network hiccups, malformed peer data, etc.
            failed[peer_ticker] = f"Unexpected error loading '{peer_ticker}': {exc}"

    table = pd.DataFrame.from_dict(rows, orient="index", columns=_TABLE_COLUMNS)

    peer_only = table.drop(index=subject_company.ticker, errors="ignore")
    if peer_only.empty:
        peer_stats = pd.DataFrame(columns=_MULTIPLE_COLUMNS)
    else:
        peer_stats = peer_only[_MULTIPLE_COLUMNS].agg(["min", "median", "mean", "max"])

    return CompsResult(table=table, peer_stats=peer_stats, failed_tickers=failed)


def implied_valuation_range(subject_company: CompanyData, comps_result: CompsResult) -> dict:
    """
    Derive an implied Enterprise Value and per-share value range for the
    subject company by applying peer median/min/max EV/Revenue and
    EV/EBITDA multiples to the subject's own revenue and EBITDA.

    Returns an empty dict if there are no peer stats to draw from (e.g. all
    peer fetches failed) or the subject is missing revenue/EBITDA/share data.
    """
    if comps_result.peer_stats.empty:
        return {}

    subject_row = comps_result.table.loc[subject_company.ticker]
    revenue, ebitda = subject_row["revenue"], subject_row["ebitda"]
    overview = subject_company.overview
    shares_outstanding = overview["shares_outstanding"]
    net_debt = (overview["total_debt"] or 0.0) - (overview["total_cash"] or 0.0)

    implied = {}
    for stat in ("min", "median", "max"):
        ev_from_revenue = revenue * comps_result.peer_stats.loc[stat, "ev_to_revenue"] if revenue is not None else None
        ev_from_ebitda = ebitda * comps_result.peer_stats.loc[stat, "ev_to_ebitda"] if ebitda is not None else None
        implied[f"ev_from_revenue_{stat}"] = ev_from_revenue
        implied[f"ev_from_ebitda_{stat}"] = ev_from_ebitda
        implied[f"share_price_from_revenue_{stat}"] = safe_divide(
            (ev_from_revenue - net_debt) if ev_from_revenue is not None else None, shares_outstanding
        )
        implied[f"share_price_from_ebitda_{stat}"] = safe_divide(
            (ev_from_ebitda - net_debt) if ev_from_ebitda is not None else None, shares_outstanding
        )
    return implied
