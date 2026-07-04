"""
utils.py
--------
Framework-agnostic helper functions shared across the dashboard.

This module intentionally has zero dependencies on Streamlit, yfinance, or
Plotly. It only deals in plain Python/numeric types so it can be unit tested
in isolation and reused regardless of what fetches or renders the data.

Two categories of helpers live here:
  1. Formatters   -- turn raw numbers into banker-readable strings.
  2. Math helpers -- safe division, CAGR, clamping, nested dict access.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #

def format_large_number(value: Optional[float], decimals: int = 1, prefix: str = "") -> str:
    """
    Format a large number using banker shorthand: T / B / M / K.

    Example:
        format_large_number(328_000_000_000, prefix="$") -> "$328.0B"
        format_large_number(-2_400_000)                  -> "-2.4M"
        format_large_number(None)                         -> "N/A"

    Args:
        value: The raw numeric value (can be None or NaN -- both return "N/A").
        decimals: Decimal places to show on the scaled value.
        prefix: Optional string prepended after the sign, e.g. "$".

    Returns:
        A formatted string, or "N/A" if value is missing/invalid.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"

    sign = "-" if value < 0 else ""
    magnitude = abs(value)

    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{sign}{prefix}{magnitude / threshold:.{decimals}f}{suffix}"

    return f"{sign}{prefix}{magnitude:.{decimals}f}"


def format_currency(value: Optional[float], decimals: int = 1) -> str:
    """Shorthand for format_large_number with a '$' prefix."""
    return format_large_number(value, decimals=decimals, prefix="$")


def format_percent(value: Optional[float], decimals: int = 1, already_pct: bool = False) -> str:
    """
    Format a ratio as a percentage string.

    Args:
        value: The raw value, e.g. 0.284 for 28.4%.
        decimals: Decimal places to show.
        already_pct: Set True if `value` is already on a 0-100 scale
            (e.g. 28.4 meaning 28.4%) rather than a 0-1 fraction.

    Returns:
        Formatted string like "28.4%", or "N/A" if value is missing/invalid.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    pct = value if already_pct else value * 100
    return f"{pct:.{decimals}f}%"


def format_ratio(value: Optional[float], decimals: int = 1, suffix: str = "x") -> str:
    """
    Format a valuation multiple, e.g. EV/EBITDA -> "14.2x".

    Returns "N/A" if value is missing, negative-infinite, or NaN.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def format_multiple_nm(value: Optional[float], decimals: int = 1, max_reasonable: float = 200.0, suffix: str = "x") -> str:
    """
    Format a valuation multiple using the standard banking "NM" (Not
    Meaningful) convention for values that would otherwise mislead --
    negative multiples (common when a company has negative book equity
    from buybacks, e.g. P/B) or implausibly extreme ones (e.g. P/E on a
    near-zero-earnings company).

    Args:
        value: The raw multiple.
        decimals: Decimal places for a normal, in-range value.
        max_reasonable: Upper bound above which the value is flagged "NM".
        suffix: Unit suffix for a normal value, e.g. "x".

    Returns:
        A formatted string, "NM" if out of a sane range, or "N/A" if missing.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    if value < 0 or value > max_reasonable:
        return "NM"
    return f"{value:.{decimals}f}{suffix}"


def format_number(value: Optional[float], decimals: int = 2) -> str:
    """Format a plain number (e.g. EPS) with thousands separators, no scaling."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #

def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """
    Divide two numbers, returning None instead of raising when the operation
    is undefined (None operands, zero/near-zero denominator, or NaN inputs).

    This is used constantly when computing margins and multiples from
    yfinance data, where either side of a ratio can legitimately be missing.
    """
    if numerator is None or denominator is None:
        return None
    if isinstance(numerator, float) and math.isnan(numerator):
        return None
    if isinstance(denominator, float) and math.isnan(denominator):
        return None
    if math.isclose(float(denominator), 0.0, abs_tol=1e-9):
        return None
    return numerator / denominator


def calculate_cagr(begin_value: Optional[float], end_value: Optional[float], periods: float) -> Optional[float]:
    """
    Compute Compound Annual Growth Rate between two values over N periods.

    Args:
        begin_value: Starting value (must be positive; CAGR is undefined
            for a negative or zero starting base).
        end_value: Ending value.
        periods: Number of periods (years) between begin and end.

    Returns:
        CAGR as a decimal fraction (e.g. 0.15 for 15%), or None if the
        inputs make the calculation undefined (non-positive base, zero
        periods, or a negative end value that would require taking a
        root of a negative number).
    """
    if begin_value is None or end_value is None or periods is None:
        return None
    if begin_value <= 0 or periods <= 0:
        return None
    if end_value < 0:
        return None
    return (end_value / begin_value) ** (1 / periods) - 1


def clamp(value: float, lo: float, hi: float) -> float:
    """Constrain `value` to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def get_nested(data: dict, keys: Sequence[Any], default: Any = None) -> Any:
    """
    Safely walk a sequence of keys/indices through nested dicts or objects
    that may be missing intermediate levels (common with yfinance's `.info`
    dict, which silently omits fields depending on ticker/exchange).

    Example:
        get_nested(info, ["financialData", "targetMeanPrice"], default=None)

    Args:
        data: The root dict to traverse.
        keys: Ordered sequence of keys to walk.
        default: Value to return if any key is missing or the path errors out.

    Returns:
        The resolved value, or `default` if the path could not be resolved.
    """
    current = data
    for key in keys:
        if current is None:
            return default
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current if current is not None else default
