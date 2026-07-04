"""
dcf.py
------
Discounted Cash Flow model: projects unlevered free cash flow off editable
assumptions, discounts it at WACC, and derives Enterprise Value, Equity
Value, and implied share price under two terminal value methods (Gordon
Growth and Exit Multiple) so the two can be cross-checked against each
other -- standard banking practice.

Also provides two sensitivity grids (WACC x Terminal Growth, and
WACC x Exit Multiple) by re-running the model across a range of inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from financials import CompanyData
from utils import safe_divide
from valuation import calculate_equity_value


def estimate_default_wacc(
    company: CompanyData,
    risk_free_rate: float = 0.045,
    equity_risk_premium: float = 0.055,
    pretax_cost_of_debt: float = 0.055,
    tax_rate: float = 0.21,
) -> float:
    """
    Estimate a starting WACC via CAPM cost of equity and a market-value
    capital structure weighting, so the DCF's default assumption isn't an
    arbitrary flat number.

        Cost of Equity = risk-free rate + beta * equity risk premium
        WACC = (E/V) * Cost of Equity + (D/V) * pretax cost of debt * (1 - tax rate)

    Falls back to a beta of 1.0 if the company's beta is unavailable, and
    to pure cost of equity if market cap and debt are both zero/missing.
    """
    overview = company.overview
    beta = overview.get("beta") or 1.0
    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    market_cap = overview.get("market_cap") or 0.0
    total_debt = overview.get("total_debt") or 0.0
    total_value = market_cap + total_debt
    if total_value <= 0:
        return cost_of_equity

    equity_weight = market_cap / total_value
    debt_weight = total_debt / total_value
    after_tax_cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)
    return equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt


@dataclass
class DCFAssumptions:
    """
    User-editable DCF inputs. All rates are decimal fractions (0.08 = 8%).

    da_pct_revenue (D&A as % of revenue) isn't in the product spec's
    editable list verbatim, but it's mathematically required to bridge
    EBITDA -> EBIT -> NOPAT correctly, so it's included with a
    historically-derived default rather than left as a hidden constant.
    """

    projection_years: int = 5
    revenue_growth: float = 0.08
    ebitda_margin: float = 0.25
    da_pct_revenue: float = 0.03
    tax_rate: float = 0.21
    capex_pct_revenue: float = 0.04
    nwc_pct_revenue_change: float = 0.10
    wacc: float = 0.09
    terminal_growth: float = 0.025
    exit_multiple: float = 12.0

    @classmethod
    def from_company(cls, company: CompanyData) -> "DCFAssumptions":
        """
        Build starting assumptions from the company's own historical
        averages (last up-to-3 fiscal years) and an estimated WACC, so the
        DCF opens on defensible numbers instead of generic guesses. Falls
        back to the dataclass defaults for any metric with insufficient
        historical data.
        """
        defaults = cls()
        df = company.annual_summary
        if df.empty:
            return defaults

        recent = df.tail(3)
        cagr_info = company.cagr_summary(years=min(3, len(df) - 1))

        revenue_growth = cagr_info.get("revenue_cagr")
        ebitda_margin = recent["ebitda_margin"].mean()
        da_pct_revenue = safe_divide(recent["d_and_a"].mean(), recent["revenue"].mean())
        capex_pct_revenue = safe_divide(-recent["capex"].mean(), recent["revenue"].mean())

        return cls(
            revenue_growth=_clean(revenue_growth, defaults.revenue_growth),
            ebitda_margin=_clean(ebitda_margin, defaults.ebitda_margin),
            da_pct_revenue=_clean(da_pct_revenue, defaults.da_pct_revenue),
            capex_pct_revenue=_clean(capex_pct_revenue, defaults.capex_pct_revenue),
            wacc=_clean(estimate_default_wacc(company), defaults.wacc),
        )


def _clean(value: Optional[float], fallback: float) -> float:
    """Return `value` if it's a finite, sane number, else `fallback`."""
    if value is None or not np.isfinite(value):
        return fallback
    return float(value)


@dataclass
class DCFResult:
    """Full output of a single DCF run under one set of assumptions."""

    projections: pd.DataFrame
    sum_pv_explicit_fcf: float
    terminal_value_gordon: Optional[float]
    terminal_value_exit_multiple: float
    enterprise_value_gordon: Optional[float]
    enterprise_value_exit_multiple: float
    equity_value_gordon: Optional[float]
    equity_value_exit_multiple: float
    implied_share_price_gordon: Optional[float]
    implied_share_price_exit_multiple: Optional[float]
    warning: Optional[str] = None


class DCFModel:
    """
    Runs a discounted cash flow valuation for a single company under a
    given set of assumptions.

    Usage:
        assumptions = DCFAssumptions.from_company(company)
        model = DCFModel(company, assumptions)
        result = model.run()
        grid = model.sensitivity_wacc_growth(wacc_range=[...], growth_range=[...])
    """

    def __init__(self, company: CompanyData, assumptions: DCFAssumptions):
        self.company = company
        self.assumptions = assumptions

    def run(self) -> DCFResult:
        """
        Project unlevered FCF for `projection_years`, discount at WACC, and
        compute EV/Equity Value/implied share price under both terminal
        value methods.

        Raises:
            ValueError: if the company has no usable base-year revenue to
                project from (e.g. no income statement data at all).
        """
        summary_df = self.company.annual_summary
        if summary_df.empty:
            raise ValueError(
                f"Cannot run a DCF for {self.company.ticker}: no historical income "
                "statement data is available to establish a base revenue figure."
            )
        base_revenue = summary_df.iloc[-1]["revenue"]
        if base_revenue is None or not np.isfinite(base_revenue):
            raise ValueError(
                f"Cannot run a DCF for {self.company.ticker}: base-year revenue is missing."
            )

        a = self.assumptions
        rows = []
        revenue = prior_revenue = float(base_revenue)

        for year in range(1, a.projection_years + 1):
            revenue = revenue * (1 + a.revenue_growth)
            ebitda = revenue * a.ebitda_margin
            da = revenue * a.da_pct_revenue
            ebit = ebitda - da
            nopat = ebit * (1 - a.tax_rate)
            capex = revenue * a.capex_pct_revenue
            delta_nwc = (revenue - prior_revenue) * a.nwc_pct_revenue_change
            unlevered_fcf = nopat + da - capex - delta_nwc
            discount_factor = 1 / (1 + a.wacc) ** year
            pv_fcf = unlevered_fcf * discount_factor

            rows.append({
                "year": year, "revenue": revenue, "ebitda": ebitda, "ebit": ebit,
                "nopat": nopat, "da": da, "capex": capex, "delta_nwc": delta_nwc,
                "unlevered_fcf": unlevered_fcf, "discount_factor": discount_factor, "pv_fcf": pv_fcf,
            })
            prior_revenue = revenue

        projections = pd.DataFrame(rows).set_index("year")
        sum_pv_fcf = projections["pv_fcf"].sum()
        terminal_year = projections.iloc[-1]
        final_discount_factor = terminal_year["discount_factor"]

        warning = None
        if a.wacc <= a.terminal_growth:
            terminal_value_gordon = None
            ev_gordon = equity_gordon = price_gordon = None
            warning = "WACC must exceed the terminal growth rate for the Gordon Growth method to be valid."
        else:
            terminal_value_gordon = terminal_year["unlevered_fcf"] * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth)
            ev_gordon = sum_pv_fcf + terminal_value_gordon * final_discount_factor
            equity_gordon = calculate_equity_value(
                ev_gordon, self.company.overview["total_debt"], self.company.overview["total_cash"]
            )
            price_gordon = safe_divide(equity_gordon, self.company.overview["shares_outstanding"])

        terminal_value_exit = terminal_year["ebitda"] * a.exit_multiple
        ev_exit = sum_pv_fcf + terminal_value_exit * final_discount_factor
        equity_exit = calculate_equity_value(
            ev_exit, self.company.overview["total_debt"], self.company.overview["total_cash"]
        )
        price_exit = safe_divide(equity_exit, self.company.overview["shares_outstanding"])

        return DCFResult(
            projections=projections,
            sum_pv_explicit_fcf=sum_pv_fcf,
            terminal_value_gordon=terminal_value_gordon,
            terminal_value_exit_multiple=terminal_value_exit,
            enterprise_value_gordon=ev_gordon,
            enterprise_value_exit_multiple=ev_exit,
            equity_value_gordon=equity_gordon,
            equity_value_exit_multiple=equity_exit,
            implied_share_price_gordon=price_gordon,
            implied_share_price_exit_multiple=price_exit,
            warning=warning,
        )

    def sensitivity_wacc_growth(self, wacc_range: Sequence[float], growth_range: Sequence[float]) -> pd.DataFrame:
        """
        Grid of implied share price (Gordon Growth method) across a range
        of WACC (rows) and terminal growth rate (columns) values. Cells
        where WACC <= growth are mathematically invalid and left as NaN.
        """
        grid = pd.DataFrame(index=list(wacc_range), columns=list(growth_range), dtype=float)
        for wacc in wacc_range:
            for growth in growth_range:
                if wacc <= growth:
                    grid.loc[wacc, growth] = np.nan
                    continue
                scenario = replace(self.assumptions, wacc=wacc, terminal_growth=growth)
                grid.loc[wacc, growth] = DCFModel(self.company, scenario).run().implied_share_price_gordon
        grid.index.name, grid.columns.name = "WACC", "Terminal Growth"
        return grid

    def sensitivity_wacc_exit_multiple(self, wacc_range: Sequence[float], multiple_range: Sequence[float]) -> pd.DataFrame:
        """
        Grid of implied share price (Exit Multiple method) across a range
        of WACC (rows) and exit EV/EBITDA multiple (columns) values.
        """
        grid = pd.DataFrame(index=list(wacc_range), columns=list(multiple_range), dtype=float)
        for wacc in wacc_range:
            for multiple in multiple_range:
                scenario = replace(self.assumptions, wacc=wacc, exit_multiple=multiple)
                grid.loc[wacc, multiple] = DCFModel(self.company, scenario).run().implied_share_price_exit_multiple
        grid.index.name, grid.columns.name = "WACC", "Exit Multiple"
        return grid
