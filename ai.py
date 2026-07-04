"""
ai.py
-----
Claude-powered analyst writing: investment memo generation and news
summarization. This module sits on top of the rest of the pipeline --
it accepts already-computed domain objects from financials.py,
valuation.py, dcf.py, and comps.py and turns them into prose.

AI calls are optional. If ANTHROPIC_API_KEY isn't configured, or the API
call fails for any reason, functions raise AIError with a clear message
instead of letting an exception surface from deep inside the Anthropic
SDK -- callers (app.py) can catch this and disable the AI panel instead
of crashing the whole dashboard.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from comps import CompsResult
from dcf import DCFResult
from financials import CompanyData
from utils import format_currency, format_percent, format_ratio
from valuation import ValuationSummary

load_dotenv()

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


class AIError(Exception):
    """Raised when AI features are unavailable (no API key) or a call fails."""


def _get_client():
    """
    Lazily construct the Anthropic client. Imported inside the function
    (rather than at module load) so that simply importing ai.py doesn't
    require the `anthropic` package to be configured with a key --
    only actually calling an AI feature does.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIError(
            "ANTHROPIC_API_KEY is not set. Add it to a .env file (see .env.example) to enable AI features."
        )
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _call_claude(prompt: str, system: str, max_tokens: int = 1600, model: str = DEFAULT_MODEL) -> str:
    """Shared call path for every AI feature in this module. Wraps SDK errors as AIError."""
    client = _get_client()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # covers auth errors, rate limits, network failures
        raise AIError(f"AI request failed: {exc}") from exc
    return response.content[0].text.strip()


_MEMO_SYSTEM_PROMPT = (
    "You are a senior equity research analyst at a top-tier investment bank writing an "
    "internal research memo. Your tone is precise, data-driven, and balanced -- you cite "
    "the specific figures you're given rather than making up numbers, you present both "
    "bull and bear considerations, and you never give definitive investment advice without "
    "hedging language appropriate for institutional research (e.g. 'suggests', 'implies', "
    "'may warrant'). End with a standard research disclaimer noting this is for "
    "informational/educational purposes and not a solicitation to buy or sell securities."
)


def generate_investment_memo(
    company: CompanyData,
    valuation: ValuationSummary,
    dcf_result: Optional[DCFResult] = None,
    comps_result: Optional[CompsResult] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a full analyst-style investment memo covering business overview,
    investment thesis, key strengths, risks, industry outlook, valuation
    summary, and a recommendation.

    Args:
        company: CompanyData for the subject ticker.
        valuation: ValuationSummary (from valuation.build_valuation_summary).
        dcf_result: Optional DCFResult to ground the valuation section in a
            specific DCF output; omitted sections are simply left out of
            the prompt rather than causing an error.
        comps_result: Optional CompsResult for peer-relative context.
        model: Anthropic model ID to use.

    Returns:
        Markdown-formatted memo text.

    Raises:
        AIError: if the API key is missing or the request fails.
    """
    prompt = _build_memo_prompt(company, valuation, dcf_result, comps_result)
    return _call_claude(prompt, system=_MEMO_SYSTEM_PROMPT, max_tokens=2200, model=model)


def _build_memo_prompt(
    company: CompanyData,
    valuation: ValuationSummary,
    dcf_result: Optional[DCFResult],
    comps_result: Optional[CompsResult],
) -> str:
    overview = company.overview
    summary_df = company.annual_summary
    cagrs = company.cagr_summary(years=3)

    lines = [
        f"Write an investment memo for {overview['name']} ({overview['ticker']}).",
        "",
        "## Company Facts",
        f"- Sector / Industry: {overview['sector']} / {overview['industry']}",
        f"- Market Cap: {format_currency(valuation.market_cap)}",
        f"- Enterprise Value: {format_currency(valuation.enterprise_value)}",
        f"- Business description: {overview['description'][:600]}",
        "",
        "## Financial Performance (most recent fiscal year)",
        f"- Revenue: {format_currency(valuation.revenue)}",
        f"- EBITDA: {format_currency(valuation.ebitda)} "
        f"(margin: {format_percent(summary_df['ebitda_margin'].iloc[-1]) if not summary_df.empty else 'N/A'})",
        f"- Net Income: {format_currency(valuation.net_income)}",
        f"- Diluted EPS: {valuation.diluted_eps}",
        f"- 3-Year Revenue CAGR: {format_percent(cagrs.get('revenue_cagr'))}",
        f"- 3-Year EBITDA CAGR: {format_percent(cagrs.get('ebitda_cagr'))}",
        "",
        "## Valuation Multiples",
        f"- EV/Revenue: {format_ratio(valuation.ev_to_revenue)}",
        f"- EV/EBITDA: {format_ratio(valuation.ev_to_ebitda)}",
        f"- P/E: {format_ratio(valuation.price_to_earnings)}",
        f"- P/B: {format_ratio(valuation.price_to_book)}",
    ]

    if comps_result is not None and not comps_result.peer_stats.empty:
        lines += [
            "",
            "## Peer Comparison (median of peer group)",
            f"- Peer median EV/EBITDA: {format_ratio(comps_result.peer_stats.loc['median', 'ev_to_ebitda'])}",
            f"- Peer median EV/Revenue: {format_ratio(comps_result.peer_stats.loc['median', 'ev_to_revenue'])}",
            f"- Peer median P/E: {format_ratio(comps_result.peer_stats.loc['median', 'price_to_earnings'])}",
        ]

    if dcf_result is not None:
        lines += [
            "",
            "## DCF Output",
            f"- Implied share price (Gordon Growth method): {format_currency(dcf_result.implied_share_price_gordon)}",
            f"- Implied share price (Exit Multiple method): {format_currency(dcf_result.implied_share_price_exit_multiple)}",
            f"- Current market price: {format_currency(overview['current_price'])}",
        ]

    lines += [
        "",
        "Structure the memo with these exact section headers (as markdown H3, i.e. '### '): "
        "Business Overview, Investment Thesis, Key Strengths, Risks, Industry Outlook, "
        "Valuation Summary, Recommendation. Keep each section to 2-4 sentences except "
        "Key Strengths and Risks, which should be 3-4 bullet points each. Base every "
        "claim strictly on the figures provided above -- do not invent financial data "
        "that wasn't given to you.",
    ]
    return "\n".join(lines)


_NEWS_SYSTEM_PROMPT = (
    "You are an equity research analyst summarizing news headlines for a morning brief. "
    "Write exactly 2 sentences per article: what happened, and why it plausibly matters "
    "for the stock. If the headline alone doesn't give enough detail to say why it "
    "matters, say so plainly rather than speculating with invented specifics."
)


def summarize_news_article(title: str, snippet: str = "", model: str = DEFAULT_MODEL) -> str:
    """
    Produce a short analyst-style summary of a news item. yfinance headlines
    often come with little or no article body, so this is designed to work
    from a title alone if `snippet` is empty.

    Args:
        title: The article headline.
        snippet: Any additional teaser/summary text available (may be "").
        model: Anthropic model ID to use.

    Returns:
        A 2-sentence summary string.

    Raises:
        AIError: if the API key is missing or the request fails.
    """
    prompt = f"Headline: {title}\n\nAdditional context: {snippet or '(none available)'}"
    return _call_claude(prompt, system=_NEWS_SYSTEM_PROMPT, max_tokens=200, model=model)
