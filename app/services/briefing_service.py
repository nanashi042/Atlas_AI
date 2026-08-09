"""
Personalized Financial Briefing Engine v1 for Atlas AI.

BriefingService orchestrates the production of a concise, personalized
financial briefing for a single user. The service:

- Pulls the user's watchlist from `WatchlistService` (the single source
  of truth — conversation memory is never used as a source).
- For each watched ticker, reuses `CompanyResearchService.get_company_research`
  (no duplicate Finnhub HTTP client, no duplicate API key handling).
- Filters retrieved news to items that are clearly relevant to the
  specific company/ticker.
- Normalizes everything into `BriefingData` and `BriefingItem` dataclasses
  so raw Finnhub JSON never reaches Gemini.
- Synthesizes a concise, mobile-friendly briefing via Gemini using a
  dedicated prompt that forbids fabrication and buy/sell recommendations.

The service is intentionally Telegram-free and stateless (it takes a
`user_id` string and returns a ready-to-send string) so a future
scheduler feature can reuse it without modifications.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from app.services.finance.company_research import CompanyResearchService
from app.services.finance.finnhub_client import (
    FinnhubAPIError,
    FinnhubAuthError,
    FinnhubException,
    FinnhubNotFoundError,
    FinnhubRateLimitError,
    FinnhubTimeoutError,
)
from app.services.finance.models import CompanyNewsItem, CompanyResearchResult
from app.services.watchlist_service import WatchlistError, watchlist_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User-facing messages
# ---------------------------------------------------------------------------

EMPTY_WATCHLIST_MESSAGE = (
    "Your watchlist is empty. "
    "Add a company first, for example: 'Track Nvidia'."
)

ALL_FINANCE_FAILURE_MESSAGE = (
    "I couldn't retrieve your market data right now. Please try again shortly."
)

WATCHLIST_LOAD_ERROR_MESSAGE = (
    "⚠️ I couldn't load your watchlist right now. Please try again shortly."
)

UNEXPECTED_ERROR_MESSAGE = (
    "⚠️ I couldn't generate your briefing right now. Please try again shortly."
)

# Maximum news items per company that survive the relevance filter.
MAX_NEWS_PER_COMPANY = 2


# ---------------------------------------------------------------------------
# Internal data models
# ---------------------------------------------------------------------------


@dataclass
class BriefingItem:
    """
    Normalized per-company briefing entry.

    Only fields relevant to a watchlist briefing are kept. Raw Finnhub
    fields (datetime epoch, weburl, finnhubIndustry, etc.) are not
    exposed here so they cannot leak into the LLM prompt.
    """

    symbol: str
    company_name: str
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    percent_change: Optional[float] = None
    previous_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    relevant_news: List[CompanyNewsItem] = field(default_factory=list)
    brief_context: Optional[str] = None
    data_available: bool = True

    def to_context_block(self) -> str:
        """Render the item as a structured text block for the LLM prompt."""
        lines = [f"Company: {self.company_name} ({self.symbol})"]
        if not self.data_available:
            lines.append(
                "Data Status: UNAVAILABLE (finance provider error). "
                "Do not invent values for this company."
            )
            return "\n".join(lines)

        if self.current_price is not None:
            lines.append(f"Current Price: ${self.current_price:.2f}")
        if self.price_change is not None and self.percent_change is not None:
            sign = "+" if self.price_change > 0 else ""
            lines.append(
                f"Daily Change: {sign}{self.price_change:.2f} "
                f"({sign}{self.percent_change:.2f}%)"
            )
        if self.previous_close is not None:
            lines.append(f"Previous Close: ${self.previous_close:.2f}")
        if self.day_high is not None and self.day_low is not None:
            lines.append(f"Day Range: ${self.day_low:.2f} - ${self.day_high:.2f}")

        if self.brief_context:
            lines.append(f"Derived Context: {self.brief_context}")

        if self.relevant_news:
            lines.append("Relevant News:")
            for idx, item in enumerate(self.relevant_news, 1):
                published = f" ({item.published_at})" if item.published_at else ""
                source = f" [{item.source}]" if item.source else ""
                lines.append(f"  {idx}. {item.headline}{source}{published}")
                if item.summary:
                    summary_clean = (
                        item.summary[:150] + "..."
                        if len(item.summary) > 150
                        else item.summary
                    )
                    lines.append(f"     Summary: {summary_clean}")
        else:
            lines.append("Relevant News: No major company-specific news found.")

        return "\n".join(lines)


@dataclass
class BriefingData:
    """
    Normalized top-level briefing payload.

    Holds the assembled briefing for a single user. `unavailable_symbols`
    tracks companies whose finance data could not be retrieved, so the
    LLM prompt (and the fallback template) can mention them honestly
    instead of fabricating data.
    """

    user_id: str
    generated_at: datetime
    items: List[BriefingItem] = field(default_factory=list)
    unavailable_symbols: List[str] = field(default_factory=list)
    market_context: Optional[str] = None


# ---------------------------------------------------------------------------
# News relevance filter
# ---------------------------------------------------------------------------

_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|plc|holdings|technologies|"
    r"tech|group|services|solutions|industries|systems)\.?\b",
    re.IGNORECASE,
)


def _candidate_name_variants(company_name: str) -> List[str]:
    """
    Build a small list of name variants used to decide whether a news
    article is about a specific company.

    Returns the original (lower-cased) name plus a suffix-stripped core
    name. Both are returned as lowercased strings.
    """
    variants: List[str] = []
    raw = (company_name or "").strip().lower()
    if raw:
        variants.append(raw)
    core = _COMPANY_SUFFIX_PATTERN.sub("", raw).strip()
    if core and core != raw:
        variants.append(core)
    return variants


def _is_news_relevant(news: CompanyNewsItem, symbol: str, company_name: str) -> bool:
    """
    Return True only when the article headline or summary clearly
    references the company being briefed.

    The check requires the ticker (e.g. "NVDA") OR a name variant (e.g.
    "nvidia" or "nvidia corp") to appear as a substring of the
    combined headline + summary. This deliberately drops unrelated
    articles that Finnhub's company-news endpoint sometimes returns
    (e.g. an NVIDIA briefing that contains Tesla or Micron news).
    """
    text = f"{news.headline or ''} {news.summary or ''}".lower()
    if not text.strip():
        return False

    candidates: List[str] = [symbol.strip().lower()]
    candidates.extend(_candidate_name_variants(company_name))

    for needle in candidates:
        if needle and needle in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_BRIEFING_PROMPT_TEMPLATE = """You are Atlas AI Financial Copilot preparing a concise personalized morning briefing for a busy professional.

Use ONLY the supplied financial/news data below. Never invent:
- stock prices
- percentage changes
- company news
- dates
- financial facts

Do not make personalized buy, sell, or investment recommendations.

If we have a price movement but no clear company-specific reason in the supplied data, say so explicitly:
"<TICKER> is up X.XX% today; no clear company-specific catalyst was identified in the retrieved data."

If a company's data is marked UNAVAILABLE, mention it briefly and skip it. Do not invent values for it.

Format the response so it reads well on a phone. Use this structure ONLY when data allows:

Good morning

Here's what matters for your watchlist today.

<UP/DOWN EMOJI> <Company> (<TICKER>)
$<price> (+/-X.XX%)
• <one or two important facts from the data>
• <why it matters, ONLY when supported by the data>

[repeat per available company]

🌐 Market context
<one or two sentences, ONLY if reliable context was supplied>

🔎 Overall takeaway
<one short paragraph explaining the most important thing>

Source: Finnhub

Rules:
- Skip any section that lacks reliable data.
- Do not invent financial facts.
- Do not produce buy/sell recommendations.
- Keep the entire response under ~1500 characters.
- Be concise. Avoid repeating the same information.
- Do not introduce yourself as Atlas in the response.

User request: {user_message}

Generated at: {generated_at}

SUPPLIED BRIEFING DATA (from Finnhub, normalized):
{briefing_context}
"""


def _build_briefing_prompt(user_message: str, briefing: BriefingData) -> str:
    """Build the dedicated briefing prompt with normalized context only."""
    context = _format_briefing_context(briefing)
    generated_at_str = briefing.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    return _BRIEFING_PROMPT_TEMPLATE.format(
        user_message=user_message or "Give me my morning briefing.",
        generated_at=generated_at_str,
        briefing_context=context,
    )


def _format_briefing_context(briefing: BriefingData) -> str:
    """Convert a BriefingData into a structured text block for the LLM."""
    sections: List[str] = []

    if briefing.items:
        sections.append("WATCHLIST COMPANIES:")
        for item in briefing.items:
            sections.append(item.to_context_block())
            sections.append("")
    else:
        sections.append("WATCHLIST COMPANIES: None available.")

    if briefing.unavailable_symbols:
        sections.append(
            "UNAVAILABLE COMPANIES (finance provider error): "
            + ", ".join(briefing.unavailable_symbols)
        )

    if briefing.market_context:
        sections.append(f"MARKET CONTEXT: {briefing.market_context}")

    return "\n".join(sections).strip()


# ---------------------------------------------------------------------------
# Fallback template (used when Gemini is unavailable)
# ---------------------------------------------------------------------------


def _format_price_line(item: BriefingItem) -> str:
    """Render a single company's price line for the deterministic fallback."""
    if item.current_price is None:
        return f"${'—'} (price unavailable)"
    sign = "+" if (item.price_change or 0) > 0 else ""
    if item.percent_change is not None:
        return f"${item.current_price:.2f} ({sign}{item.percent_change:.2f}%)"
    return f"${item.current_price:.2f}"


def _build_fallback_text(briefing: BriefingData) -> str:
    """
    Deterministic template rendered directly from normalized data.

    Used when Gemini is unavailable. Never fabricates facts — it only
    restates the structured values returned by the finance service and
    notes when a company is unavailable.
    """
    lines: List[str] = ["Good morning ☀️", "", "Here's what matters for your watchlist today.", ""]

    for item in briefing.items:
        if not item.data_available:
            lines.append(f"⚠️ {item.company_name} ({item.symbol}) — data unavailable for this briefing.")
            lines.append("")
            continue
        emoji = "📈" if (item.percent_change or 0) >= 0 else "📉"
        lines.append(f"{emoji} {item.company_name} ({item.symbol})")
        lines.append(f"   {_format_price_line(item)}")
        if item.relevant_news:
            first = item.relevant_news[0]
            lines.append(f"   • {first.headline}")
        else:
            lines.append("   • No major company-specific news found.")
        if item.percent_change is not None and abs(item.percent_change) >= 1.0:
            lines.append(
                f"   • {item.symbol} is "
                f"{'up' if item.percent_change > 0 else 'down'} "
                f"{item.percent_change:+.2f}% today; no clear company-specific "
                "catalyst was identified in the retrieved data."
            )
        lines.append("")

    if briefing.unavailable_symbols:
        skipped = ", ".join(briefing.unavailable_symbols)
        lines.append(f"Note: data was unavailable for {skipped}.")
        lines.append("")

    lines.append("Source: Finnhub")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# BriefingService
# ---------------------------------------------------------------------------


class BriefingError(Exception):
    """Raised by BriefingService for internal/unexpected failures."""


class BriefingService:
    """
    Orchestrates a personalized financial briefing for a single user.

    Reuses:
      - `WatchlistService.get_watchlist`
      - `CompanyResearchService.get_company_research` (which itself
        calls the existing `FinnhubClient`)
      - `app.ai.llm.generate_response` (the existing Gemini wrapper
        with model fallback)

    No Telegram imports, no direct SQLAlchemy session, no raw Finnhub
    JSON reaches the LLM.
    """

    def __init__(
        self,
        research_service: Optional[CompanyResearchService] = None,
        llm_generate=None,
        watchlist=None,
    ):
        # Allow dependency injection for tests.
        self.research_service = research_service or CompanyResearchService()
        self.llm_generate = llm_generate  # callable: async (prompt, history=None)
        self.watchlist = watchlist or watchlist_service

    async def generate_briefing(
        self, user_id: str, user_message: str = "Give me my morning briefing."
    ) -> str:
        """
        Produce a briefing string for the given user.

        Returns one of:
          - EMPTY_WATCHLIST_MESSAGE when the user has no watchlist.
          - WATCHLIST_LOAD_ERROR_MESSAGE on watchlist DB failure.
          - ALL_FINANCE_FAILURE_MESSAGE when every ticker failed.
          - A briefing string (LLM-synthesized or deterministic fallback)
            when at least one ticker returned data.
          - UNEXPECTED_ERROR_MESSAGE on any uncaught exception.
        """
        if not user_id:
            logger.warning("BriefingService.generate_briefing called with empty user_id.")
            return EMPTY_WATCHLIST_MESSAGE

        try:
            entries = self.watchlist.get_watchlist(user_id)
        except WatchlistError as e:
            logger.error(f"Briefing failed: watchlist load error for user '{user_id}': {e}")
            return WATCHLIST_LOAD_ERROR_MESSAGE

        if not entries:
            logger.info(f"Briefing requested for user '{user_id}' (watchlist size: 0). Empty.")
            return EMPTY_WATCHLIST_MESSAGE

        logger.info(
            f"Briefing requested for user '{user_id}' (watchlist size: {len(entries)})."
        )

        briefing = BriefingData(
            user_id=str(user_id),
            generated_at=datetime.now(timezone.utc),
        )

        for entry in entries:
            symbol = entry["symbol"]
            company_name = entry["company_name"]
            try:
                result = await self.research_service.get_company_research(symbol)
                item = _build_briefing_item(result, company_name)
                briefing.items.append(item)
                logger.info(
                    f"Finance data retrieved for '{symbol}' (briefing user '{user_id}')."
                )
            except FinnhubAuthError as e:
                logger.warning(
                    f"Finance data unavailable for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)
            except FinnhubRateLimitError as e:
                logger.warning(
                    f"Finance data unavailable for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)
            except FinnhubTimeoutError as e:
                logger.warning(
                    f"Finance data unavailable for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)
            except FinnhubNotFoundError as e:
                logger.warning(
                    f"Finance data unavailable for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)
            except FinnhubAPIError as e:
                logger.warning(
                    f"Finance data unavailable for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)
            except FinnhubException as e:
                logger.warning(
                    f"Finance data unavailable for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)
            except Exception as e:  # defensive: never let one ticker break the briefing
                logger.warning(
                    f"Unexpected finance error for '{symbol}' "
                    f"(briefing user '{user_id}'): {e}"
                )
                briefing.unavailable_symbols.append(symbol)

        if not briefing.items:
            logger.error(
                f"Briefing total finance failure for user '{user_id}' "
                f"(unavailable: {briefing.unavailable_symbols})."
            )
            return ALL_FINANCE_FAILURE_MESSAGE

        # Synthesize with Gemini (or fall back to deterministic template).
        try:
            response_text = await self._synthesize(user_message, briefing)
        except Exception as e:
            logger.error(f"Briefing generation failed for user '{user_id}': {e}", exc_info=True)
            response_text = _build_fallback_text(briefing)

        # Gemini returned an error code string — fall back deterministically.
        if not response_text or response_text.startswith("[Error"):
            logger.warning(
                f"Briefing Gemini fallback for user '{user_id}' "
                f"(reason: {response_text[:120] if response_text else 'empty'})."
            )
            response_text = _build_fallback_text(briefing)

        logger.info(
            f"Briefing generated for user '{user_id}' "
            f"(companies: {len(briefing.items)}, "
            f"unavailable: {len(briefing.unavailable_symbols)})."
        )
        return response_text

    async def _synthesize(self, user_message: str, briefing: BriefingData) -> str:
        """Generate the briefing text using the configured LLM callable."""
        prompt = _build_briefing_prompt(user_message, briefing)
        if self.llm_generate is None:
            # Late import to avoid a circular import at module load.
            from app.ai.llm import generate_response

            return await generate_response(prompt, history=[])
        return await self.llm_generate(prompt, history=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_briefing_item(result: CompanyResearchResult, fallback_name: str) -> BriefingItem:
    """
    Project a `CompanyResearchResult` into a `BriefingItem`, applying the
    news relevance filter. If all news is filtered out, the item simply
    carries an empty list and the prompt will instruct Gemini to say so.
    """
    company_name = result.company_name or fallback_name

    relevant_news: List[CompanyNewsItem] = []
    for news in result.recent_news:
        if _is_news_relevant(news, result.symbol, company_name):
            relevant_news.append(news)
        if len(relevant_news) >= MAX_NEWS_PER_COMPANY:
            break

    brief_context: Optional[str] = None
    if result.percent_change is not None and abs(result.percent_change) >= 1.0:
        direction = "up" if result.percent_change > 0 else "down"
        brief_context = (
            f"Trading {direction} {abs(result.percent_change):.2f}% today; "
            "no further catalyst available beyond the supplied news."
        )

    return BriefingItem(
        symbol=result.symbol,
        company_name=company_name,
        current_price=result.current_price,
        price_change=result.price_change,
        percent_change=result.percent_change,
        previous_close=result.previous_close,
        day_high=result.day_high,
        day_low=result.day_low,
        relevant_news=relevant_news,
        brief_context=brief_context,
        data_available=True,
    )


# Module-level singleton — handlers and the manager import this directly.
briefing_service = BriefingService()
