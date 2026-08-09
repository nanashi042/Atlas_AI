import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from app.services.finance.finnhub_client import (
    FinnhubClient,
    FinnhubException,
    FinnhubNotFoundError,
)
from app.services.finance.models import CompanyResearchResult, CompanyNewsItem

logger = logging.getLogger(__name__)


class CompanyResearchService:
    """
    Orchestrates company profile, stock quote, and news data from financial data providers (Finnhub).
    Normalizes provider responses into an internal CompanyResearchResult model.
    """

    def __init__(self, client: Optional[FinnhubClient] = None):
        self.client = client or FinnhubClient()

    async def get_company_research(self, symbol: str) -> CompanyResearchResult:
        """
        Retrieves company profile, current quote, and recent news for a given ticker symbol.
        Normalizes into internal CompanyResearchResult.
        """
        symbol_upper = symbol.upper().strip()
        logger.info(f"Starting company research for symbol: '{symbol_upper}'")

        # 1. Fetch profile
        profile = await self.client.get_company_profile(symbol_upper)
        
        # If profile is empty, check quote as fallback to verify symbol existence
        quote = await self.client.get_quote(symbol_upper)

        # Determine company name and basic existence
        company_name = profile.get("name")
        current_price = quote.get("c")

        # Symbol is considered not found if profile is empty AND quote current_price is 0/None
        if not company_name and (current_price is None or current_price == 0):
            logger.warning(f"Company profile and quote empty for symbol: '{symbol_upper}'")
            raise FinnhubNotFoundError(f"Could not find financial data for ticker symbol '{symbol_upper}'.")

        # Fallback company name if profile was sparse
        if not company_name:
            company_name = f"{symbol_upper} Inc."

        # 2. Fetch recent news (past 7 days)
        today = datetime.now(timezone.utc).date()
        from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        recent_news_raw: List[Dict[str, Any]] = []
        try:
            recent_news_raw = await self.client.get_company_news(symbol_upper, from_date=from_date, to_date=to_date)
        except Exception as e:
            logger.warning(f"Failed to fetch news for '{symbol_upper}': {e}. Proceeding without news.")

        # 3. Normalize news items
        news_items: List[CompanyNewsItem] = []
        for raw in recent_news_raw[:5]:  # Take top 5 news items
            headline = raw.get("headline")
            if not headline:
                continue

            summary = raw.get("summary", "")
            source = raw.get("source", "Market News")
            url = raw.get("url", "")
            
            # Format datetime if present
            epoch_time = raw.get("datetime")
            published_at = ""
            if epoch_time:
                try:
                    published_at = datetime.fromtimestamp(epoch_time, timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    published_at = ""

            news_items.append(CompanyNewsItem(
                headline=headline,
                summary=summary,
                source=source,
                url=url,
                published_at=published_at,
            ))

        # 4. Construct internal model
        result = CompanyResearchResult(
            symbol=symbol_upper,
            company_name=company_name,
            exchange=profile.get("exchange") or "N/A",
            country=profile.get("country") or "N/A",
            currency=profile.get("currency") or "USD",
            industry=profile.get("finnhubIndustry") or "N/A",
            website=profile.get("weburl") or "N/A",
            current_price=quote.get("c") if quote.get("c") != 0 else None,
            price_change=quote.get("d"),
            percent_change=quote.get("dp"),
            previous_close=quote.get("pc"),
            day_high=quote.get("h"),
            day_low=quote.get("l"),
            recent_news=news_items,
        )

        logger.info(f"Successfully completed company research for '{symbol_upper}'.")
        return result
