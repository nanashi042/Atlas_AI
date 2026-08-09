from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompanyNewsItem:
    """
    Internal data model for a single company news item normalized from Finnhub.
    """
    headline: str
    summary: str
    source: str
    url: str
    published_at: str


@dataclass
class CompanyResearchResult:
    """
    Internal data model representing normalized company research information.
    Isolates external API responses (Finnhub JSON) from internal business & LLM logic.
    """
    symbol: str
    company_name: str
    exchange: str = "N/A"
    country: str = "N/A"
    currency: str = "USD"
    industry: str = "N/A"
    website: str = "N/A"
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    percent_change: Optional[float] = None
    previous_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    recent_news: List[CompanyNewsItem] = field(default_factory=list)

    def to_formatted_context(self) -> str:
        """
        Formats the research data into a clear, structured context string for Gemini prompt.
        """
        lines = [
            f"Company Name: {self.company_name} ({self.symbol})",
            f"Exchange: {self.exchange}",
            f"Country: {self.country}",
            f"Currency: {self.currency}",
            f"Industry: {self.industry}",
            f"Website: {self.website}",
        ]

        if self.current_price is not None:
            price_str = f"${self.current_price:.2f}"
            change_str = (
                f"{'+' if self.price_change and self.price_change > 0 else ''}{self.price_change:.2f}"
                if self.price_change is not None else "N/A"
            )
            pct_str = (
                f"{'+' if self.percent_change and self.percent_change > 0 else ''}{self.percent_change:.2f}%"
                if self.percent_change is not None else "N/A"
            )
            lines.append(f"Current Price: {price_str}")
            lines.append(f"Price Change Today: {change_str} ({pct_str})")

        if self.previous_close is not None:
            lines.append(f"Previous Close: ${self.previous_close:.2f}")
        if self.day_high is not None and self.day_low is not None:
            lines.append(f"Day Range: ${self.day_low:.2f} - ${self.day_high:.2f}")

        if self.recent_news:
            lines.append("\nRecent News Highlights:")
            for idx, item in enumerate(self.recent_news[:3], 1):
                published = f" ({item.published_at})" if item.published_at else ""
                source = f" [{item.source}]" if item.source else ""
                lines.append(f"{idx}. {item.headline}{source}{published}")
                if item.summary:
                    # Truncate summary if too long
                    summary_clean = item.summary[:150] + "..." if len(item.summary) > 150 else item.summary
                    lines.append(f"   Summary: {summary_clean}")

        return "\n".join(lines)
