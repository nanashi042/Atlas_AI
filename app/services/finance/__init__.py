"""
Finance services module for Atlas AI Financial Copilot.
Provides market data providers, company research orchestration, and data models.
"""

from app.services.finance.models import CompanyResearchResult, CompanyNewsItem
from app.services.finance.finnhub_client import FinnhubClient
from app.services.finance.company_research import CompanyResearchService

__all__ = [
    "CompanyResearchResult",
    "CompanyNewsItem",
    "FinnhubClient",
    "CompanyResearchService",
]
