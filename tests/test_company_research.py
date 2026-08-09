import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.finance.company_research import CompanyResearchService
from app.services.finance.company_resolver import resolve_company_ticker
from app.services.finance.finnhub_client import FinnhubNotFoundError
from app.services.finance.models import CompanyResearchResult


class TestCompanyResearch(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_client = AsyncMock()
        self.service = CompanyResearchService(client=self.mock_client)

    async def test_successful_research_normalization(self):
        self.mock_client.get_company_profile.return_value = {
            "name": "NVIDIA Corp",
            "ticker": "NVDA",
            "exchange": "NASDAQ NMS",
            "country": "US",
            "currency": "USD",
            "finnhubIndustry": "Semiconductors",
            "weburl": "https://www.nvidia.com/",
        }

        self.mock_client.get_quote.return_value = {
            "c": 125.50,
            "d": 2.50,
            "dp": 2.03,
            "pc": 123.00,
            "h": 126.00,
            "l": 122.50,
        }

        self.mock_client.get_company_news.return_value = [
            {
                "headline": "Nvidia announces new AI chip family",
                "summary": "Nvidia unveiled its latest GPU architecture today.",
                "source": "Reuters",
                "url": "https://reuters.com/nvidia-chip",
                "datetime": 1723145600,
            }
        ]

        result = await self.service.get_company_research("NVDA")

        self.assertIsInstance(result, CompanyResearchResult)
        self.assertEqual(result.symbol, "NVDA")
        self.assertEqual(result.company_name, "NVIDIA Corp")
        self.assertEqual(result.current_price, 125.50)
        self.assertEqual(result.percent_change, 2.03)
        self.assertEqual(len(result.recent_news), 1)
        self.assertEqual(result.recent_news[0].headline, "Nvidia announces new AI chip family")
        self.assertEqual(result.recent_news[0].source, "Reuters")

    async def test_company_not_found(self):
        self.mock_client.get_company_profile.return_value = {}
        self.mock_client.get_quote.return_value = {"c": 0}

        with self.assertRaises(FinnhubNotFoundError):
            await self.service.get_company_research("NONEXISTENT_TICKER")

    def test_ticker_resolution(self):
        # Company names
        self.assertEqual(resolve_company_ticker("Tell me about Nvidia"), "NVDA")
        self.assertEqual(resolve_company_ticker("What is Tesla?"), "TSLA")
        self.assertEqual(resolve_company_ticker("Research Microsoft"), "MSFT")
        self.assertEqual(resolve_company_ticker("How is Apple doing?"), "AAPL")
        self.assertEqual(resolve_company_ticker("Give me info on Google"), "GOOGL")

        # Explicit tickers
        self.assertEqual(resolve_company_ticker("What is NVDA?"), "NVDA")
        self.assertEqual(resolve_company_ticker("Tell me about $AAPL"), "AAPL")
        self.assertEqual(resolve_company_ticker("TSLA"), "TSLA")

        # Unknown / Ambiguous query
        self.assertIsNone(resolve_company_ticker("Tell me about ABC_XYZ_UNKNOWN"))


if __name__ == "__main__":
    unittest.main()
