import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

from app.services.finance.finnhub_client import (
    FinnhubClient,
    FinnhubAuthError,
    FinnhubRateLimitError,
    FinnhubTimeoutError,
    FinnhubNotFoundError,
    FinnhubAPIError,
)


class TestFinnhubClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.client = FinnhubClient(api_key="test_dummy_key")

    async def test_missing_api_key(self):
        client = FinnhubClient(api_key="")
        with patch("app.services.finance.finnhub_client.settings") as mock_settings:
            mock_settings.FINNHUB_API_KEY = ""
            with self.assertRaises(FinnhubAuthError):
                await client.get_company_profile("NVDA")

    @patch("httpx.AsyncClient.get")
    async def test_successful_profile_and_quote(self, mock_get):
        mock_response_profile = MagicMock()
        mock_response_profile.status_code = 200
        mock_response_profile.json.return_value = {
            "name": "NVIDIA Corp",
            "ticker": "NVDA",
            "exchange": "NASDAQ",
            "finnhubIndustry": "Semiconductors",
        }

        mock_get.return_value = mock_response_profile

        profile = await self.client.get_company_profile("NVDA")
        self.assertEqual(profile["name"], "NVIDIA Corp")
        self.assertEqual(profile["ticker"], "NVDA")

    @patch("httpx.AsyncClient.get")
    async def test_auth_error_401(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        with self.assertRaises(FinnhubAuthError):
            await self.client.get_company_profile("NVDA")

    @patch("httpx.AsyncClient.get")
    async def test_rate_limit_429(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        with self.assertRaises(FinnhubRateLimitError):
            await self.client.get_quote("NVDA")

    @patch("httpx.AsyncClient.get")
    async def test_timeout_error(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        with self.assertRaises(FinnhubTimeoutError):
            await self.client.get_company_news("NVDA", "2026-08-01", "2026-08-08")

    @patch("httpx.AsyncClient.get")
    async def test_api_server_error_500(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(FinnhubAPIError):
            await self.client.get_company_profile("NVDA")


if __name__ == "__main__":
    unittest.main()
