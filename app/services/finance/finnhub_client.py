import logging
from typing import Dict, Any, List, Optional
import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class FinnhubException(Exception):
    """Base exception for Finnhub API interactions."""
    pass


class FinnhubAuthError(FinnhubException):
    """Raised when authentication fails or API key is missing/invalid."""
    pass


class FinnhubRateLimitError(FinnhubException):
    """Raised when rate limit (429) is exceeded."""
    pass


class FinnhubTimeoutError(FinnhubException):
    """Raised when API request times out."""
    pass


class FinnhubNotFoundError(FinnhubException):
    """Raised when requested company or ticker symbol is not found."""
    pass


class FinnhubAPIError(FinnhubException):
    """Raised when general API error occurs (5xx or unexpected 4xx)."""
    pass


class FinnhubClient:
    """
    Low-level REST API Client for Finnhub financial market data.
    Handles HTTP communication, authentication, timeouts, and error code normalization.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout

    def _get_api_key(self) -> str:
        """Returns the configured API key or raises FinnhubAuthError."""
        key = settings.FINNHUB_API_KEY if self.api_key is None else self.api_key
        if not key:
            logger.error("Finnhub API key is not configured in settings.")
            raise FinnhubAuthError("Finnhub API key is missing. Please set FINNHUB_API_KEY in environment.")
        return key

    async def _request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """
        Executes HTTP GET request against Finnhub API with error handling.
        """
        api_key = self._get_api_key()
        request_params = {**params, "token": api_key}
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=request_params)
            except httpx.TimeoutException as e:
                logger.error(f"Finnhub API request timeout to {endpoint}: {e}")
                raise FinnhubTimeoutError(f"Request to Finnhub timed out after {self.timeout}s.") from e
            except httpx.RequestError as e:
                logger.error(f"Finnhub API HTTP network error to {endpoint}: {e}")
                raise FinnhubAPIError(f"Network error connecting to Finnhub: {e}") from e

            status = response.status_code

            if status == 200:
                try:
                    return response.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response from Finnhub: {e}")
                    raise FinnhubAPIError("Malformed JSON response from Finnhub.") from e
            elif status in (401, 403):
                logger.error(f"Finnhub API authentication failure (HTTP {status}).")
                raise FinnhubAuthError("Invalid or unauthorized Finnhub API Key.")
            elif status == 429:
                logger.warning(f"Finnhub API rate limit hit (HTTP 429).")
                raise FinnhubRateLimitError("Finnhub API rate limit reached. Please try again in a minute.")
            elif status == 404:
                logger.warning(f"Finnhub resource not found (HTTP 404) for {endpoint}.")
                raise FinnhubNotFoundError(f"Requested symbol or resource not found on Finnhub.")
            else:
                logger.error(f"Finnhub API error HTTP {status}: {response.text[:200]}")
                raise FinnhubAPIError(f"Finnhub API returned status code {status}.")

    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches company profile (Profile2) for a ticker symbol.
        Returns dict containing name, exchange, industry, country, weburl, etc.
        """
        logger.info(f"Fetching Finnhub profile for symbol: '{symbol}'")
        data = await self._request("stock/profile2", {"symbol": symbol.upper()})
        return data if isinstance(data, dict) else {}

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches current stock quote (c, d, dp, h, l, o, pc, t) for a ticker symbol.
        """
        logger.info(f"Fetching Finnhub quote for symbol: '{symbol}'")
        data = await self._request("quote", {"symbol": symbol.upper()})
        return data if isinstance(data, dict) else {}

    async def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """
        Fetches company news for a ticker symbol between from_date and to_date (YYYY-MM-DD).
        """
        logger.info(f"Fetching Finnhub news for symbol: '{symbol}' ({from_date} to {to_date})")
        data = await self._request("company-news", {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date,
        })
        return data if isinstance(data, list) else []
