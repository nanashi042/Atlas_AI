"""Tests for BriefingService.

All external dependencies (Finnhub, Gemini, WatchlistService) are
mocked. No live API calls. No live DB writes against the real atlas.db
— we use a dedicated in-memory SQLite session via dependency injection
of a fake watchlist service.
"""

import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from app.services.briefing_service import (
    BriefingData,
    BriefingItem,
    BriefingService,
    _is_news_relevant,
    _build_fallback_text,
    _build_briefing_prompt,
    _format_briefing_context,
    EMPTY_WATCHLIST_MESSAGE,
    ALL_FINANCE_FAILURE_MESSAGE,
    WATCHLIST_LOAD_ERROR_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
)
from app.services.finance.finnhub_client import (
    FinnhubAPIError,
    FinnhubAuthError,
    FinnhubNotFoundError,
    FinnhubRateLimitError,
    FinnhubTimeoutError,
)
from app.services.finance.models import (
    CompanyNewsItem,
    CompanyResearchResult,
)
from app.services.watchlist_service import WatchlistError


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeWatchlistService:
    """In-memory replacement for WatchlistService so tests don't touch
    the real atlas.db."""

    def __init__(self, entries_by_user: Optional[Dict[str, List[Dict[str, str]]]] = None,
                 fail: bool = False):
        self.entries_by_user = entries_by_user or {}
        self.fail = fail

    def get_watchlist(self, user_id: str) -> List[Dict[str, Any]]:
        if self.fail:
            raise WatchlistError("Simulated DB failure")
        return list(self.entries_by_user.get(user_id, []))


def _make_research_result(
    symbol: str,
    company_name: str,
    current_price: Optional[float] = 100.0,
    price_change: Optional[float] = 1.5,
    percent_change: Optional[float] = 1.5,
    previous_close: Optional[float] = 98.5,
    day_high: Optional[float] = 101.0,
    day_low: Optional[float] = 97.0,
    news: Optional[List[CompanyNewsItem]] = None,
) -> CompanyResearchResult:
    return CompanyResearchResult(
        symbol=symbol,
        company_name=company_name,
        current_price=current_price,
        price_change=price_change,
        percent_change=percent_change,
        previous_close=previous_close,
        day_high=day_high,
        day_low=day_low,
        recent_news=news or [],
    )


def _relevant_news(symbol: str, company_name: str) -> CompanyNewsItem:
    return CompanyNewsItem(
        headline=f"{company_name} announces new product",
        summary=f"{symbol} shares react to company-specific news.",
        source="Reuters",
        url="https://example.com",
        published_at="2026-08-08",
    )


def _unrelated_news() -> CompanyNewsItem:
    return CompanyNewsItem(
        headline="Tesla unveils new model",
        summary="TSLA shares climb on strong delivery numbers.",
        source="CNBC",
        url="https://example.com",
        published_at="2026-08-08",
    )


def _make_service(
    watchlist: FakeWatchlistService,
    research_results_by_symbol: Dict[str, CompanyResearchResult],
    research_errors_by_symbol: Optional[Dict[str, Exception]] = None,
    llm_response: str = "LLM synthesized briefing text.",
) -> tuple:
    """Build a BriefingService with fully-injected dependencies and return
    (service, mock_research, captured_llm_prompts)."""
    mock_research = MagicMock()
    async def fake_research(symbol: str) -> CompanyResearchResult:
        if research_errors_by_symbol and symbol in research_errors_by_symbol:
            raise research_errors_by_symbol[symbol]
        if symbol not in research_results_by_symbol:
            raise FinnhubNotFoundError(f"Unknown {symbol}")
        return research_results_by_symbol[symbol]
    mock_research.get_company_research.side_effect = fake_research

    captured: List[str] = []

    async def fake_llm(prompt: str, history=None) -> str:
        captured.append(prompt)
        return llm_response

    service = BriefingService(
        research_service=mock_research,
        llm_generate=fake_llm,
        watchlist=watchlist,
    )
    return service, mock_research, captured


# ---------------------------------------------------------------------------
# News relevance
# ---------------------------------------------------------------------------


class TestNewsRelevance(unittest.TestCase):
    """Ensure the news filter keeps company-specific items and drops
    unrelated articles that Finnhub sometimes returns."""

    def test_relevant_news_is_kept(self):
        news = _relevant_news("NVDA", "NVIDIA")
        self.assertTrue(_is_news_relevant(news, "NVDA", "NVIDIA"))

    def test_unrelated_news_is_dropped(self):
        news = _unrelated_news()
        self.assertFalse(_is_news_relevant(news, "NVDA", "NVIDIA"))

    def test_company_name_variants_are_accepted(self):
        # "NVIDIA Corp" → core variant "nvidia" should match.
        news = CompanyNewsItem(
            headline="NVIDIA reports record earnings",
            summary="",
            source="",
            url="",
            published_at="",
        )
        self.assertTrue(_is_news_relevant(news, "NVDA", "NVIDIA Corp"))

    def test_empty_news_is_dropped(self):
        news = CompanyNewsItem(headline="", summary="", source="", url="", published_at="")
        self.assertFalse(_is_news_relevant(news, "NVDA", "NVIDIA"))


# ---------------------------------------------------------------------------
# Empty watchlist
# ---------------------------------------------------------------------------


class TestEmptyWatchlist(unittest.IsolatedAsyncioTestCase):
    async def test_empty_watchlist_returns_friendly_message(self):
        watchlist = FakeWatchlistService(entries_by_user={"u1": []})
        service, mock_research, captured_prompts = _make_service(
            watchlist,
            research_results_by_symbol={},
            llm_response="SHOULD NOT BE CALLED",
        )
        result = await service.generate_briefing("u1")
        self.assertEqual(result, EMPTY_WATCHLIST_MESSAGE)
        # No Finnhub calls, no Gemini calls.
        mock_research.get_company_research.assert_not_called()
        self.assertEqual(captured_prompts, [])


# ---------------------------------------------------------------------------
# One-company / multi-company watchlists
# ---------------------------------------------------------------------------


class TestSingleCompanyWatchlist(unittest.IsolatedAsyncioTestCase):
    async def test_single_company_calls_finance_once_and_llm_once(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        research_results = {
            "NVDA": _make_research_result(
                "NVDA", "NVIDIA",
                current_price=125.0, price_change=2.5, percent_change=2.04,
                news=[_relevant_news("NVDA", "NVIDIA")],
            )
        }
        service, mock_research, captured_prompts = _make_service(
            watchlist, research_results, llm_response="Briefing for NVDA only."
        )
        result = await service.generate_briefing("u1")
        self.assertEqual(result, "Briefing for NVDA only.")
        mock_research.get_company_research.assert_called_once_with("NVDA")
        self.assertEqual(len(captured_prompts), 1)


class TestMultipleCompanyWatchlist(unittest.IsolatedAsyncioTestCase):
    async def test_three_companies_three_finance_calls_one_llm_call(self):
        watchlist = FakeWatchlistService(
            entries_by_user={
                "u1": [
                    {"symbol": "NVDA", "company_name": "NVIDIA"},
                    {"symbol": "TSLA", "company_name": "Tesla"},
                    {"symbol": "AMD", "company_name": "AMD"},
                ]
            }
        )
        research_results = {
            "NVDA": _make_research_result("NVDA", "NVIDIA",
                                          news=[_relevant_news("NVDA", "NVIDIA")]),
            "TSLA": _make_research_result("TSLA", "Tesla",
                                          news=[_relevant_news("TSLA", "Tesla")]),
            "AMD": _make_research_result("AMD", "AMD",
                                         news=[_relevant_news("AMD", "AMD")]),
        }
        service, mock_research, captured_prompts = _make_service(
            watchlist, research_results, llm_response="Multi briefing."
        )
        result = await service.generate_briefing("u1")
        self.assertEqual(result, "Multi briefing.")
        self.assertEqual(mock_research.get_company_research.call_count, 3)
        self.assertEqual(len(captured_prompts), 1)

    async def test_user_isolation_only_uses_target_users_watchlist(self):
        watchlist = FakeWatchlistService(
            entries_by_user={
                "user_a": [{"symbol": "NVDA", "company_name": "NVIDIA"}],
                "user_b": [{"symbol": "TSLA", "company_name": "Tesla"}],
            }
        )
        research_results = {
            "NVDA": _make_research_result("NVDA", "NVIDIA"),
        }
        service, mock_research, _ = _make_service(
            watchlist, research_results
        )
        await service.generate_briefing("user_a")
        # Only NVDA was looked up — TSLA was user_b's, never requested.
        mock_research.get_company_research.assert_called_once_with("NVDA")


# ---------------------------------------------------------------------------
# Successful retrieval populates BriefingItem
# ---------------------------------------------------------------------------


class TestSuccessfulRetrieval(unittest.IsolatedAsyncioTestCase):
    async def test_briefing_item_populated_with_normalized_fields(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        result = _make_research_result(
            "NVDA", "NVIDIA",
            current_price=125.0, price_change=2.5, percent_change=2.04,
            previous_close=122.5, day_high=126.0, day_low=121.0,
            news=[_relevant_news("NVDA", "NVIDIA")],
        )
        service, _, captured_prompts = _make_service(
            watchlist,
            {"NVDA": result},
            llm_response="Briefing OK",
        )
        await service.generate_briefing("u1")
        # Verify the LLM prompt contains normalized fields.
        prompt = captured_prompts[0]
        self.assertIn("NVDA", prompt)
        self.assertIn("$125.00", prompt)
        self.assertIn("+2.04%", prompt)
        self.assertIn("Previous Close", prompt)
        self.assertIn("Day Range", prompt)
        self.assertIn("NVIDIA announces new product", prompt)


# ---------------------------------------------------------------------------
# Partial and complete finance failure
# ---------------------------------------------------------------------------


class TestPartialFinanceFailure(unittest.IsolatedAsyncioTestCase):
    async def test_one_company_failure_does_not_break_briefing(self):
        watchlist = FakeWatchlistService(
            entries_by_user={
                "u1": [
                    {"symbol": "NVDA", "company_name": "NVIDIA"},
                    {"symbol": "TSLA", "company_name": "Tesla"},
                ]
            }
        )
        research_results = {
            "NVDA": _make_research_result("NVDA", "NVIDIA",
                                          news=[_relevant_news("NVDA", "NVIDIA")]),
        }
        research_errors = {"TSLA": FinnhubRateLimitError("Rate limit")}
        service, mock_research, captured_prompts = _make_service(
            watchlist, research_results,
            research_errors_by_symbol=research_errors,
            llm_response="Briefing for NVDA, TSLA unavailable.",
        )
        result = await service.generate_briefing("u1")
        # Briefing still produced.
        self.assertEqual(result, "Briefing for NVDA, TSLA unavailable.")
        self.assertEqual(mock_research.get_company_research.call_count, 2)
        # Prompt must mention the unavailable ticker.
        self.assertIn("TSLA", captured_prompts[0])
        self.assertIn("UNAVAILABLE", captured_prompts[0])

    async def test_auth_error_treated_as_per_company_failure(self):
        watchlist = FakeWatchlistService(
            entries_by_user={
                "u1": [
                    {"symbol": "NVDA", "company_name": "NVIDIA"},
                    {"symbol": "TSLA", "company_name": "Tesla"},
                ]
            }
        )
        research_results = {
            "NVDA": _make_research_result("NVDA", "NVIDIA"),
        }
        research_errors = {"TSLA": FinnhubAuthError("Auth fail")}
        service, _, captured_prompts = _make_service(
            watchlist, research_results,
            research_errors_by_symbol=research_errors,
            llm_response="Briefing produced.",
        )
        result = await service.generate_briefing("u1")
        self.assertEqual(result, "Briefing produced.")
        self.assertIn("TSLA", captured_prompts[0])


class TestCompleteFinanceFailure(unittest.IsolatedAsyncioTestCase):
    async def test_all_companies_fail_returns_friendly_message(self):
        watchlist = FakeWatchlistService(
            entries_by_user={
                "u1": [
                    {"symbol": "NVDA", "company_name": "NVIDIA"},
                    {"symbol": "TSLA", "company_name": "Tesla"},
                ]
            }
        )
        research_errors = {
            "NVDA": FinnhubAPIError("API down"),
            "TSLA": FinnhubTimeoutError("Timeout"),
        }
        service, _, captured_prompts = _make_service(
            watchlist, research_results_by_symbol={},
            research_errors_by_symbol=research_errors,
            llm_response="SHOULD NOT BE CALLED",
        )
        result = await service.generate_briefing("u1")
        self.assertEqual(result, ALL_FINANCE_FAILURE_MESSAGE)
        # Gemini must NOT be called when every company failed.
        self.assertEqual(captured_prompts, [])


# ---------------------------------------------------------------------------
# Gemini receives normalized data, not raw Finnhub JSON
# ---------------------------------------------------------------------------


class TestNormalizedDataToLLM(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_does_not_contain_raw_finnhub_keys(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        result = _make_research_result(
            "NVDA", "NVIDIA",
            news=[_relevant_news("NVDA", "NVIDIA")],
        )
        service, _, captured_prompts = _make_service(
            watchlist, {"NVDA": result}
        )
        await service.generate_briefing("u1")
        prompt = captured_prompts[0]
        # Forbidden: raw Finnhub keys
        forbidden = ["datetime", "weburl", "finnhubIndustry", "exchange", "currency"]
        for token in forbidden:
            self.assertNotIn(token, prompt,
                             f"Raw Finnhub key '{token}' leaked into LLM prompt")
        # Required: normalized labels
        required = ["Current Price", "Day Range", "Previous Close", "Relevant News"]
        for token in required:
            self.assertIn(token, prompt,
                          f"Normalized label '{token}' missing from prompt")
        # Timestamp present
        self.assertIn("Generated at:", prompt)

    async def test_prompt_is_not_the_raw_news_object(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        result = _make_research_result(
            "NVDA", "NVIDIA",
            news=[_relevant_news("NVDA", "NVIDIA")],
        )
        service, _, captured_prompts = _make_service(
            watchlist, {"NVDA": result}
        )
        await service.generate_briefing("u1")
        prompt = captured_prompts[0]
        # The repr of a CompanyResearchResult would include dataclass-looking
        # markers. We just verify we don't have raw list/object reprs.
        self.assertNotIn("CompanyResearchResult(", prompt)
        self.assertNotIn("CompanyNewsItem(", prompt)


# ---------------------------------------------------------------------------
# News relevance inside service
# ---------------------------------------------------------------------------


class TestServiceFiltersUnrelatedNews(unittest.IsolatedAsyncioTestCase):
    async def test_unrelated_news_dropped_relevant_news_kept(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        result = _make_research_result(
            "NVDA", "NVIDIA",
            news=[
                _relevant_news("NVDA", "NVIDIA"),
                _unrelated_news(),  # Tesla article — should be filtered out
                CompanyNewsItem(
                    headline="Micron shares rise on memory demand",
                    summary="Micron reports strong quarter.",
                    source="WSJ", url="", published_at="2026-08-08",
                ),
            ],
        )
        service, _, captured_prompts = _make_service(
            watchlist, {"NVDA": result}
        )
        await service.generate_briefing("u1")
        prompt = captured_prompts[0]
        # NVDA-relevant news kept
        self.assertIn("NVIDIA announces new product", prompt)
        # Unrelated news dropped
        self.assertNotIn("Tesla unveils new model", prompt)
        self.assertNotIn("Micron shares rise", prompt)


# ---------------------------------------------------------------------------
# Concise output (fallback path)
# ---------------------------------------------------------------------------


class TestFallbackIsConcise(unittest.TestCase):
    def test_fallback_text_under_reasonable_budget_for_three_companies(self):
        briefing = BriefingData(
            user_id="u1",
            generated_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            items=[
                BriefingItem(
                    symbol="NVDA", company_name="NVIDIA",
                    current_price=123.45, price_change=2.5, percent_change=2.07,
                    relevant_news=[_relevant_news("NVDA", "NVIDIA")],
                ),
                BriefingItem(
                    symbol="TSLA", company_name="Tesla",
                    current_price=222.0, price_change=-2.0, percent_change=-0.89,
                    relevant_news=[_relevant_news("TSLA", "Tesla")],
                ),
                BriefingItem(
                    symbol="AMD", company_name="AMD",
                    current_price=145.0, price_change=1.2, percent_change=0.83,
                    relevant_news=[_relevant_news("AMD", "AMD")],
                ),
            ],
        )
        text = _build_fallback_text(briefing)
        self.assertLess(len(text), 1500,
                        f"Fallback too long ({len(text)} chars):\n{text}")
        self.assertIn("Source: Finnhub", text)


# ---------------------------------------------------------------------------
# Gemini error → fallback path
# ---------------------------------------------------------------------------


class TestGeminiFailureFallsBack(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_error_returns_deterministic_template(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        result = _make_research_result(
            "NVDA", "NVIDIA",
            current_price=120.0, price_change=1.0, percent_change=0.84,
            news=[_relevant_news("NVDA", "NVIDIA")],
        )
        service, _, _ = _make_service(
            watchlist, {"NVDA": result},
            llm_response="[Error 503] Gemini service unavailable",
        )
        text = await service.generate_briefing("u1")
        # Fallback template must contain the price line and source line.
        self.assertIn("NVDA", text)
        self.assertIn("$120.00", text)
        self.assertIn("Source: Finnhub", text)

    async def test_unexpected_llm_exception_uses_fallback(self):
        watchlist = FakeWatchlistService(
            entries_by_user={"u1": [{"symbol": "NVDA", "company_name": "NVIDIA"}]}
        )
        result = _make_research_result("NVDA", "NVIDIA")
        mock_research = MagicMock()
        mock_research.get_company_research = AsyncMock(return_value=result)

        async def boom(prompt, history=None):
            raise RuntimeError("kaboom")
        service = BriefingService(
            research_service=mock_research,
            llm_generate=boom,
            watchlist=watchlist,
        )
        text = await service.generate_briefing("u1")
        self.assertIn("NVDA", text)
        self.assertIn("Source: Finnhub", text)


# ---------------------------------------------------------------------------
# Watchlist DB error
# ---------------------------------------------------------------------------


class TestWatchlistLoadError(unittest.IsolatedAsyncioTestCase):
    async def test_watchlist_db_error_returns_friendly_message(self):
        watchlist = FakeWatchlistService(fail=True)
        service, _, captured_prompts = _make_service(
            watchlist, research_results_by_symbol={},
            llm_response="SHOULD NOT BE CALLED",
        )
        result = await service.generate_briefing("u1")
        self.assertEqual(result, WATCHLIST_LOAD_ERROR_MESSAGE)
        self.assertEqual(captured_prompts, [])


# ---------------------------------------------------------------------------
# Empty user_id safety
# ---------------------------------------------------------------------------


class TestEmptyUserId(unittest.IsolatedAsyncioTestCase):
    async def test_empty_user_id_returns_empty_watchlist_message(self):
        watchlist = FakeWatchlistService()
        service, _, _ = _make_service(watchlist, {})
        result = await service.generate_briefing("")
        self.assertEqual(result, EMPTY_WATCHLIST_MESSAGE)


if __name__ == "__main__":
    unittest.main()
