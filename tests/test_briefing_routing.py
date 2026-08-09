"""Tests for BRIEFING routing through the Conversation Manager.

These verify that `process_message` correctly dispatches natural
briefing requests to `BriefingService` and that existing flows
(COMPANY_RESEARCH, WATCHLIST_*, GENERAL_CHAT) remain unaffected.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.intent import Intent
from app.memory.conversation_memory import conversation_memory
from app.services.briefing_service import EMPTY_WATCHLIST_MESSAGE
from app.services.finance.finnhub_client import FinnhubAuthError
from app.services.finance.models import CompanyResearchResult


class TestBriefingRouting(unittest.IsolatedAsyncioTestCase):
    """End-to-end routing tests for BRIEFING intent through the manager."""

    TEST_SESSION = "briefing_route_user_001"
    OTHER_SESSION = "briefing_route_user_002"

    def setUp(self):
        # Clean state for our test users.
        from app.services.watchlist_service import watchlist_service
        watchlist_service.clear_watchlist(self.TEST_SESSION)
        watchlist_service.clear_watchlist(self.OTHER_SESSION)
        conversation_memory.clear_history(self.TEST_SESSION)
        conversation_memory.clear_history(self.OTHER_SESSION)

    def tearDown(self):
        from app.services.watchlist_service import watchlist_service
        watchlist_service.clear_watchlist(self.TEST_SESSION)
        watchlist_service.clear_watchlist(self.OTHER_SESSION)
        conversation_memory.clear_history(self.TEST_SESSION)
        conversation_memory.clear_history(self.OTHER_SESSION)

    @patch("app.agent.manager.briefing_service")
    async def test_morning_briefing_invokes_service(self, mock_service):
        mock_service.generate_briefing = AsyncMock(
            return_value="📈 NVIDIA (NVDA) ...\nSource: Finnhub"
        )
        from app.agent.manager import process_message
        response = await process_message(
            "Give me my morning briefing", session_id=self.TEST_SESSION
        )
        mock_service.generate_briefing.assert_awaited_once()
        self.assertEqual(
            response, "📈 NVIDIA (NVDA) ...\nSource: Finnhub"
        )

    @patch("app.agent.manager.briefing_service")
    async def test_whats_happening_with_watchlist_routes_to_briefing(
        self, mock_service
    ):
        mock_service.generate_briefing = AsyncMock(return_value="Briefing.")
        from app.agent.manager import process_message
        response = await process_message(
            "What's happening with my watchlist?", session_id=self.TEST_SESSION
        )
        mock_service.generate_briefing.assert_awaited_once()
        self.assertEqual(response, "Briefing.")

    async def test_empty_watchlist_short_circuits_no_finance_calls(self):
        """When the watchlist is empty, the manager path must return the
        empty message WITHOUT invoking CompanyResearchService."""
        # Use a session with an empty watchlist.
        from app.agent.manager import process_message
        from app.services.watchlist_service import watchlist_service
        empty_session = "briefing_route_empty_user"
        watchlist_service.clear_watchlist(empty_session)
        conversation_memory.clear_history(empty_session)

        with patch(
            "app.services.finance.company_research.CompanyResearchService"
        ) as mock_service_cls:
            response = await process_message(
                "Give me my morning briefing", session_id=empty_session
            )
            # Empty watchlist must not call any finance research.
            mock_service_cls.return_value.get_company_research.assert_not_called()
        self.assertEqual(response, EMPTY_WATCHLIST_MESSAGE)

        # Cleanup
        watchlist_service.clear_watchlist(empty_session)
        conversation_memory.clear_history(empty_session)

    @patch("app.agent.manager.briefing_service")
    async def test_briefing_response_persisted_to_memory(self, mock_service):
        mock_service.generate_briefing = AsyncMock(return_value="My briefing text")
        from app.agent.manager import process_message
        await process_message(
            "Give me my morning briefing", session_id=self.TEST_SESSION
        )
        history = conversation_memory.get_history(self.TEST_SESSION)
        # Most recent message is the assistant's reply; user message came first.
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "model")
        self.assertEqual(history[1]["parts"][0]["text"], "My briefing text")

    # ---- Regression: existing flows still work ----

    @patch("app.agent.manager.research_company_ai")
    @patch("app.agent.manager.CompanyResearchService")
    async def test_company_research_still_works(self, mock_svc_cls, mock_ai):
        mock_svc = AsyncMock()
        mock_svc_cls.return_value = mock_svc
        mock_svc.get_company_research.return_value = CompanyResearchResult(
            symbol="NVDA", company_name="NVIDIA",
        )
        mock_ai.return_value = "NVIDIA Corp (NVDA) — briefing"
        from app.agent.manager import process_message
        response = await process_message(
            "Tell me about Nvidia", session_id=self.TEST_SESSION
        )
        self.assertIn("NVDA", response)
        mock_svc.get_company_research.assert_awaited_once_with("NVDA")

    async def test_watchlist_list_still_returns_list_not_briefing(self):
        from app.agent.manager import process_message
        from app.services.watchlist_service import watchlist_service
        # Add a ticker.
        watchlist_service.add_to_watchlist(self.TEST_SESSION, "NVDA", "NVIDIA")
        response = await process_message(
            "Show my watchlist", session_id=self.TEST_SESSION
        )
        self.assertIn("NVDA", response)
        self.assertIn("NVIDIA", response)
        # Should be a list-style message (no emoji ticker lines).
        self.assertNotIn("📈", response)

    @patch("app.agent.manager.chat")
    async def test_general_chat_still_works(self, mock_chat):
        mock_chat.return_value = "Inflation means prices rise."
        from app.agent.manager import process_message
        response = await process_message(
            "What is inflation?", session_id=self.TEST_SESSION
        )
        self.assertEqual(response, "Inflation means prices rise.")
        mock_chat.assert_awaited_once()


class TestBriefingIntentSurface(unittest.TestCase):
    """Light sanity check that the BRIEFING intent is what we expect
    for the canonical natural-language requests."""

    def test_briefing_intent_is_an_enum_member(self):
        self.assertTrue(hasattr(Intent, "BRIEFING"))
        self.assertEqual(Intent.BRIEFING.value, "briefing")

    def test_canonical_phrases_classify_as_briefing(self):
        from app.agent.intent import detect_intent
        for q in [
            "Give me my morning briefing",
            "What's happening with my watchlist?",
            "Give me today's market briefing",
            "Daily briefing",
        ]:
            with self.subTest(q=q):
                self.assertEqual(detect_intent(q), Intent.BRIEFING)


if __name__ == "__main__":
    unittest.main()
