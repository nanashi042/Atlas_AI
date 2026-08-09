import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from app.agent.manager import process_message
from app.services.finance.finnhub_client import FinnhubAuthError, FinnhubRateLimitError, FinnhubNotFoundError
from app.services.finance.models import CompanyResearchResult


class TestConversationManager(unittest.IsolatedAsyncioTestCase):

    async def test_greeting_routing(self):
        response = await process_message("hello", session_id="test_session_1")
        self.assertIn("Hello! I'm Atlas AI", response)

    @patch("app.agent.manager.CompanyResearchService")
    @patch("app.agent.manager.research_company_ai")
    async def test_company_research_routing_success(self, mock_ai_synth, mock_service_cls):
        mock_service_instance = AsyncMock()
        mock_service_cls.return_value = mock_service_instance

        dummy_result = CompanyResearchResult(
            symbol="NVDA",
            company_name="NVIDIA Corp",
            current_price=125.00,
        )
        mock_service_instance.get_company_research.return_value = dummy_result
        mock_ai_synth.return_value = "NVIDIA Corp (NVDA)\nPrice: $125.00\nAnalyst Summary: Leading AI hardware."

        response = await process_message("Tell me about Nvidia", session_id="test_session_2")

        mock_service_instance.get_company_research.assert_called_once_with("NVDA")
        mock_ai_synth.assert_called_once()
        self.assertIn("NVIDIA Corp", response)

    async def test_unknown_company_clarification(self):
        with patch("app.agent.manager.resolve_company_ticker", return_value=None):
            response = await process_message("Tell me about UNKNOWN_XYZ_COMP", session_id="test_session_3")
            self.assertEqual(
                response,
                "Which company do you mean? Please provide the company name or ticker symbol."
            )

    @patch("app.agent.manager.CompanyResearchService")
    async def test_finnhub_auth_error_handling(self, mock_service_cls):
        mock_service_instance = AsyncMock()
        mock_service_cls.return_value = mock_service_instance
        mock_service_instance.get_company_research.side_effect = FinnhubAuthError("Missing API key")

        response = await process_message("What is NVDA?", session_id="test_session_4")

        self.assertIn("Finnhub API Key is missing or invalid", response)

    @patch("app.agent.manager.CompanyResearchService")
    async def test_finnhub_rate_limit_handling(self, mock_service_cls):
        mock_service_instance = AsyncMock()
        mock_service_cls.return_value = mock_service_instance
        mock_service_instance.get_company_research.side_effect = FinnhubRateLimitError("Rate limit")

        response = await process_message("What is Tesla?", session_id="test_session_5")

        self.assertIn("rate limit reached", response)

    @patch("app.agent.manager.CompanyResearchService")
    async def test_company_not_found_handling(self, mock_service_cls):
        mock_service_instance = AsyncMock()
        mock_service_cls.return_value = mock_service_instance
        mock_service_instance.get_company_research.side_effect = FinnhubNotFoundError("Not found")

        response = await process_message("Tell me about NVDA", session_id="test_session_6")

        self.assertIn("Could not find company or market data", response)

    @patch("app.agent.manager.chat")
    async def test_general_chat_routing(self, mock_chat):
        mock_chat.return_value = "Inflation is the rate at which prices increase."
        response = await process_message("What is inflation?", session_id="test_session_7")

        mock_chat.assert_called_once_with("What is inflation?", session_id="test_session_7")
        self.assertEqual(response, "Inflation is the rate at which prices increase.")

    async def test_clear_command_does_not_touch_watchlist(self):
        """
        Conversation /clear must clear chat memory only. The watchlist is a
        persistent user preference and must survive a /clear.
        """
        from app.services.watchlist_service import watchlist_service
        from app.memory.conversation_memory import conversation_memory

        session = "clear_command_isolation_session"
        watchlist_service.clear_watchlist(session)
        conversation_memory.clear_history(session)

        await process_message("Track Nvidia", session_id=session)
        self.assertEqual(len(watchlist_service.get_watchlist(session)), 1)

        # Simulate what /clear does in the handler.
        conversation_memory.clear_history(session)

        items = watchlist_service.get_watchlist(session)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["symbol"], "NVDA")

        # Cleanup.
        watchlist_service.clear_watchlist(session)


if __name__ == "__main__":
    unittest.main()
