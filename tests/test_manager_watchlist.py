"""Tests for watchlist routing through the Conversation Manager."""

import unittest
from unittest.mock import patch

from app.agent.manager import process_message
from app.memory.conversation_memory import conversation_memory
from app.services.watchlist_service import watchlist_service


class TestWatchlistRouting(unittest.IsolatedAsyncioTestCase):

    TEST_SESSION = "watchlist_route_user_001"

    def setUp(self):
        watchlist_service.clear_watchlist(self.TEST_SESSION)
        conversation_memory.clear_history(self.TEST_SESSION)

    def tearDown(self):
        watchlist_service.clear_watchlist(self.TEST_SESSION)
        conversation_memory.clear_history(self.TEST_SESSION)

    async def test_watchlist_add_routes_and_persists(self):
        response = await process_message(
            "Track Nvidia", session_id=self.TEST_SESSION
        )
        self.assertIn("NVDA", response)
        items = watchlist_service.get_watchlist(self.TEST_SESSION)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["symbol"], "NVDA")

    async def test_watchlist_add_duplicate_response(self):
        await process_message("Track Nvidia", session_id=self.TEST_SESSION)
        response = await process_message(
            "Add NVDA to my watchlist", session_id=self.TEST_SESSION
        )
        self.assertIn("already tracking", response.lower())
        items = watchlist_service.get_watchlist(self.TEST_SESSION)
        self.assertEqual(len(items), 1)

    async def test_watchlist_remove_routes(self):
        await process_message("Track Tesla", session_id=self.TEST_SESSION)
        response = await process_message(
            "Stop tracking Tesla", session_id=self.TEST_SESSION
        )
        self.assertIn("removed", response.lower())
        self.assertEqual(watchlist_service.get_watchlist(self.TEST_SESSION), [])

    async def test_watchlist_remove_missing_is_graceful(self):
        response = await process_message(
            "Stop tracking NVDA", session_id=self.TEST_SESSION
        )
        self.assertIn("isn't on your watchlist", response.lower())

    async def test_watchlist_list_empty(self):
        response = await process_message(
            "Show my watchlist", session_id=self.TEST_SESSION
        )
        self.assertIn("empty", response.lower())

    async def test_watchlist_list_populated(self):
        await process_message("Track Nvidia", session_id=self.TEST_SESSION)
        await process_message("Track Tesla", session_id=self.TEST_SESSION)
        response = await process_message(
            "What stocks am I following?", session_id=self.TEST_SESSION
        )
        self.assertIn("NVDA", response)
        self.assertIn("TSLA", response)

    async def test_watchlist_unknown_company_clarification(self):
        # "Add X to my watchlist" fires the strong add pattern even when the
        # company can't be resolved, so the manager should reply with a
        # clarification rather than guess.
        response = await process_message(
            "Add SUPERUNKNOWN_XYZ to my watchlist",
            session_id=self.TEST_SESSION,
        )
        self.assertIn("which company", response.lower())
        self.assertEqual(watchlist_service.get_watchlist(self.TEST_SESSION), [])

    async def test_user_isolation_through_manager(self):
        # Ensure clean history for both sessions.
        conversation_memory.clear_history("user_A")
        conversation_memory.clear_history("user_B")

        await process_message("Track Nvidia", session_id="user_A")
        response_b = await process_message(
            "Show my watchlist", session_id="user_B"
        )
        self.assertIn("empty", response_b.lower())
        # Cleanup
        watchlist_service.clear_watchlist("user_A")
        watchlist_service.clear_watchlist("user_B")
        conversation_memory.clear_history("user_A")
        conversation_memory.clear_history("user_B")


if __name__ == "__main__":
    unittest.main()