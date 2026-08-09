"""
Tests for the persistent Watchlist service.

These tests use the same SQLite DB as the running app. They isolate state
by writing and removing only the test session IDs they create, and assert
explicit cleanup before exit so they don't pollute other tests.
"""

import unittest

from app.services.watchlist_service import watchlist_service, WatchlistError


class TestWatchlistService(unittest.TestCase):

    TEST_USER = "watchlist_test_user_001"
    OTHER_USER = "watchlist_test_user_002"

    def setUp(self):
        # Make sure each test starts from a clean slate.
        watchlist_service.clear_watchlist(self.TEST_USER)
        watchlist_service.clear_watchlist(self.OTHER_USER)

    def tearDown(self):
        watchlist_service.clear_watchlist(self.TEST_USER)
        watchlist_service.clear_watchlist(self.OTHER_USER)

    def test_add_company(self):
        added, msg = watchlist_service.add_to_watchlist(
            self.TEST_USER, "nvda", "NVIDIA"
        )
        self.assertTrue(added)
        self.assertIn("NVDA", msg)
        self.assertIn("NVIDIA", msg)

        items = watchlist_service.get_watchlist(self.TEST_USER)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["symbol"], "NVDA")
        self.assertEqual(items[0]["company_name"], "NVIDIA")

    def test_add_duplicate_blocked(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "NVDA", "NVIDIA")
        added, msg = watchlist_service.add_to_watchlist(
            self.TEST_USER, "NVDA", "NVIDIA"
        )
        self.assertFalse(added)
        self.assertIn("already tracking", msg.lower())

        items = watchlist_service.get_watchlist(self.TEST_USER)
        self.assertEqual(len(items), 1)

    def test_symbol_normalized_to_uppercase(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "tsla", "Tesla")
        items = watchlist_service.get_watchlist(self.TEST_USER)
        self.assertEqual(items[0]["symbol"], "TSLA")
        self.assertTrue(
            watchlist_service.is_in_watchlist(self.TEST_USER, "tsla")
        )

    def test_remove_company(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "AAPL", "Apple")
        removed, msg = watchlist_service.remove_from_watchlist(
            self.TEST_USER, "AAPL"
        )
        self.assertTrue(removed)
        self.assertIn("AAPL", msg)
        self.assertEqual(watchlist_service.get_watchlist(self.TEST_USER), [])

    def test_remove_missing_company_is_graceful(self):
        removed, msg = watchlist_service.remove_from_watchlist(
            self.TEST_USER, "NVDA"
        )
        self.assertFalse(removed)
        self.assertIn("isn't on your watchlist", msg.lower())

    def test_list_empty(self):
        items = watchlist_service.get_watchlist(self.TEST_USER)
        self.assertEqual(items, [])

    def test_list_multiple(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "NVDA", "NVIDIA")
        watchlist_service.add_to_watchlist(self.TEST_USER, "TSLA", "Tesla")
        items = watchlist_service.get_watchlist(self.TEST_USER)
        symbols = {item["symbol"] for item in items}
        self.assertEqual(symbols, {"NVDA", "TSLA"})

    def test_user_isolation(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "MSFT", "Microsoft")

        # Other user should see nothing.
        other_items = watchlist_service.get_watchlist(self.OTHER_USER)
        self.assertEqual(other_items, [])

        # Original user still sees it.
        items = watchlist_service.get_watchlist(self.TEST_USER)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["symbol"], "MSFT")

        # Removal by another user must not affect original.
        removed, _ = watchlist_service.remove_from_watchlist(
            self.OTHER_USER, "MSFT"
        )
        self.assertFalse(removed)
        self.assertEqual(len(watchlist_service.get_watchlist(self.TEST_USER)), 1)

    def test_is_in_watchlist(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "GOOGL", "Alphabet")
        self.assertTrue(
            watchlist_service.is_in_watchlist(self.TEST_USER, "GOOGL")
        )
        self.assertTrue(
            watchlist_service.is_in_watchlist(self.TEST_USER, "googl")
        )
        self.assertFalse(
            watchlist_service.is_in_watchlist(self.TEST_USER, "AMZN")
        )

    def test_clear_watchlist_helper(self):
        watchlist_service.add_to_watchlist(self.TEST_USER, "NVDA", "NVIDIA")
        watchlist_service.add_to_watchlist(self.TEST_USER, "TSLA", "Tesla")
        deleted = watchlist_service.clear_watchlist(self.TEST_USER)
        self.assertEqual(deleted, 2)
        self.assertEqual(watchlist_service.get_watchlist(self.TEST_USER), [])

    def test_invalid_input_does_not_crash(self):
        added, msg = watchlist_service.add_to_watchlist(
            "", "NVDA", "NVIDIA"
        )
        self.assertFalse(added)
        added, msg = watchlist_service.add_to_watchlist(
            self.TEST_USER, "", "NVIDIA"
        )
        self.assertFalse(added)
        added, msg = watchlist_service.add_to_watchlist(
            self.TEST_USER, "NVDA", ""
        )
        self.assertFalse(added)

    def test_resolve_company_for_user_uses_resolver(self):
        resolved = watchlist_service.resolve_company_for_user("Track Nvidia")
        self.assertIsNotNone(resolved)
        symbol, name = resolved
        self.assertEqual(symbol, "NVDA")
        self.assertEqual(name, "NVIDIA")

        resolved = watchlist_service.resolve_company_for_user(
            "Add Tesla to my watchlist"
        )
        self.assertIsNotNone(resolved)
        symbol, name = resolved
        self.assertEqual(symbol, "TSLA")

        # Unknown company should return None.
        self.assertIsNone(
            watchlist_service.resolve_company_for_user(
                "Track UNKNOWN_ZZZ_COMPANY"
            )
        )


if __name__ == "__main__":
    unittest.main()