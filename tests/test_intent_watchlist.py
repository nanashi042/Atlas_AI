"""Tests for watchlist-related intent detection."""

import unittest

from app.agent.intent import detect_intent, Intent


class TestWatchlistIntents(unittest.TestCase):

    def test_watchlist_add(self):
        queries = [
            "Track Nvidia",
            "Add Tesla to my watchlist",
            "I am interested in Microsoft",
            "Keep an eye on AMD",
            "Start tracking Apple",
            "Follow NVDA",
            "Put Netflix on my watchlist",
        ]
        for q in queries:
            self.assertEqual(
                detect_intent(q),
                Intent.WATCHLIST_ADD,
                f"Failed for: {q}",
            )

    def test_watchlist_remove(self):
        queries = [
            "Stop tracking Nvidia",
            "Remove Tesla from my watchlist",
            "Untrack Microsoft",
            "Stop following Apple",
            "Delete NVDA from my watchlist",
        ]
        for q in queries:
            self.assertEqual(
                detect_intent(q),
                Intent.WATCHLIST_REMOVE,
                f"Failed for: {q}",
            )

    def test_watchlist_list(self):
        queries = [
            "What stocks am I following?",
            "Show my watchlist",
            "List my watchlist",
            "What's on my watchlist?",
            "What companies am I tracking?",
            "My tracked stocks",
        ]
        for q in queries:
            self.assertEqual(
                detect_intent(q),
                Intent.WATCHLIST_LIST,
                f"Failed for: {q}",
            )

    def test_company_research_still_works(self):
        # "Tell me about X" remains COMPANY_RESEARCH.
        self.assertEqual(detect_intent("Tell me about Nvidia"), Intent.COMPANY_RESEARCH)
        self.assertEqual(detect_intent("What is TSLA?"), Intent.COMPANY_RESEARCH)
        self.assertEqual(detect_intent("How is Apple doing?"), Intent.COMPANY_RESEARCH)


if __name__ == "__main__":
    unittest.main()