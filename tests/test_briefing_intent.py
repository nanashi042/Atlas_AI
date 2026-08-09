"""Tests for BRIEFING intent detection."""

import unittest

from app.agent.intent import detect_intent, Intent


class TestBriefingIntent(unittest.TestCase):
    """Ensure natural briefing requests route to BRIEFING and that the
    new patterns do not over-trigger COMPANY_RESEARCH, WATCHLIST_ADD,
    WATCHLIST_LIST, or general chat."""

    def test_explicit_briefing_phrases(self):
        queries = [
            "Give me my morning briefing",
            "Give me my daily briefing",
            "Give me today's market briefing",
            "Daily briefing",
            "Morning briefing please",
            "Today's briefing",
            "Evening briefing",
            "Briefing",
            "Can I get my briefing?",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.BRIEFING,
                    f"Failed for: {q!r}",
                )

    def test_whats_happening_with_watchlist(self):
        queries = [
            "What's happening with my watchlist?",
            "What's happening with my stocks?",
            "What's happening with my stocks today?",
            "What's happening with my companies?",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.BRIEFING,
                    f"Failed for: {q!r}",
                )

    def test_what_should_i_know(self):
        queries = [
            "What should I know about my stocks today?",
            "What should I know today?",
            "What should I know about my watchlist?",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.BRIEFING,
                    f"Failed for: {q!r}",
                )

    def test_whats_important(self):
        queries = [
            "What's important for my stocks?",
            "What's important today?",
            "What's important for my watchlist?",
            "What's important for the companies I'm following?",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.BRIEFING,
                    f"Failed for: {q!r}",
                )

    def test_company_research_still_routes_to_research(self):
        queries = [
            "What is Nvidia?",
            "Tell me about Nvidia",
            "Research Microsoft",
            "How is Apple doing?",
            "What is happening with Tesla?",
            "Analyze AAPL",
            "Give me info on NVDA",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.COMPANY_RESEARCH,
                    f"Failed for: {q!r}",
                )

    def test_watchlist_add_still_routes_to_add(self):
        queries = [
            "Track Nvidia.",
            "Add TSLA to my watchlist",
            "Keep an eye on AMD",
            "Follow NVDA",
            "Start tracking Apple",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.WATCHLIST_ADD,
                    f"Failed for: {q!r}",
                )

    def test_watchlist_remove_still_routes_to_remove(self):
        queries = [
            "Stop tracking Tesla",
            "Remove AAPL from my watchlist",
            "Untrack Microsoft",
            "Stop following Apple",
            "Delete NVDA from my watchlist",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.WATCHLIST_REMOVE,
                    f"Failed for: {q!r}",
                )

    def test_watchlist_list_still_routes_to_list(self):
        # Pure list-style queries must remain WATCHLIST_LIST so users
        # asking "what's on my watchlist?" still get the list view.
        queries = [
            "Show my watchlist",
            "List my watchlist",
            "View my watchlist",
            "What's on my watchlist?",
            "My watchlist",
            "My tracked stocks",
            "What stocks am I following?",
            "What am I tracking?",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertEqual(
                    detect_intent(q),
                    Intent.WATCHLIST_LIST,
                    f"Failed for: {q!r}",
                )

    def test_general_chat_does_not_over_trigger_briefing(self):
        queries = [
            "What is inflation?",
            "How do stock options work?",
            "Tell me a joke",
            "What's important for dinner?",
        ]
        for q in queries:
            with self.subTest(q=q):
                self.assertNotEqual(
                    detect_intent(q),
                    Intent.BRIEFING,
                    f"Incorrectly routed: {q!r}",
                )


if __name__ == "__main__":
    unittest.main()
