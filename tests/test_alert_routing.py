"""Natural-language routing tests for Phase 2 Step 2 alert management."""

import unittest

from app.agent.intent import Intent, detect_intent
from app.database.session import SessionLocal
from app.memory.conversation_memory import conversation_memory
from app.models.alert import Alert
from app.services.alert_service import alert_service


class TestAlertRouting(unittest.IsolatedAsyncioTestCase):
    USER = "alert_routing_user"
    OTHER_USER = "alert_routing_other_user"

    def setUp(self):
        self._clean()
        conversation_memory.clear_history(self.USER)
        conversation_memory.clear_history(self.OTHER_USER)

    def tearDown(self):
        self._clean()
        conversation_memory.clear_history(self.USER)
        conversation_memory.clear_history(self.OTHER_USER)

    def _clean(self):
        db = SessionLocal()
        try:
            db.query(Alert).filter(Alert.user_id.in_([self.USER, self.OTHER_USER])).delete(
                synchronize_session=False
            )
            db.commit()
        finally:
            db.close()

    async def test_natural_creation_normalizes_symbol_and_threshold(self):
        from app.agent.manager import process_message
        response = await process_message("Alert me if nvda moves more than 5%", self.USER)
        self.assertIn("NVDA", response)
        self.assertIn("5%", response)
        self.assertIn("PRICE_CHANGE", response)
        alert = alert_service.list_alerts(self.USER, enabled_only=True)[0]
        self.assertEqual(alert["symbol"], "NVDA")
        self.assertEqual(alert["threshold_percentage"], 5.0)

    async def test_company_name_and_percent_word_are_supported(self):
        from app.agent.manager import process_message
        await process_message("Notify me when Tesla changes by 3 percent", self.USER)
        alert = alert_service.list_alerts(self.USER, enabled_only=True)[0]
        self.assertEqual(alert["symbol"], "TSLA")
        self.assertEqual(alert["threshold_percentage"], 3.0)

    async def test_duplicate_listing_and_disabling_are_user_scoped(self):
        from app.agent.manager import process_message
        await process_message("Create an alert for TSLA at 4%", self.USER)
        duplicate = await process_message("Alert me if TSLA moves more than 4%", self.USER)
        self.assertIn("already active", duplicate)
        self.assertEqual(len(alert_service.list_alerts(self.USER)), 1)

        await process_message("Alert me if NVDA moves more than 5%", self.OTHER_USER)
        listing = await process_message("Show my alerts", self.USER)
        self.assertIn("TSLA", listing)
        self.assertNotIn("NVDA", listing)

        removed = await process_message("Stop my TSLA alert", self.USER)
        self.assertIn("disabled", removed)
        self.assertEqual(alert_service.list_alerts(self.USER, enabled_only=True), [])
        self.assertEqual(len(alert_service.list_alerts(self.OTHER_USER, enabled_only=True)), 1)

    async def test_missing_threshold_and_symbol_return_clarifications(self):
        from app.agent.manager import process_message
        missing_threshold = await process_message("Alert me if NVDA moves", self.USER)
        self.assertIn("positive percentage", missing_threshold)
        invalid_threshold = await process_message("Alert me if NVDA moves 0%", self.USER)
        self.assertIn("positive number", invalid_threshold)
        conversation_memory.clear_history(self.USER)
        missing_symbol = await process_message("Alert me if it moves more than 5%", self.USER)
        self.assertIn("Which stock", missing_symbol)


class TestAlertIntentRegression(unittest.TestCase):
    def test_alert_management_intents(self):
        cases = {
            "Alert me if NVDA moves more than 5%": Intent.ALERT_CREATE,
            "Track NVDA if it moves 5%": Intent.ALERT_CREATE,
            "What alerts do I have?": Intent.ALERT_LIST,
            "Remove my TSLA alert": Intent.ALERT_REMOVE,
        }
        for message, intent in cases.items():
            with self.subTest(message=message):
                self.assertEqual(detect_intent(message), intent)

    def test_research_and_watchlist_requests_do_not_become_alerts(self):
        self.assertEqual(detect_intent("What is NVDA?"), Intent.COMPANY_RESEARCH)
        self.assertEqual(detect_intent("Track Nvidia"), Intent.WATCHLIST_ADD)
        self.assertEqual(detect_intent("Show my watchlist"), Intent.WATCHLIST_LIST)
