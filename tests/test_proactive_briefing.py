"""Tests for Phase 1: persistent proactive daily briefing scheduling."""

import unittest
from datetime import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.agent.intent import Intent, detect_intent
from app.database.session import SessionLocal
from app.models.briefing_preference import BriefingPreference
from app.scheduler.scheduler import MorningBriefingScheduler
from app.scheduler.live_test import trigger_live_briefing_test
from app.services.briefing_preference_service import (
    DEFAULT_BRIEFING_TIME,
    DEFAULT_TIMEZONE,
    briefing_preference_service,
)


class PreferenceCleanupMixin:
    USER_IDS = ()

    def tearDown(self):
        db = SessionLocal()
        try:
            db.query(BriefingPreference).filter(
                BriefingPreference.user_id.in_(self.USER_IDS)
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()


class TestBriefingPreferences(PreferenceCleanupMixin, unittest.TestCase):
    USER_IDS = ("phase1_pref_user",)

    def test_default_is_disabled_with_india_morning_time(self):
        preference = briefing_preference_service.get_preference(self.USER_IDS[0])
        self.assertFalse(preference.morning_briefing_enabled)
        self.assertEqual(preference.briefing_time, DEFAULT_BRIEFING_TIME)
        self.assertEqual(preference.timezone, DEFAULT_TIMEZONE)

    def test_enabled_state_persists_across_service_reads(self):
        briefing_preference_service.set_enabled(self.USER_IDS[0], True)
        reloaded = briefing_preference_service.get_preference(self.USER_IDS[0])
        self.assertTrue(reloaded.morning_briefing_enabled)
        self.assertEqual(reloaded.briefing_time, time(8, 0))
        self.assertEqual(reloaded.timezone, "Asia/Kolkata")

    def test_disable(self):
        briefing_preference_service.set_enabled(self.USER_IDS[0], True)
        preference = briefing_preference_service.set_enabled(self.USER_IDS[0], False)
        self.assertFalse(preference.morning_briefing_enabled)


class TestProactiveBriefingIntent(unittest.TestCase):
    def test_enable_disable_and_status_intents(self):
        cases = {
            "Enable my morning briefing": Intent.BRIEFING_ENABLE,
            "Start my daily briefing": Intent.BRIEFING_ENABLE,
            "Turn off my briefing": Intent.BRIEFING_DISABLE,
            "Is my daily briefing enabled?": Intent.BRIEFING_STATUS,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(detect_intent(message), expected)


class TestMorningBriefingScheduler(unittest.IsolatedAsyncioTestCase):
    def _scheduler(self, preferences=None, briefings=None, watchlist=None, notifications=None):
        return MorningBriefingScheduler(
            scheduler=AsyncIOScheduler(),
            preferences=preferences or MagicMock(),
            briefings=briefings or MagicMock(),
            watchlist=watchlist or MagicMock(),
            notifications=notifications or MagicMock(),
        )

    async def test_stable_job_id_and_duplicate_replacement(self):
        scheduler = self._scheduler()
        preference = SimpleNamespace(
            user_id="1001", morning_briefing_enabled=True,
            briefing_time=time(8, 0), timezone="Asia/Kolkata",
        )
        scheduler.schedule_daily_briefing(preference)
        scheduler.schedule_daily_briefing(preference)
        self.assertEqual(scheduler.job_id("1001"), "morning_briefing_1001")
        self.assertEqual(len(scheduler.scheduler.get_jobs()), 1)
        job = scheduler.scheduler.get_job("morning_briefing_1001")
        self.assertEqual(job.trigger.timezone, ZoneInfo("Asia/Kolkata"))

    async def test_start_restores_enabled_users_and_stop_shuts_down(self):
        preference = SimpleNamespace(
            user_id="1002", morning_briefing_enabled=True,
            briefing_time=time(8, 0), timezone="Asia/Kolkata",
        )
        preferences = MagicMock()
        preferences.get_enabled_preferences.return_value = [preference]
        notifications = MagicMock()
        scheduler = self._scheduler(preferences=preferences, notifications=notifications)
        application = MagicMock()
        await scheduler.start(application)
        self.assertTrue(scheduler.scheduler.running)
        notifications.bind_application.assert_called_once_with(application)
        self.assertIsNotNone(scheduler.scheduler.get_job("morning_briefing_1002"))
        await scheduler.stop()
        self.assertFalse(scheduler.scheduler.running)

    async def test_enabled_user_receives_generated_briefing(self):
        watchlist = MagicMock()
        watchlist.get_watchlist.return_value = [{"symbol": "NVDA"}]
        briefings = MagicMock()
        briefings.generate_briefing = AsyncMock(return_value="Morning briefing text")
        notifications = MagicMock()
        notifications.send_message = AsyncMock()
        scheduler = self._scheduler(briefings=briefings, watchlist=watchlist, notifications=notifications)
        await scheduler.execute_user_briefing("1003")
        briefings.generate_briefing.assert_awaited_once_with(user_id="1003")
        notifications.send_message.assert_awaited_once_with("1003", "Morning briefing text")

    async def test_empty_watchlist_skips_briefing_and_notification(self):
        watchlist = MagicMock()
        watchlist.get_watchlist.return_value = []
        briefings = MagicMock()
        briefings.generate_briefing = AsyncMock()
        notifications = MagicMock()
        notifications.send_message = AsyncMock()
        scheduler = self._scheduler(briefings=briefings, watchlist=watchlist, notifications=notifications)
        await scheduler.execute_user_briefing("1004")
        briefings.generate_briefing.assert_not_awaited()
        notifications.send_message.assert_not_awaited()

    async def test_one_user_failure_does_not_stop_other_users(self):
        preferences = MagicMock()
        preferences.get_enabled_preferences.return_value = [
            SimpleNamespace(user_id="bad"), SimpleNamespace(user_id="good"),
        ]
        watchlist = MagicMock()
        watchlist.get_watchlist.return_value = [{"symbol": "NVDA"}]
        briefings = MagicMock()
        briefings.generate_briefing = AsyncMock(side_effect=[RuntimeError("boom"), "good text"])
        notifications = MagicMock()
        notifications.send_message = AsyncMock()
        scheduler = self._scheduler(
            preferences=preferences,
            briefings=briefings,
            watchlist=watchlist,
            notifications=notifications,
        )
        await scheduler.execute_enabled_briefings()
        self.assertEqual(briefings.generate_briefing.await_count, 2)
        notifications.send_message.assert_awaited_once_with("good", "good text")


class TestLiveBriefingTestTrigger(unittest.IsolatedAsyncioTestCase):
    async def test_invokes_production_callback_without_creating_a_job(self):
        application = MagicMock()
        application.initialize = AsyncMock()
        application.shutdown = AsyncMock()
        factory = MagicMock(return_value=application)

        with patch(
            "app.scheduler.live_test.notification_service.bind_application"
        ) as bind, patch(
            "app.scheduler.live_test.morning_briefing_scheduler.execute_user_briefing",
            new_callable=AsyncMock,
        ) as execute:
            await trigger_live_briefing_test("123456", application_factory=factory)

        application.initialize.assert_awaited_once()
        bind.assert_called_once_with(application)
        execute.assert_awaited_once_with("123456")
        application.shutdown.assert_awaited_once()

    async def test_rejects_non_numeric_telegram_user_id(self):
        with self.assertRaises(ValueError):
            await trigger_live_briefing_test("not-a-telegram-id")


class TestBriefingPreferenceManager(PreferenceCleanupMixin, unittest.IsolatedAsyncioTestCase):
    USER_IDS = ("phase1_manager_user",)

    @patch("app.agent.manager.morning_briefing_scheduler")
    async def test_enable_disable_and_status_responses(self, scheduler):
        from app.agent.manager import process_message

        enabled = await process_message("Enable my morning briefing", self.USER_IDS[0])
        self.assertIn("8:00 AM Asia/Kolkata", enabled)
        scheduler.schedule_daily_briefing.assert_called_once()

        status = await process_message("Is my daily briefing enabled?", self.USER_IDS[0])
        self.assertIn("enabled for 8:00 AM Asia/Kolkata", status)

        disabled = await process_message("Stop my morning briefing", self.USER_IDS[0])
        self.assertIn("disabled", disabled)
        scheduler.remove_daily_briefing.assert_called_once_with(self.USER_IDS[0])
