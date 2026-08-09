"""Tests for scheduler delivery of evaluated price-change alerts."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.scheduler.alert_live_test import trigger_live_alert_evaluation
from app.scheduler.scheduler import MorningBriefingScheduler, PRICE_ALERT_JOB_ID
from app.services.alert_evaluator import TriggeredAlert


def _triggered(user_id="1001", symbol="NVDA", change=5.8, threshold=5):
    return TriggeredAlert(
        alert_id=1,
        user_id=user_id,
        symbol=symbol,
        current_price=105.8,
        previous_close=100.0,
        percentage_change=change,
        threshold_percentage=threshold,
        reason="threshold met",
    )


class TestPriceAlertScheduler(unittest.IsolatedAsyncioTestCase):
    def _scheduler(self, evaluator=None, notifications=None):
        return MorningBriefingScheduler(
            scheduler=AsyncIOScheduler(),
            preferences=MagicMock(),
            briefings=MagicMock(),
            watchlist=MagicMock(),
            evaluator=evaluator or MagicMock(),
            notifications=notifications or MagicMock(),
            alert_interval_minutes=15,
        )

    async def test_global_alert_job_has_stable_id_without_duplicates(self):
        scheduler = self._scheduler()
        scheduler.schedule_price_alert_evaluation()
        scheduler.schedule_price_alert_evaluation()
        jobs = [job for job in scheduler.scheduler.get_jobs() if job.id == PRICE_ALERT_JOB_ID]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].trigger.interval.total_seconds(), 15 * 60)

    async def test_triggered_alert_sends_deterministic_up_message_to_its_user(self):
        evaluator = MagicMock()
        evaluator.evaluate_active_alerts = AsyncMock(return_value=[_triggered()])
        notifications = MagicMock()
        notifications.send_message = AsyncMock()
        scheduler = self._scheduler(evaluator, notifications)
        await scheduler.execute_price_alert_evaluation()
        sent_user, sent_text = notifications.send_message.await_args.args
        self.assertEqual(sent_user, "1001")
        self.assertIn("NVDA is UP 5.80%", sent_text)
        self.assertIn("Your threshold: 5%", sent_text)
        self.assertIn("Source: Finnhub", sent_text)

    async def test_negative_message_and_multiple_alerts_are_sent_separately(self):
        evaluator = MagicMock()
        evaluator.evaluate_active_alerts = AsyncMock(return_value=[
            _triggered("one", "NVDA", 6), _triggered("two", "TSLA", -6.2),
        ])
        notifications = MagicMock()
        notifications.send_message = AsyncMock()
        scheduler = self._scheduler(evaluator, notifications)
        await scheduler.execute_price_alert_evaluation()
        self.assertEqual(notifications.send_message.await_count, 2)
        self.assertEqual(notifications.send_message.await_args_list[1].args[0], "two")
        self.assertIn("TSLA is DOWN 6.20%", notifications.send_message.await_args_list[1].args[1])

    async def test_no_triggered_or_active_alerts_sends_nothing(self):
        evaluator = MagicMock()
        evaluator.evaluate_active_alerts = AsyncMock(return_value=[])
        notifications = MagicMock()
        notifications.send_message = AsyncMock()
        scheduler = self._scheduler(evaluator, notifications)
        await scheduler.execute_price_alert_evaluation()
        notifications.send_message.assert_not_awaited()

    async def test_one_telegram_failure_does_not_block_later_notifications(self):
        evaluator = MagicMock()
        evaluator.evaluate_active_alerts = AsyncMock(return_value=[
            _triggered("bad", "NVDA", 6), _triggered("good", "TSLA", 7),
        ])
        notifications = MagicMock()
        notifications.send_message = AsyncMock(side_effect=[RuntimeError("blocked"), None])
        scheduler = self._scheduler(evaluator, notifications)
        await scheduler.execute_price_alert_evaluation()
        self.assertEqual(notifications.send_message.await_count, 2)
        self.assertEqual(notifications.send_message.await_args_list[1].args[0], "good")


class TestLiveAlertEvaluationTrigger(unittest.IsolatedAsyncioTestCase):
    async def test_calls_production_callback_without_creating_a_job(self):
        application = MagicMock()
        application.initialize = AsyncMock()
        application.shutdown = AsyncMock()
        factory = MagicMock(return_value=application)

        from unittest.mock import patch
        with patch("app.scheduler.alert_live_test.notification_service.bind_application") as bind, patch(
            "app.scheduler.alert_live_test.morning_briefing_scheduler.execute_price_alert_evaluation",
            new_callable=AsyncMock,
        ) as execute:
            await trigger_live_alert_evaluation(application_factory=factory)

        application.initialize.assert_awaited_once()
        bind.assert_called_once_with(application)
        execute.assert_awaited_once()
        application.shutdown.assert_awaited_once()

