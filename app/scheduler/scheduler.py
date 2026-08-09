"""APScheduler coordination for proactive daily Telegram briefings."""

from __future__ import annotations

import logging
import asyncio
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config.settings import settings
from app.services.alert_evaluator import alert_evaluator
from app.services.briefing_preference_service import briefing_preference_service
from app.services.briefing_service import briefing_service
from app.services.notification_service import notification_service
from app.services.watchlist_service import watchlist_service

logger = logging.getLogger(__name__)

PRICE_ALERT_JOB_ID = "price_alert_evaluation"


class MorningBriefingScheduler:
    """Coordinates daily briefings and global periodic price-alert evaluation."""

    def __init__(
        self, scheduler=None, preferences=None, briefings=None, watchlist=None,
        notifications=None, evaluator=None, alert_interval_minutes=None,
    ):
        self.scheduler = scheduler or AsyncIOScheduler()
        self.preferences = preferences or briefing_preference_service
        self.briefings = briefings or briefing_service
        self.watchlist = watchlist or watchlist_service
        self.notifications = notifications or notification_service
        self.evaluator = evaluator or alert_evaluator
        self.alert_interval_minutes = (
            settings.PRICE_ALERT_INTERVAL_MINUTES
            if alert_interval_minutes is None else max(5, alert_interval_minutes)
        )

    @staticmethod
    def job_id(user_id: str) -> str:
        return f"morning_briefing_{user_id}"

    async def start(self, application) -> None:
        """Bind delivery, start once, and restore jobs from durable preferences."""
        self.notifications.bind_application(application)
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Morning briefing scheduler started.")
        self.schedule_price_alert_evaluation()
        for preference in self.preferences.get_enabled_preferences():
            self.schedule_daily_briefing(preference)

    async def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            # AsyncIOScheduler dispatches shutdown onto the current event loop.
            # Yield once so lifecycle callers do not return while it is active.
            await asyncio.sleep(0)
            logger.info("Morning briefing scheduler stopped.")

    def schedule_daily_briefing(self, preference) -> None:
        """Create or replace the stable job for one enabled user."""
        if not preference.morning_briefing_enabled:
            self.remove_daily_briefing(preference.user_id)
            return
        timezone = ZoneInfo(preference.timezone)
        trigger = CronTrigger(
            hour=preference.briefing_time.hour,
            minute=preference.briefing_time.minute,
            timezone=timezone,
        )
        # APScheduler only applies ``replace_existing`` once pending jobs are
        # materialized at scheduler startup, so explicitly remove a pending
        # job first as well. This keeps a stable ID unique before and after
        # startup/restarts.
        self.remove_daily_briefing(preference.user_id)
        self.scheduler.add_job(
            self.execute_user_briefing,
            trigger=trigger,
            id=self.job_id(preference.user_id),
            args=[preference.user_id],
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        logger.info("Scheduled morning briefing for user '%s'.", preference.user_id)

    def remove_daily_briefing(self, user_id: str) -> None:
        job = self.scheduler.get_job(self.job_id(user_id))
        if job:
            self.scheduler.remove_job(job.id)
            logger.info("Removed morning briefing for user '%s'.", user_id)

    def schedule_price_alert_evaluation(self) -> None:
        """Create or replace the single global periodic price-alert job."""
        existing = self.scheduler.get_job(PRICE_ALERT_JOB_ID)
        if existing:
            self.scheduler.remove_job(PRICE_ALERT_JOB_ID)
        self.scheduler.add_job(
            self.execute_price_alert_evaluation,
            trigger=IntervalTrigger(minutes=self.alert_interval_minutes),
            id=PRICE_ALERT_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        logger.info(
            "Scheduled global price-alert evaluation every %s minutes.",
            self.alert_interval_minutes,
        )

    async def execute_user_briefing(self, user_id: str) -> None:
        """Generate and deliver one user's briefing; errors never escape the job."""
        try:
            if not self.watchlist.get_watchlist(user_id):
                logger.info("Skipping morning briefing for user %s: empty watchlist.", user_id)
                return
            text = await self.briefings.generate_briefing(user_id=user_id)
            await self.notifications.send_message(user_id, text)
        except Exception as exc:
            logger.exception("Morning briefing failed for user '%s': %s", user_id, exc)

    async def execute_enabled_briefings(self) -> None:
        """Run enabled users independently; useful for tests and future batch triggers."""
        for preference in self.preferences.get_enabled_preferences():
            await self.execute_user_briefing(preference.user_id)

    async def execute_price_alert_evaluation(self) -> None:
        """Evaluate price alerts and deliver only newly triggered events."""
        try:
            triggered_alerts = await self.evaluator.evaluate_active_alerts()
        except Exception as exc:
            logger.exception("Price-alert evaluation failed: %s", exc)
            return

        for alert in triggered_alerts:
            try:
                await self.notifications.send_message(
                    alert.user_id, self.format_price_alert_notification(alert)
                )
            except Exception as exc:
                logger.exception(
                    "Price-alert notification failed for user '%s', alert '%s': %s",
                    alert.user_id, alert.alert_id, exc,
                )

    @staticmethod
    def format_price_alert_notification(alert) -> str:
        """Render deterministic mobile-friendly PRICE_CHANGE notification text."""
        direction = "UP" if alert.percentage_change >= 0 else "DOWN"
        return (
            f"🚨 {alert.symbol} Price Alert\n\n"
            f"{alert.symbol} is {direction} {abs(alert.percentage_change):.2f}% today.\n\n"
            f"Current price: ${alert.current_price:.2f}\n"
            f"Previous close: ${alert.previous_close:.2f}\n"
            f"Your threshold: {alert.threshold_percentage:g}%\n\n"
            "Your alert threshold has been crossed.\n\n"
            "Source: Finnhub"
        )


morning_briefing_scheduler = MorningBriefingScheduler()
