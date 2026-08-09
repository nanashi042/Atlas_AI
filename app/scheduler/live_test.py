"""Development-only one-shot verification for the morning briefing delivery flow.

Run with a Telegram user ID supplied at the command line. This module never
starts APScheduler or creates a job; it invokes the exact callback registered
by the production 08:00 job once and then closes its Telegram application.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from telegram.ext import Application

from app.config.settings import settings
from app.scheduler.scheduler import morning_briefing_scheduler
from app.services.notification_service import notification_service

logger = logging.getLogger(__name__)


def _build_application():
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
    return Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()


async def trigger_live_briefing_test(user_id: str, application_factory=None) -> None:
    """Send one live briefing through the production scheduler callback.

    The caller supplies the intended Telegram ID explicitly. No preference is
    read or written, and no recurring or temporary APScheduler job is added.
    ``application_factory`` exists only to make this development helper testable.
    """
    if not user_id or not str(user_id).isdigit():
        raise ValueError("user_id must be a numeric Telegram user ID.")

    application_factory = application_factory or _build_application
    application = application_factory()
    await application.initialize()
    try:
        notification_service.bind_application(application)
        # This is the same coroutine registered by schedule_daily_briefing.
        await morning_briefing_scheduler.execute_user_briefing(str(user_id))
    finally:
        await application.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only: send one live Atlas morning briefing."
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Numeric Telegram user ID that should receive the test briefing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(trigger_live_briefing_test(args.user_id))
    logger.info("Live morning briefing test finished for user '%s'.", args.user_id)


if __name__ == "__main__":
    main()
