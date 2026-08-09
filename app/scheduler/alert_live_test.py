"""Development-only one-shot verification for the price-alert delivery flow."""

from __future__ import annotations

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


async def trigger_live_alert_evaluation(application_factory=None) -> None:
    """Run the scheduler's real alert callback once without registering a job."""
    application = (application_factory or _build_application)()
    await application.initialize()
    try:
        notification_service.bind_application(application)
        await morning_briefing_scheduler.execute_price_alert_evaluation()
    finally:
        await application.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(trigger_live_alert_evaluation())
    logger.info("Live price-alert evaluation finished.")


if __name__ == "__main__":
    main()
