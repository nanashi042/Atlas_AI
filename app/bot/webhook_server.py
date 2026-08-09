import logging

from fastapi import FastAPI, Request, HTTPException
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from app.config.settings import ConfigurationError, configure_logging, settings
from app.database.session import init_db
from app.bot.handlers import (
    start_command,
    clear_command,
    briefing_on_command,
    briefing_off_command,
    briefing_status_command,
    document_clear_command,
    help_command,
    handle_message,
)
from app.bot.pdf_handler import handle_pdf_document
from app.scheduler.scheduler import morning_briefing_scheduler

logger = logging.getLogger(__name__)

configure_logging()


async def error_handler(update, context):
    logger.error("Unhandled error: %s", context.error)


async def _start_scheduler(application):
    await morning_briefing_scheduler.start(application)
    await _set_command_menu(application)


async def _set_command_menu(application):
    from telegram import BotCommand

    COMMAND_MENU = (
        BotCommand("start", "Welcome and feature overview"),
        BotCommand("help", "Show help"),
        BotCommand("clear", "Clear conversation memory"),
        BotCommand("reset", "Clear conversation memory"),
        BotCommand("briefing_on", "Enable daily briefing"),
        BotCommand("briefing_off", "Disable daily briefing"),
        BotCommand("briefing_status", "Check briefing status"),
        BotCommand("document_clear", "Clear active PDF document"),
    )
    await application.bot.set_my_commands(list(COMMAND_MENU))


async def _stop_scheduler(application):
    await morning_briefing_scheduler.stop()


def build_application():
    app = (
        Application.builder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .post_init(_start_scheduler)
        .post_shutdown(_stop_scheduler)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("reset", clear_command))
    app.add_handler(CommandHandler("briefing_on", briefing_on_command))
    app.add_handler(CommandHandler("briefing_off", briefing_off_command))
    app.add_handler(CommandHandler("briefing_status", briefing_status_command))
    app.add_handler(CommandHandler("document_clear", document_clear_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handle_pdf_document,
        )
    )

    app.add_error_handler(error_handler)

    return app


def attach_telegram_to_fastapi(fastapi_app: FastAPI):
    """Attach telegram application to a FastAPI app via startup/shutdown events and a webhook route."""

    async def _startup():
        try:
            settings.validate_bot_runtime()
            init_db(raise_on_error=True)
        except ConfigurationError as exc:
            logger.critical("Bot startup blocked by configuration: %s", exc)
            raise

        application = build_application()
        fastapi_app.state.telegram_app = application

        await application.initialize()
        await application.start()

        # Configure webhook if base URL provided
        if settings.TELEGRAM_WEBHOOK_BASE_URL:
            webhook_url = settings.TELEGRAM_WEBHOOK_BASE_URL.rstrip("/") + f"/telegram/{settings.TELEGRAM_BOT_TOKEN}"
            await application.bot.set_webhook(webhook_url)
            logger.info("Set Telegram webhook to %s", webhook_url)
        else:
            logger.info("No TELEGRAM_WEBHOOK_BASE_URL set; webhook not configured.")

    async def _shutdown():
        application = getattr(fastapi_app.state, "telegram_app", None)
        if not application:
            return
        try:
            await application.bot.delete_webhook()
        except Exception:
            pass
        await application.stop()
        await application.shutdown()

    @fastapi_app.post("/telegram/{token}")
    async def telegram_webhook(token: str, request: Request):
        if token != settings.TELEGRAM_BOT_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")
        application = getattr(fastapi_app.state, "telegram_app", None)
        if application is None:
            raise HTTPException(status_code=503, detail="Telegram application not initialized")

        update_json = await request.json()
        update = Update.de_json(update_json, application.bot)
        await application.process_update(update)
        return {"ok": True}

    fastapi_app.add_event_handler("startup", _startup)
    fastapi_app.add_event_handler("shutdown", _shutdown)

