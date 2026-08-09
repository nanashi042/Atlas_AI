import logging

from fastapi import FastAPI, Request, HTTPException

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
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
    )

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
        except ConfigurationError as exc:
            logger.critical("Bot startup blocked by configuration: %s", exc)
            raise

        # Attempt to initialize the database but do not abort startup if the
        # filesystem is read-only (common on serverless platforms). The
        # application can continue in a degraded mode without persistent DB.
        try:
            ok = init_db(raise_on_error=False)
            if not ok:
                logger.warning("Database initialization did not complete; continuing without DB.")
        except Exception:
            logger.exception("Unexpected error during database initialization; continuing without DB.")

        application = build_application()
        fastapi_app.state.telegram_app = application

        await application.initialize()
        await application.start()

        # Configure webhook if base URL provided
        if settings.TELEGRAM_WEBHOOK_BASE_URL:
            base = settings.TELEGRAM_WEBHOOK_BASE_URL.rstrip("/")
            path = settings.TELEGRAM_WEBHOOK_PATH or f"/telegram/{settings.TELEGRAM_BOT_TOKEN}"
            webhook_url = base + path
            # If a webhook secret is configured, pass it to Telegram so
            # Telegram will include the secret header on incoming requests.
            if settings.TELEGRAM_WEBHOOK_SECRET:
                await application.bot.set_webhook(webhook_url, secret_token=settings.TELEGRAM_WEBHOOK_SECRET)
            else:
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

    # Determine webhook path and register a route programmatically so the
    # path can be a fixed host route (e.g. '/api/telegram') or include the
    # bot token (default '/telegram/<token>'). If a webhook secret is set,
    # validate the incoming header `X-Telegram-Bot-Api-Secret-Token`.
    webhook_path = (
        settings.TELEGRAM_WEBHOOK_PATH
        if settings.TELEGRAM_WEBHOOK_PATH
        else f"/telegram/{settings.TELEGRAM_BOT_TOKEN}"
    )

    async def telegram_webhook(request: Request):
        # If a secret is configured, validate the header
        if settings.TELEGRAM_WEBHOOK_SECRET:
            header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if header != settings.TELEGRAM_WEBHOOK_SECRET:
                raise HTTPException(status_code=403, detail="Invalid webhook secret")

        application = getattr(fastapi_app.state, "telegram_app", None)
        if application is None:
            raise HTTPException(status_code=503, detail="Telegram application not initialized")
        # Safely handle empty or non-JSON bodies (some platforms send probes).
        try:
            # Read raw body first so we can log or early-return on empty bodies.
            body_bytes = await request.body()
        except Exception as e:
            logger.exception("Failed to read request body: %s", e)
            raise HTTPException(status_code=400, detail="Failed to read request body")

        if not body_bytes:
            # Some HTTP probes hit this endpoint with no body; acknowledge them
            # so the sender doesn't retry. Return 200 so Telegram (or probes)
            # consider the delivery successful.
            logger.warning("Empty request body received at Telegram webhook; ignoring.")
            return {"ok": True}

        import json
        from json import JSONDecodeError

        try:
            update_json = json.loads(body_bytes)
        except JSONDecodeError:
            logger.warning("Received non-JSON body at webhook: %s", body_bytes[:200])
            # Don't cause repeated retries from Telegram; acknowledge but log.
            return {"ok": True}

        try:
            from telegram import Update

            update = Update.de_json(update_json, application.bot)
            await application.process_update(update)
        except Exception:
            logger.exception("Failed to process Telegram update")
            # Acknowledge to avoid retries; inspect logs for the root cause.
            return {"ok": True}

        return {"ok": True}

    # Add the webhook route for POST requests at the configured path.
    fastapi_app.add_api_route(webhook_path, telegram_webhook, methods=["POST"])

    async def _telegram_status():
        """Internal status endpoint: whether the Telegram Application started and webhook info.

        This endpoint is intentionally non-sensitive: it does not return tokens or
        secrets. It helps confirm the bot application initialized and what webhook
        Telegram reports.
        """
        application = getattr(fastapi_app.state, "telegram_app", None)
        status = {"initialized": bool(application)}
        if not application:
            return status

        try:
            # Get webhook info from Telegram via the bot API. The returned
            # object may include the webhook URL; mask any token-like segments
            # before returning.
            info = await application.bot.get_webhook_info()
            webhook_url = getattr(info, "url", None)
            if webhook_url and settings.TELEGRAM_BOT_TOKEN:
                webhook_url = webhook_url.replace(settings.TELEGRAM_BOT_TOKEN, "<redacted>")

            status.update(
                {
                    "webhook_url": webhook_url,
                    "pending_update_count": getattr(info, "pending_update_count", None),
                    "last_error_message": getattr(info, "last_error_message", None),
                }
            )
        except Exception:
            logger.exception("Failed to fetch webhook info")
            status["webhook_error"] = True

        return status

    fastapi_app.add_api_route("/_internal/telegram_status", _telegram_status, methods=["GET"])

    async def _send_test_message(request: Request):
        """Send a test message: simulate a user message through `process_message`
        and have the bot send the generated reply to `chat_id`.

        Request JSON: { "chat_id": 12345, "text": "hello" }
        Header: X-Debug-Secret must match `TELEGRAM_DEBUG_SECRET` if configured.
        """
        # Protect the endpoint with the debug secret if configured.
        if settings.TELEGRAM_DEBUG_SECRET:
            header = request.headers.get("X-Debug-Secret")
            if header != settings.TELEGRAM_DEBUG_SECRET:
                raise HTTPException(status_code=403, detail="Invalid debug secret")

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        chat_id = body.get("chat_id")
        user_text = body.get("text", "hello")
        if not chat_id:
            raise HTTPException(status_code=400, detail="chat_id is required")

        application = getattr(fastapi_app.state, "telegram_app", None)
        if application is None:
            raise HTTPException(status_code=503, detail="Telegram application not initialized")

        # Run the message through the manager to produce a reply, then send it.
        try:
            from app.agent.manager import process_message

            reply = await process_message(user_text, session_id=str(chat_id))
        except Exception:
            logger.exception("Failed to generate reply via process_message")
            raise HTTPException(status_code=500, detail="Failed to generate reply")

        try:
            await application.bot.send_message(chat_id=chat_id, text=reply)
        except Exception:
            logger.exception("Failed to send test message via bot.send_message")
            raise HTTPException(status_code=500, detail="Failed to send message")

        return {"ok": True, "reply": reply}

    fastapi_app.add_api_route("/_internal/send_test_message", _send_test_message, methods=["POST"])

    # Register startup/shutdown handlers using the `on_event` decorator
    fastapi_app.on_event("startup")(_startup)
    fastapi_app.on_event("shutdown")(_shutdown)

