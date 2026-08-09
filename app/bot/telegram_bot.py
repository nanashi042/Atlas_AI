import logging

from telegram import BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.error import Conflict

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
    if isinstance(context.error, Conflict):
        logger.warning("Conflict error: another bot instance was running. Ignoring.")
        return
    logger.error("Unhandled error: %s", context.error)


async def _start_scheduler(application):
    await morning_briefing_scheduler.start(application)
    await _set_command_menu(application)


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


async def _set_command_menu(application):
    """Publish only commands implemented by this bot to Telegram."""
    await application.bot.set_my_commands(list(COMMAND_MENU))


async def _stop_scheduler(application):
    await morning_briefing_scheduler.stop()


def run_bot():
    try:
        settings.validate_bot_runtime()
        init_db(raise_on_error=True)
    except ConfigurationError as exc:
        logger.critical("Bot startup blocked by configuration: %s", exc)
        raise

    app = (
        Application.builder()
        .token("8782141286:AAEvQFokpuk1YhfXvH2K-327cR5yvIcA1k")
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
            # Route every document so the PDF handler can clearly reject
            # unsupported uploads instead of silently ignoring them.
            filters.Document.ALL,
            handle_pdf_document,
        )
    )

    # Register error handler to suppress Conflict spam
    app.add_error_handler(error_handler)

    logger.info("Atlas AI Bot is starting polling and its in-process scheduler.")

    # drop_pending_updates=True clears any old sessions on startup
    try:
        app.run_polling(drop_pending_updates=True)
    except Exception as exc:
        # Surface invalid token errors with a clearer message for users.
        from telegram.error import InvalidToken

        if isinstance(exc, InvalidToken) or getattr(exc, '__class__', None).__name__ == 'InvalidToken':
            logger.critical("The provided TELEGRAM_BOT_TOKEN is invalid or was rejected by Telegram.")
        raise
