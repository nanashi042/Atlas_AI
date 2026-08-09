"""Environment-backed runtime configuration without secret-bearing defaults."""

from __future__ import annotations

import logging
import os
from typing import Mapping

from dotenv import load_dotenv

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when the bot cannot safely start with its current configuration."""


def _positive_int(values: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(values.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _is_configured(value: str | None) -> bool:
    """Reject empty values and the safe placeholders used in .env.example."""
    return bool(value and value.strip() and not value.strip().upper().startswith("YOUR_"))


class Settings:
    def __init__(self, environ: Mapping[str, str] | None = None):
        values = os.environ if environ is None else environ
        self.APP_NAME = values.get("APP_NAME", "Atlas AI")
        self.APP_VERSION = values.get("APP_VERSION", "1.0.0")
        self.ENVIRONMENT = values.get("ENVIRONMENT", "development").lower()
        self.DATABASE_URL = values.get("DATABASE_URL", "sqlite:///./atlas.db")
        self.GEMINI_API_KEY = values.get("GEMINI_API_KEY")
        self.GEMINI_MODEL = values.get("GEMINI_MODEL", "gemini-flash-latest")
        self.TELEGRAM_BOT_TOKEN = values.get("TELEGRAM_BOT_TOKEN")
        # Public base URL for the Telegram webhook (e.g. "https://example.com")
        self.TELEGRAM_WEBHOOK_BASE_URL = values.get("TELEGRAM_WEBHOOK_BASE_URL")
        # Optional custom path for webhook on your host (e.g. '/api/telegram').
        # If not set, defaults to '/telegram/<bot_token>'.
        self.TELEGRAM_WEBHOOK_PATH = values.get("TELEGRAM_WEBHOOK_PATH")
        # Optional secret token to validate incoming webhook requests. If set,
        # the webhook will be configured with Telegram's `secret_token` and
        # incoming requests must include the header
        # 'X-Telegram-Bot-Api-Secret-Token' with this value.
        self.TELEGRAM_WEBHOOK_SECRET = values.get("TELEGRAM_WEBHOOK_SECRET")
        self.FINNHUB_API_KEY = values.get("FINNHUB_API_KEY")
        self.PRICE_ALERT_INTERVAL_MINUTES = _positive_int(values, "PRICE_ALERT_INTERVAL_MINUTES", 15, 5)
        self.DOCUMENT_MAX_CHARACTERS = _positive_int(values, "DOCUMENT_MAX_CHARACTERS", 24000, 2000)
        self.DOCUMENT_MAX_PAGES = _positive_int(values, "DOCUMENT_MAX_PAGES", 40, 1)
        self.TELEGRAM_MAX_MESSAGE_LENGTH = _positive_int(values, "TELEGRAM_MAX_MESSAGE_LENGTH", 3900, 500)
        self.LOG_LEVEL = values.get("LOG_LEVEL", "INFO").upper()

    def validate_bot_runtime(self) -> None:
        """Ensure every external service required by the polling bot is configured."""
        required = {
            "TELEGRAM_BOT_TOKEN": self.TELEGRAM_BOT_TOKEN,
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "FINNHUB_API_KEY": self.FINNHUB_API_KEY,
        }
        missing = [name for name, value in required.items() if not _is_configured(value)]
        if missing:
            raise ConfigurationError(
                "Missing required runtime configuration: " + ", ".join(missing) + "."
            )
        if not self.DATABASE_URL:
            raise ConfigurationError("Missing required runtime configuration: DATABASE_URL.")


def configure_logging() -> None:
    """Configure normal production logging once, without logging secrets."""
    level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
    )


settings = Settings()
