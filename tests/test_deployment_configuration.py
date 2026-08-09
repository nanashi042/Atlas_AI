"""Focused production-readiness checks that do not use real credentials."""

import unittest

from fastapi.testclient import TestClient

from app.config.settings import ConfigurationError, Settings
from app.main import app


class TestDeploymentConfiguration(unittest.TestCase):
    def test_required_bot_configuration_rejects_missing_or_placeholder_secrets(self):
        settings = Settings({"DATABASE_URL": "sqlite:///./test.db", "TELEGRAM_BOT_TOKEN": "YOUR_TELEGRAM_BOT_TOKEN"})
        with self.assertRaises(ConfigurationError) as error:
            settings.validate_bot_runtime()
        self.assertIn("TELEGRAM_BOT_TOKEN", str(error.exception))
        self.assertIn("GEMINI_API_KEY", str(error.exception))
        self.assertNotIn("YOUR_TELEGRAM_BOT_TOKEN", str(error.exception))

    def test_configuration_uses_safe_defaults_and_parses_limits(self):
        settings = Settings({
            "TELEGRAM_BOT_TOKEN": "token", "GEMINI_API_KEY": "key", "FINNHUB_API_KEY": "finnhub",
            "PRICE_ALERT_INTERVAL_MINUTES": "invalid", "DOCUMENT_MAX_CHARACTERS": "100", "DOCUMENT_MAX_PAGES": "0",
        })
        settings.validate_bot_runtime()
        self.assertEqual(settings.PRICE_ALERT_INTERVAL_MINUTES, 15)
        self.assertEqual(settings.DOCUMENT_MAX_CHARACTERS, 2000)
        self.assertEqual(settings.DOCUMENT_MAX_PAGES, 1)

    def test_health_endpoint_exposes_only_status(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
