"""Tests for the Telegram onboarding and command menu surface."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from app.bot.handlers import HELP_MESSAGE, START_MESSAGE, help_command, start_command
from app.bot.telegram_bot import COMMAND_MENU, _set_command_menu


class TestOnboardingMessages(unittest.IsolatedAsyncioTestCase):
    async def test_start_contains_feature_overview_and_try_section(self):
        message = MagicMock()
        message.reply_text = AsyncMock()
        await start_command(MagicMock(message=message), MagicMock())
        text = message.reply_text.await_args.args[0]
        self.assertEqual(text, START_MESSAGE)
        for feature in ("Company research", "Watchlist", "Price alerts", "Daily briefing", "Financial PDFs", "General AI questions"):
            self.assertIn(feature, text)
        for example in ("What is NVIDIA?", "Add Tesla to my watchlist", "What was the revenue growth?"):
            self.assertIn(example, text)
        self.assertIn("Try this:", text)

    async def test_help_is_dedicated_and_reference_oriented(self):
        message = MagicMock()
        message.reply_text = AsyncMock()
        await help_command(MagicMock(message=message), MagicMock())
        text = message.reply_text.await_args.args[0]
        self.assertEqual(text, HELP_MESSAGE)
        self.assertIn("/briefing_status", text)
        self.assertIn("/document_clear", text)
        self.assertIn("Upload a financial PDF", text)


class TestCommandMenu(unittest.IsolatedAsyncioTestCase):
    async def test_command_menu_is_registered_with_implemented_commands(self):
        application = MagicMock()
        application.bot.set_my_commands = AsyncMock()
        await _set_command_menu(application)
        application.bot.set_my_commands.assert_awaited_once()
        commands = application.bot.set_my_commands.await_args.args[0]
        names = [command.command for command in commands]
        self.assertIn("start", names)
        self.assertIn("help", names)
        self.assertIn("briefing_on", names)
        self.assertIn("document_clear", names)
        self.assertNotIn("watchlist", names)
        self.assertNotIn("alerts", names)
        self.assertEqual(names, [command.command for command in COMMAND_MENU])
