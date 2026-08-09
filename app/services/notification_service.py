"""Thin Telegram delivery abstraction for proactive notifications."""

from __future__ import annotations


class NotificationService:
    """Sends ready-to-display text without owning scheduling or briefing logic."""

    def __init__(self):
        self._application = None

    def bind_application(self, application) -> None:
        self._application = application

    async def send_message(self, user_id: str, text: str) -> None:
        if self._application is None:
            raise RuntimeError("Telegram application has not been initialized.")
        await self._application.bot.send_message(chat_id=int(user_id), text=text)


notification_service = NotificationService()
