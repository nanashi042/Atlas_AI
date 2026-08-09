"""Persistent per-user settings for proactive daily briefings."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Time

from app.database.session import Base


class BriefingPreference(Base):
    """The minimal scheduling preference set for one Telegram user."""

    __tablename__ = "briefing_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    morning_briefing_enabled = Column(Boolean, nullable=False, default=False)
    briefing_time = Column(Time, nullable=False)
    timezone = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
