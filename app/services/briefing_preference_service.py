"""Database-backed preferences for a user's scheduled morning briefing."""

import logging
from datetime import time
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from app.database.session import SessionLocal, init_db
from app.models.briefing_preference import BriefingPreference

logger = logging.getLogger(__name__)

DEFAULT_BRIEFING_TIME = time(hour=8, minute=0)
DEFAULT_TIMEZONE = "Asia/Kolkata"


class BriefingPreferenceError(Exception):
    """Raised when briefing preferences cannot be loaded or saved."""


class BriefingPreferenceService:
    """Owns CRUD for the small, persistent briefing-preference record."""

    def __init__(self):
        init_db()

    def get_preference(self, user_id: str) -> BriefingPreference:
        """Return a user's preference, creating the disabled default if needed."""
        db = SessionLocal()
        try:
            preference = (
                db.query(BriefingPreference)
                .filter(BriefingPreference.user_id == str(user_id))
                .one_or_none()
            )
            if preference is None:
                preference = BriefingPreference(
                    user_id=str(user_id),
                    morning_briefing_enabled=False,
                    briefing_time=DEFAULT_BRIEFING_TIME,
                    timezone=DEFAULT_TIMEZONE,
                )
                db.add(preference)
                db.commit()
                db.refresh(preference)
            db.expunge(preference)
            return preference
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not load briefing preference for user '%s': %s", user_id, exc)
            raise BriefingPreferenceError("Could not load briefing preference.") from exc
        finally:
            db.close()

    def set_enabled(self, user_id: str, enabled: bool) -> BriefingPreference:
        """Persist enabled state while preserving the user's time and timezone."""
        preference = self.get_preference(user_id)
        db = SessionLocal()
        try:
            stored = db.merge(preference)
            stored.morning_briefing_enabled = enabled
            db.commit()
            db.refresh(stored)
            db.expunge(stored)
            return stored
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not update briefing preference for user '%s': %s", user_id, exc)
            raise BriefingPreferenceError("Could not update briefing preference.") from exc
        finally:
            db.close()

    def get_enabled_preferences(self) -> list[BriefingPreference]:
        """Return enabled preferences for scheduler startup synchronization."""
        db = SessionLocal()
        try:
            preferences = (
                db.query(BriefingPreference)
                .filter(BriefingPreference.morning_briefing_enabled.is_(True))
                .all()
            )
            for preference in preferences:
                db.expunge(preference)
            return preferences
        except SQLAlchemyError as exc:
            logger.error("Could not load enabled briefing preferences: %s", exc)
            raise BriefingPreferenceError("Could not load enabled briefing preferences.") from exc
        finally:
            db.close()


briefing_preference_service = BriefingPreferenceService()
