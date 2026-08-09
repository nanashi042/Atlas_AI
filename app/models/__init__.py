"""SQLAlchemy ORM models for Atlas AI."""

# Import models here so ``init_db`` always sees every table.
from app.models.conversation import ChatMessage  # noqa: F401
from app.models.alert import Alert  # noqa: F401
from app.models.briefing_preference import BriefingPreference  # noqa: F401
from app.models.watchlist import WatchlistEntry  # noqa: F401
