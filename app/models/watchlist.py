"""
SQLAlchemy model for the persistent user-specific watchlist.

The watchlist is structured user preference data (not conversation memory),
stored separately in its own table. Entries are unique per (user_id, symbol)
to prevent duplicates and ticker symbols are normalized to uppercase.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint

from app.database.session import Base


class WatchlistEntry(Base):
    """
    A single company tracked by a single Telegram user.

    Each row represents one ticker the user wants to monitor.
    The (user_id, symbol) pair is unique to prevent duplicate entries.
    """

    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    symbol = Column(String, nullable=False)
    company_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_watchlist_user_symbol"),
    )

    def __repr__(self) -> str:
        return (
            f"<WatchlistEntry(user_id='{self.user_id}', "
            f"symbol='{self.symbol}', company_name='{self.company_name}')>"
        )