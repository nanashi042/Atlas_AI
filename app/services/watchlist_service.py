"""
Watchlist service layer for Atlas AI Financial Copilot.

Provides persistent, user-isolated watchlist operations backed by SQLAlchemy.
This is the single source of truth for what each user is tracking.
Future features (daily briefings, alerts) should reuse this service.
"""

import logging
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database.session import SessionLocal
from app.models.watchlist import WatchlistEntry

logger = logging.getLogger(__name__)


class WatchlistError(Exception):
    """Base exception for watchlist service failures."""


class WatchlistService:
    """
    Service exposing clean CRUD operations over the user watchlist.

    All operations are user-scoped: the user_id is required and treated as
    opaque (Telegram user id is a string in this project). Duplicate entries
    for the same (user_id, symbol) pair are prevented by the DB constraint.
    """

    def add_to_watchlist(
        self, user_id: str, symbol: str, company_name: str
    ) -> Tuple[bool, str]:
        """
        Add a company to the user's watchlist.

        Args:
            user_id: Telegram user identifier (session-scoped).
            symbol: Stock ticker symbol (will be uppercased).
            company_name: Human-readable company name.

        Returns:
            A tuple of (added, message):
              - (True, "<name> added") if newly inserted
              - (False, "<name> already tracked") if duplicate
        """
        if not user_id or not symbol or not company_name:
            logger.warning("add_to_watchlist called with missing fields.")
            return False, "Invalid watchlist request."

        normalized_symbol = symbol.strip().upper()
        normalized_name = company_name.strip()

        db = SessionLocal()
        try:
            entry = WatchlistEntry(
                user_id=str(user_id),
                symbol=normalized_symbol,
                company_name=normalized_name,
            )
            db.add(entry)
            db.commit()
            logger.info(
                f"Added {normalized_symbol} to watchlist for user '{user_id}'."
            )
            return True, f"{normalized_name} ({normalized_symbol}) added to your watchlist."
        except IntegrityError:
            db.rollback()
            logger.info(
                f"Duplicate watchlist entry blocked for user '{user_id}' "
                f"symbol '{normalized_symbol}'."
            )
            return False, f"I'm already tracking {normalized_name} ({normalized_symbol})."
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error adding to watchlist: {e}", exc_info=True)
            raise WatchlistError("Could not add to watchlist.") from e
        finally:
            db.close()

    def remove_from_watchlist(
        self, user_id: str, symbol: str
    ) -> Tuple[bool, str]:
        """
        Remove a company from the user's watchlist.

        Args:
            user_id: Telegram user identifier.
            symbol: Ticker symbol (will be uppercased).

        Returns:
            A tuple of (removed, message).
        """
        if not user_id or not symbol:
            logger.warning("remove_from_watchlist called with missing fields.")
            return False, "Invalid watchlist removal request."

        normalized_symbol = symbol.strip().upper()

        db = SessionLocal()
        try:
            deleted_count = (
                db.query(WatchlistEntry)
                .filter(
                    WatchlistEntry.user_id == str(user_id),
                    WatchlistEntry.symbol == normalized_symbol,
                )
                .delete()
            )
            db.commit()
            if deleted_count > 0:
                logger.info(
                    f"Removed {normalized_symbol} from watchlist for user '{user_id}'."
                )
                return True, f"{normalized_symbol} has been removed from your watchlist."
            logger.info(
                f"No-op remove: {normalized_symbol} not on watchlist for user '{user_id}'."
            )
            return False, f"{normalized_symbol} isn't on your watchlist."
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error removing from watchlist: {e}", exc_info=True)
            raise WatchlistError("Could not remove from watchlist.") from e
        finally:
            db.close()

    def get_watchlist(self, user_id: str) -> List[dict]:
        """
        Return the user's full watchlist, sorted by created_at ascending.

        Returns:
            A list of dicts with keys: symbol, company_name, created_at.
        """
        if not user_id:
            return []

        db = SessionLocal()
        try:
            entries = (
                db.query(WatchlistEntry)
                .filter(WatchlistEntry.user_id == str(user_id))
                .order_by(WatchlistEntry.created_at.asc())
                .all()
            )
            return [
                {
                    "symbol": e.symbol,
                    "company_name": e.company_name,
                    "created_at": e.created_at,
                }
                for e in entries
            ]
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching watchlist: {e}", exc_info=True)
            raise WatchlistError("Could not fetch watchlist.") from e
        finally:
            db.close()

    def is_in_watchlist(self, user_id: str, symbol: str) -> bool:
        """Return True if (user_id, symbol) is on the watchlist."""
        if not user_id or not symbol:
            return False

        normalized_symbol = symbol.strip().upper()
        db = SessionLocal()
        try:
            return (
                db.query(WatchlistEntry)
                .filter(
                    WatchlistEntry.user_id == str(user_id),
                    WatchlistEntry.symbol == normalized_symbol,
                )
                .first()
                is not None
            )
        except SQLAlchemyError as e:
            logger.error(f"Database error checking watchlist: {e}", exc_info=True)
            raise WatchlistError("Could not check watchlist.") from e
        finally:
            db.close()

    def clear_watchlist(self, user_id: str) -> int:
        """
        Remove every entry for a user. Returns the number of deleted rows.

        NOTE: This is NOT what /clear should call — /clear only clears
        conversation memory. This helper exists for explicit user-driven
        resets such as 'clear my watchlist' (future feature).
        """
        if not user_id:
            return 0

        db = SessionLocal()
        try:
            deleted = (
                db.query(WatchlistEntry)
                .filter(WatchlistEntry.user_id == str(user_id))
                .delete()
            )
            db.commit()
            logger.info(f"Cleared {deleted} watchlist entries for user '{user_id}'.")
            return deleted
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error clearing watchlist: {e}", exc_info=True)
            raise WatchlistError("Could not clear watchlist.") from e
        finally:
            db.close()

    def resolve_company_for_user(
        self,
        text: str,
        history: Optional[list] = None,
        default_name: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Reuse the company resolver used by Company Research and pair the
        ticker with a sensible display name.

        Returns:
            (ticker, company_name) on success, or None if not resolvable.
        """
        # Local import to avoid an import cycle at module load.
        from app.services.finance.company_resolver import resolve_company_ticker

        symbol = resolve_company_ticker(text, history=history)
        if not symbol:
            return None

        company_name = default_name or _ticker_to_default_name(symbol)
        return symbol, company_name


def _ticker_to_default_name(symbol: str) -> str:
    """Best-effort display name when no profile lookup was performed."""
    common = {
        "NVDA": "NVIDIA",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet (Google)",
        "GOOG": "Alphabet (Google)",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "AMD": "AMD",
        "NFLX": "Netflix",
        "INTC": "Intel",
        "PLTR": "Palantir",
        "UBER": "Uber",
        "DIS": "Walt Disney",
        "BA": "Boeing",
        "KO": "Coca-Cola",
        "PEP": "PepsiCo",
    }
    return common.get(symbol.upper(), symbol.upper())


# Singleton instance — handlers and the manager import this directly.
watchlist_service = WatchlistService()