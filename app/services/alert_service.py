"""Persistent CRUD for Phase 2's initial PRICE_CHANGE alert type."""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database.session import SessionLocal, init_db
from app.models.alert import Alert

logger = logging.getLogger(__name__)

PRICE_CHANGE = "PRICE_CHANGE"


class AlertError(Exception):
    """Raised when alert data is invalid or cannot be persisted."""


class AlertService:
    """Owns user-scoped alert creation, listing, and lifecycle operations."""

    def __init__(self):
        init_db()

    def create_alert(
        self,
        user_id: str,
        symbol: str,
        threshold_percentage: float,
        alert_type: str = PRICE_CHANGE,
    ) -> tuple[bool, Alert]:
        """Create an enabled alert, or reactivate an equivalent disabled one."""
        normalized_symbol, normalized_threshold = self._validate_alert_input(
            user_id, symbol, threshold_percentage, alert_type
        )
        db = SessionLocal()
        try:
            existing = self._find_equivalent(db, user_id, normalized_symbol, normalized_threshold)
            if existing:
                if existing.enabled:
                    db.expunge(existing)
                    return False, existing
                existing.enabled = True
                existing.last_triggered_at = None
                existing.last_triggered_price = None
                existing.last_triggered_change_percentage = None
                db.commit()
                db.refresh(existing)
                db.expunge(existing)
                return True, existing

            alert = Alert(
                user_id=str(user_id),
                symbol=normalized_symbol,
                alert_type=PRICE_CHANGE,
                threshold_percentage=normalized_threshold,
                enabled=True,
            )
            db.add(alert)
            db.commit()
            db.refresh(alert)
            db.expunge(alert)
            return True, alert
        except IntegrityError as exc:
            db.rollback()
            raise AlertError("An equivalent alert already exists.") from exc
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not create alert for user '%s': %s", user_id, exc)
            raise AlertError("Could not create alert.") from exc
        finally:
            db.close()

    def list_alerts(self, user_id: str, enabled_only: bool = False) -> list[dict]:
        """Return a user's alerts ordered by creation time."""
        if not user_id:
            return []
        db = SessionLocal()
        try:
            query = db.query(Alert).filter(Alert.user_id == str(user_id))
            if enabled_only:
                query = query.filter(Alert.enabled.is_(True))
            return [self._to_dict(alert) for alert in query.order_by(Alert.created_at.asc()).all()]
        except SQLAlchemyError as exc:
            logger.error("Could not list alerts for user '%s': %s", user_id, exc)
            raise AlertError("Could not list alerts.") from exc
        finally:
            db.close()

    def list_active_alerts(self) -> list[dict]:
        """Return every enabled alert for the internal evaluator only."""
        db = SessionLocal()
        try:
            alerts = (
                db.query(Alert)
                .filter(Alert.enabled.is_(True))
                .order_by(Alert.created_at.asc())
                .all()
            )
            return [self._to_dict(alert) for alert in alerts]
        except SQLAlchemyError as exc:
            logger.error("Could not load active alerts: %s", exc)
            raise AlertError("Could not load active alerts.") from exc
        finally:
            db.close()

    def record_trigger(
        self, alert_id: int, current_price: float, change_percentage: float
    ) -> bool:
        """Persist delivery-deduplication state after an evaluator trigger."""
        db = SessionLocal()
        try:
            alert = db.query(Alert).filter(Alert.id == alert_id, Alert.enabled.is_(True)).one_or_none()
            if not alert:
                return False
            alert.last_triggered_at = datetime.now(timezone.utc)
            alert.last_triggered_price = float(current_price)
            alert.last_triggered_change_percentage = float(change_percentage)
            db.commit()
            return True
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not record trigger state for alert '%s': %s", alert_id, exc)
            raise AlertError("Could not record trigger state.") from exc
        finally:
            db.close()

    def clear_trigger_state(self, alert_id: int) -> bool:
        """Reset deduplication state once a price change has recovered below threshold."""
        db = SessionLocal()
        try:
            alert = db.query(Alert).filter(Alert.id == alert_id, Alert.enabled.is_(True)).one_or_none()
            if not alert:
                return False
            alert.last_triggered_at = None
            alert.last_triggered_price = None
            alert.last_triggered_change_percentage = None
            db.commit()
            return True
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not clear trigger state for alert '%s': %s", alert_id, exc)
            raise AlertError("Could not clear trigger state.") from exc
        finally:
            db.close()

    def has_equivalent_alert(
        self, user_id: str, symbol: str, threshold_percentage: float
    ) -> bool:
        """Return whether the user already has the same active PRICE_CHANGE alert."""
        normalized_symbol, normalized_threshold = self._validate_alert_input(
            user_id, symbol, threshold_percentage, PRICE_CHANGE
        )
        db = SessionLocal()
        try:
            alert = self._find_equivalent(db, user_id, normalized_symbol, normalized_threshold)
            return alert is not None and alert.enabled
        except SQLAlchemyError as exc:
            logger.error("Could not check alert equivalence for user '%s': %s", user_id, exc)
            raise AlertError("Could not check alert.") from exc
        finally:
            db.close()

    def disable_alert(self, user_id: str, alert_id: int) -> bool:
        """Disable one user-owned alert without deleting its delivery state."""
        db = SessionLocal()
        try:
            alert = self._find_owned_alert(db, user_id, alert_id)
            if not alert:
                return False
            alert.enabled = False
            db.commit()
            return True
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not disable alert '%s' for user '%s': %s", alert_id, user_id, exc)
            raise AlertError("Could not disable alert.") from exc
        finally:
            db.close()

    def remove_alert(self, user_id: str, alert_id: int) -> bool:
        """Permanently remove one user-owned alert."""
        db = SessionLocal()
        try:
            alert = self._find_owned_alert(db, user_id, alert_id)
            if not alert:
                return False
            db.delete(alert)
            db.commit()
            return True
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Could not remove alert '%s' for user '%s': %s", alert_id, user_id, exc)
            raise AlertError("Could not remove alert.") from exc
        finally:
            db.close()

    @staticmethod
    def _find_equivalent(db, user_id: str, symbol: str, threshold_percentage: float) -> Optional[Alert]:
        return (
            db.query(Alert)
            .filter(
                Alert.user_id == str(user_id),
                Alert.symbol == symbol,
                Alert.alert_type == PRICE_CHANGE,
                Alert.threshold_percentage == threshold_percentage,
            )
            .one_or_none()
        )

    @staticmethod
    def _find_owned_alert(db, user_id: str, alert_id: int) -> Optional[Alert]:
        return (
            db.query(Alert)
            .filter(Alert.id == alert_id, Alert.user_id == str(user_id))
            .one_or_none()
        )

    @staticmethod
    def _validate_alert_input(user_id, symbol, threshold_percentage, alert_type) -> tuple[str, float]:
        if not user_id or not symbol or not str(symbol).strip():
            raise AlertError("user_id and symbol are required.")
        if alert_type != PRICE_CHANGE:
            raise AlertError("Only PRICE_CHANGE alerts are supported.")
        try:
            threshold = float(threshold_percentage)
        except (TypeError, ValueError) as exc:
            raise AlertError("threshold_percentage must be a positive number.") from exc
        if not math.isfinite(threshold) or threshold <= 0:
            raise AlertError("threshold_percentage must be a positive number.")
        return symbol.strip().upper(), threshold

    @staticmethod
    def _to_dict(alert: Alert) -> dict:
        return {
            "id": alert.id,
            "user_id": alert.user_id,
            "symbol": alert.symbol,
            "alert_type": alert.alert_type,
            "threshold_percentage": alert.threshold_percentage,
            "enabled": alert.enabled,
            "last_triggered_at": alert.last_triggered_at,
            "last_triggered_price": alert.last_triggered_price,
            "last_triggered_change_percentage": alert.last_triggered_change_percentage,
            "created_at": alert.created_at,
            "updated_at": alert.updated_at,
        }


alert_service = AlertService()
