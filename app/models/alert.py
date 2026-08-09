"""Persistent user alerts for the next phase of Atlas AI."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Float, Integer, String, UniqueConstraint

from app.database.session import Base


class Alert(Base):
    """One user-defined market alert; Phase 2 initially supports PRICE_CHANGE only."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    symbol = Column(String, nullable=False)
    alert_type = Column(String, nullable=False, default="PRICE_CHANGE")
    threshold_percentage = Column(Float, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)

    # Durable state for a later alert evaluator to avoid repeat delivery for
    # the same observed movement. No market data is read in this phase.
    last_triggered_at = Column(DateTime, nullable=True)
    last_triggered_price = Column(Float, nullable=True)
    last_triggered_change_percentage = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint("alert_type = 'PRICE_CHANGE'", name="ck_alert_price_change_type"),
        CheckConstraint("threshold_percentage > 0", name="ck_alert_positive_threshold"),
        UniqueConstraint(
            "user_id", "symbol", "alert_type", "threshold_percentage",
            name="uq_alert_equivalent",
        ),
    )
