"""Telegram-free evaluation for persistent PRICE_CHANGE alerts."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from app.services.alert_service import PRICE_CHANGE, alert_service
from app.services.finance.company_research import CompanyResearchService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggeredAlert:
    """A structured alert event ready for a later notification step."""

    alert_id: int
    user_id: str
    symbol: str
    current_price: float
    previous_close: float
    percentage_change: float
    threshold_percentage: float
    reason: str


class AlertEvaluator:
    """Evaluates enabled PRICE_CHANGE alerts using existing company research."""

    def __init__(self, research_service: Optional[CompanyResearchService] = None, alerts=None):
        self.research_service = research_service or CompanyResearchService()
        self.alerts = alerts or alert_service

    async def evaluate_active_alerts(self) -> list[TriggeredAlert]:
        """Evaluate all active alerts and return only newly triggered events.

        Percentage movement is calculated from Finnhub's current price and
        previous close: ``(current - previous_close) / previous_close * 100``.
        Finance failures and malformed quotes skip only their affected symbol.
        """
        active_alerts = self.alerts.list_active_alerts()
        alerts_by_symbol: dict[str, list[dict]] = {}
        for alert in active_alerts:
            if alert.get("alert_type") != PRICE_CHANGE:
                continue
            alerts_by_symbol.setdefault(alert["symbol"], []).append(alert)

        triggered: list[TriggeredAlert] = []
        for symbol, symbol_alerts in alerts_by_symbol.items():
            try:
                research = await self.research_service.get_company_research(symbol)
                current_price = research.current_price
                previous_close = research.previous_close
                change = self.calculate_percentage_change(current_price, previous_close)
                if change is None:
                    logger.warning("Skipping alerts for '%s': invalid current price or previous close.", symbol)
                    continue
            except Exception as exc:
                logger.warning("Skipping alerts for '%s': market data unavailable (%s).", symbol, exc)
                continue

            for alert in symbol_alerts:
                threshold = alert["threshold_percentage"]
                if abs(change) < threshold:
                    # A recovered price movement begins a new future event.
                    if alert.get("last_triggered_at") is not None:
                        self.alerts.clear_trigger_state(alert["id"])
                    continue
                if not self._is_distinct_trigger(alert, change):
                    continue

                self.alerts.record_trigger(alert["id"], current_price, change)
                triggered.append(
                    TriggeredAlert(
                        alert_id=alert["id"],
                        user_id=alert["user_id"],
                        symbol=symbol,
                        current_price=current_price,
                        previous_close=previous_close,
                        percentage_change=change,
                        threshold_percentage=threshold,
                        reason=(
                            f"{symbol} moved {change:+.2f}% from its previous close, "
                            f"meeting the {threshold:g}% PRICE_CHANGE threshold."
                        ),
                    )
                )
        return triggered

    @staticmethod
    def calculate_percentage_change(current_price, previous_close) -> Optional[float]:
        """Return current-vs-previous-close percentage change, or None if invalid."""
        try:
            current = float(current_price)
            previous = float(previous_close)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(current) or not math.isfinite(previous) or previous <= 0:
            return None
        return ((current - previous) / previous) * 100

    @staticmethod
    def _is_distinct_trigger(alert: dict, change: float) -> bool:
        """Suppress the same breach until it reverses or grows by one threshold.

        A movement that falls below the threshold clears state in the caller.
        While it remains beyond the threshold, a direction reversal or another
        full threshold percentage-point extension is treated as a new event.
        """
        previous_change = alert.get("last_triggered_change_percentage")
        if previous_change is None:
            return True
        previous_change = float(previous_change)
        if (change > 0) != (previous_change > 0):
            return True
        return abs(change - previous_change) >= alert["threshold_percentage"]


alert_evaluator = AlertEvaluator()
