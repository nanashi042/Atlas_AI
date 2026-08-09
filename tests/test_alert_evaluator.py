"""Unit tests for Phase 2 Step 3 alert evaluation without network access."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.alert_evaluator import AlertEvaluator
from app.services.alert_service import PRICE_CHANGE


def _alert(alert_id, user_id, symbol, threshold, **overrides):
    alert = {
        "id": alert_id,
        "user_id": user_id,
        "symbol": symbol,
        "alert_type": PRICE_CHANGE,
        "threshold_percentage": threshold,
        "enabled": True,
        "last_triggered_at": None,
        "last_triggered_price": None,
        "last_triggered_change_percentage": None,
    }
    alert.update(overrides)
    return alert


class FakeAlertService:
    def __init__(self, alerts):
        self.alerts = alerts
        self.recorded = []
        self.cleared = []

    def list_active_alerts(self):
        return [alert for alert in self.alerts if alert["enabled"]]

    def record_trigger(self, alert_id, price, change):
        self.recorded.append((alert_id, price, change))
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["last_triggered_at"] = object()
                alert["last_triggered_price"] = price
                alert["last_triggered_change_percentage"] = change
        return True

    def clear_trigger_state(self, alert_id):
        self.cleared.append(alert_id)
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["last_triggered_at"] = None
                alert["last_triggered_price"] = None
                alert["last_triggered_change_percentage"] = None
        return True


class TestAlertEvaluator(unittest.IsolatedAsyncioTestCase):
    def _evaluator(self, alerts, results):
        research = AsyncMock()

        async def get_research(symbol):
            result = results[symbol]
            if isinstance(result, Exception):
                raise result
            return result

        research.get_company_research.side_effect = get_research
        store = FakeAlertService(alerts)
        return AlertEvaluator(research_service=research, alerts=store), store, research

    @staticmethod
    def _quote(current, previous):
        return SimpleNamespace(current_price=current, previous_close=previous)

    async def test_positive_negative_and_exact_threshold_movements_trigger(self):
        alerts = [_alert(1, "a", "NVDA", 5), _alert(2, "b", "TSLA", 5), _alert(3, "c", "AMD", 5)]
        evaluator, _, _ = self._evaluator(alerts, {
            "NVDA": self._quote(105, 100), "TSLA": self._quote(94, 100), "AMD": self._quote(105, 100),
        })
        triggered = await evaluator.evaluate_active_alerts()
        self.assertEqual([event.alert_id for event in triggered], [1, 2, 3])
        self.assertEqual(triggered[0].percentage_change, 5.0)
        self.assertEqual(triggered[1].percentage_change, -6.0)

    async def test_below_threshold_is_not_triggered(self):
        evaluator, store, _ = self._evaluator([_alert(1, "a", "NVDA", 5)], {"NVDA": self._quote(104, 100)})
        self.assertEqual(await evaluator.evaluate_active_alerts(), [])
        self.assertEqual(store.recorded, [])

    async def test_duplicate_breach_is_suppressed_and_alert_remains_enabled(self):
        alert = _alert(1, "a", "NVDA", 5)
        evaluator, store, _ = self._evaluator([alert], {"NVDA": self._quote(106, 100)})
        self.assertEqual(len(await evaluator.evaluate_active_alerts()), 1)
        self.assertEqual(await evaluator.evaluate_active_alerts(), [])
        self.assertTrue(alert["enabled"])
        self.assertEqual(len(store.recorded), 1)

    async def test_recovery_and_later_distinct_movement_can_trigger_again(self):
        alert = _alert(1, "a", "NVDA", 5, last_triggered_at=object(), last_triggered_change_percentage=5.0)
        evaluator, store, research = self._evaluator([alert], {"NVDA": self._quote(102, 100)})
        self.assertEqual(await evaluator.evaluate_active_alerts(), [])
        self.assertEqual(store.cleared, [1])
        research.get_company_research.side_effect = lambda symbol: self._quote(106, 100)
        # AsyncMock side effects must be awaitable for the second call.
        async def renewed(symbol):
            return self._quote(106, 100)
        research.get_company_research.side_effect = renewed
        self.assertEqual(len(await evaluator.evaluate_active_alerts()), 1)

    async def test_multiple_users_symbols_and_failure_are_isolated(self):
        alerts = [_alert(1, "user_a", "NVDA", 5), _alert(2, "user_b", "TSLA", 5), _alert(3, "user_c", "AMD", 5)]
        evaluator, _, research = self._evaluator(alerts, {
            "NVDA": self._quote(106, 100), "TSLA": RuntimeError("provider down"), "AMD": self._quote(94, 100),
        })
        triggered = await evaluator.evaluate_active_alerts()
        self.assertEqual({event.user_id for event in triggered}, {"user_a", "user_c"})
        self.assertEqual(research.get_company_research.await_count, 3)

    async def test_invalid_market_data_and_inactive_alert_are_ignored(self):
        inactive = _alert(2, "b", "TSLA", 5, enabled=False)
        evaluator, store, research = self._evaluator(
            [_alert(1, "a", "NVDA", 5), inactive], {"NVDA": self._quote(None, 100)}
        )
        self.assertEqual(await evaluator.evaluate_active_alerts(), [])
        research.get_company_research.assert_awaited_once_with("NVDA")
        self.assertEqual(store.recorded, [])

    def test_percentage_calculation(self):
        self.assertEqual(AlertEvaluator.calculate_percentage_change(105, 100), 5.0)
        self.assertEqual(AlertEvaluator.calculate_percentage_change(94, 100), -6.0)
        self.assertIsNone(AlertEvaluator.calculate_percentage_change(100, 0))

