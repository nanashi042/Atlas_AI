"""Focused persistence tests for Phase 2 Step 1 price-movement alerts."""

import unittest

from app.database.session import SessionLocal
from app.models.alert import Alert
from app.services.alert_service import AlertError, PRICE_CHANGE, alert_service


class TestAlertService(unittest.TestCase):
    USER = "alert_service_test_user"
    OTHER_USER = "alert_service_other_user"

    def setUp(self):
        self._clean()

    def tearDown(self):
        self._clean()

    def _clean(self):
        db = SessionLocal()
        try:
            db.query(Alert).filter(Alert.user_id.in_([self.USER, self.OTHER_USER])).delete(
                synchronize_session=False
            )
            db.commit()
        finally:
            db.close()

    def test_create_normalizes_symbol_and_lists_alert(self):
        created, alert = alert_service.create_alert(self.USER, "nvda", 5)
        self.assertTrue(created)
        self.assertEqual(alert.symbol, "NVDA")
        self.assertEqual(alert.alert_type, PRICE_CHANGE)
        self.assertTrue(alert.enabled)

        alerts = alert_service.list_alerts(self.USER)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["threshold_percentage"], 5.0)
        self.assertIsNone(alerts[0]["last_triggered_at"])

    def test_equivalent_active_alert_is_not_duplicated(self):
        alert_service.create_alert(self.USER, "NVDA", 5)
        created, _ = alert_service.create_alert(self.USER, "nvda", 5.0)
        self.assertFalse(created)
        self.assertTrue(alert_service.has_equivalent_alert(self.USER, "NVDA", 5))
        self.assertEqual(len(alert_service.list_alerts(self.USER)), 1)

    def test_disabled_alert_can_be_reactivated_without_duplicate(self):
        _, alert = alert_service.create_alert(self.USER, "NVDA", 5)
        self.assertTrue(alert_service.disable_alert(self.USER, alert.id))
        self.assertFalse(alert_service.has_equivalent_alert(self.USER, "NVDA", 5))

        created, reactivated = alert_service.create_alert(self.USER, "NVDA", 5)
        self.assertTrue(created)
        self.assertEqual(reactivated.id, alert.id)
        self.assertTrue(reactivated.enabled)
        self.assertEqual(len(alert_service.list_alerts(self.USER)), 1)

    def test_remove_and_user_isolation(self):
        _, alert = alert_service.create_alert(self.USER, "NVDA", 5)
        self.assertFalse(alert_service.remove_alert(self.OTHER_USER, alert.id))
        self.assertTrue(alert_service.remove_alert(self.USER, alert.id))
        self.assertEqual(alert_service.list_alerts(self.USER), [])

    def test_invalid_alert_inputs_are_rejected(self):
        with self.assertRaises(AlertError):
            alert_service.create_alert(self.USER, "NVDA", 0)
        with self.assertRaises(AlertError):
            alert_service.create_alert(self.USER, "NVDA", 5, "PRICE_TARGET")


if __name__ == "__main__":
    unittest.main()
