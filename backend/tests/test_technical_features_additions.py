import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import duckdb

from backend.db.schema import apply_core_schema
from backend.features.technical_features import compute_technical_features


def _insert_hourly_candles(con, symbol: str, closes, start_ts: datetime, interval: str = "1h") -> None:
    rows = []
    for idx, close in enumerate(closes):
        ts = start_ts + timedelta(hours=idx)
        c = float(close)
        rows.append((symbol, interval, ts, c, c + 1.0, c - 1.0, c, 100.0))
    con.executemany(
        """
        INSERT INTO candles(symbol, interval, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


class TestTechnicalFeatureAdditions(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect(":memory:")
        apply_core_schema(self.con)

    def tearDown(self) -> None:
        self.con.close()

    def test_keltner_and_squeeze_fields_present(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        _insert_hourly_candles(self.con, "BTC-USD", [100.0] * 90, start)

        with patch(
            "backend.features.technical_features.get_settings",
            return_value=SimpleNamespace(paper_symbols=["BTC-USD"]),
        ):
            features = compute_technical_features("BTC-USD", self.con, interval="1h")

        vol = features.get("volatility") or {}
        self.assertIsNotNone(vol.get("kc_mid_20"))
        self.assertIsNotNone(vol.get("kc_upper_20_15"))
        self.assertIsNotNone(vol.get("kc_lower_20_15"))
        self.assertIsNotNone(vol.get("bb_kc_compression_ratio"))
        self.assertIsNotNone(vol.get("squeeze_on"))
        self.assertIn(vol.get("squeeze_level"), {"high", "medium", "off"})
        self.assertIsInstance(vol.get("squeeze_fired"), (bool, type(None)))

    def test_squeeze_fired_has_boolean_state(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        closes = [100.0] * 89 + [160.0]
        _insert_hourly_candles(self.con, "BTC-USD", closes, start)

        with patch(
            "backend.features.technical_features.get_settings",
            return_value=SimpleNamespace(paper_symbols=["BTC-USD"]),
        ):
            features = compute_technical_features("BTC-USD", self.con, interval="1h")

        vol = features.get("volatility") or {}
        self.assertIsInstance(vol.get("squeeze_on"), (bool, type(None)))
        self.assertIsInstance(vol.get("squeeze_fired"), (bool, type(None)))

    def test_vwap_fields_are_computed(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        closes = [float(100 + i) for i in range(220)]
        _insert_hourly_candles(self.con, "BTC-USD", closes, start)

        with patch(
            "backend.features.technical_features.get_settings",
            return_value=SimpleNamespace(paper_symbols=["BTC-USD"]),
        ):
            features = compute_technical_features("BTC-USD", self.con, interval="1h")

        volume = features.get("volume") or {}
        self.assertIsNotNone(volume.get("vwap_24h"))
        self.assertIsNotNone(volume.get("vwap_7d"))
        self.assertIsInstance(volume.get("price_above_vwap"), bool)
        self.assertIsNotNone(volume.get("vwap_distance_pct"))

        expected_vwap_24h = sum(closes[-24:]) / 24.0
        self.assertAlmostEqual(float(volume.get("vwap_24h")), expected_vwap_24h, places=6)
        self.assertTrue(bool(volume.get("price_above_vwap")))

    def test_relative_strength_and_breadth_fields(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        steps = 220
        btc = [100.0 + (0.25 * i) for i in range(steps)]
        eth = [100.0 + (0.40 * i) for i in range(steps)]
        xrp = [100.0 + (0.10 * i) for i in range(steps)]
        _insert_hourly_candles(self.con, "BTC-USD", btc, start)
        _insert_hourly_candles(self.con, "ETH-USD", eth, start)
        _insert_hourly_candles(self.con, "XRP-USD", xrp, start)

        with patch(
            "backend.features.technical_features.get_settings",
            return_value=SimpleNamespace(paper_symbols=["BTC-USD", "ETH-USD", "XRP-USD"]),
        ):
            features = compute_technical_features("ETH-USD", self.con, interval="1h")

        rs = features.get("relative_strength") or {}
        breadth = features.get("breadth") or {}
        self.assertIsNotNone(rs.get("rs_vs_btc_7d"))
        self.assertGreater(float(rs.get("rs_vs_btc_7d")), 0.0)
        self.assertIsNotNone(rs.get("rs_rank"))
        self.assertIsNotNone(rs.get("rs_percentile"))
        self.assertIsNotNone(breadth.get("pct_symbols_above_ema20"))
        self.assertIsNotNone(breadth.get("pct_symbols_bullish_ema_cross"))
        self.assertIsNotNone(breadth.get("breadth_score"))
        self.assertGreaterEqual(float(breadth.get("breadth_score")), 0.0)
        self.assertLessEqual(float(breadth.get("breadth_score")), 1.0)


if __name__ == "__main__":
    unittest.main()
