import unittest
from datetime import datetime, timedelta, timezone

import duckdb

from backend.db.schema import apply_core_schema
from backend.features.fundamentals_features import compute_fundamentals_features
from backend.services.fundamentals_ingest import ingest_fundamentals


def _to_float(value):
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


class TestFundamentalsHistoryFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect(":memory:")
        apply_core_schema(self.con)

    def tearDown(self) -> None:
        self.con.close()

    def test_ingest_writes_latest_and_history(self):
        symbol = "BTC-USD"
        payload_1 = {
            "price": 100.0,
            "mkt_cap": 1000.0,
            "vol_24h": 100.0,
            "vol_mcap_ratio": 0.10,
        }
        payload_2 = {
            "price": 110.0,
            "mkt_cap": 1200.0,
            "vol_24h": 140.0,
            "vol_mcap_ratio": 0.12,
        }
        written_1 = ingest_fundamentals(self.con, symbol, payload_1)
        written_2 = ingest_fundamentals(self.con, symbol, payload_2)

        latest_count = self.con.execute(
            "SELECT COUNT(*) FROM fundamentals WHERE symbol = ?",
            [symbol],
        ).fetchone()[0]
        history_count = self.con.execute(
            "SELECT COUNT(*) FROM fundamentals_history WHERE symbol = ?",
            [symbol],
        ).fetchone()[0]

        self.assertEqual(latest_count, written_2)
        self.assertGreaterEqual(history_count, written_1 + written_2)

    def test_history_derived_fields_are_computed(self):
        symbol = "BTC-USD"
        now = datetime(2026, 2, 1, tzinfo=timezone.utc)

        latest_rows = [
            (symbol, now, "price", 100.0),
            (symbol, now, "mkt_cap", 1000.0),
            (symbol, now, "fdv", 1250.0),
            (symbol, now, "vol_24h", 240.0),
            (symbol, now, "vol_mcap_ratio", 0.24),
            (symbol, now, "mktcap_fdv_ratio", 0.8),
            (symbol, now, "dominance_pct", 52.0),
            (symbol, now, "liquidity_score", 0.8),
            (symbol, now, "cg_score", 0.7),
            (symbol, now, "dev_score", 0.6),
            (symbol, now, "community_score", 0.65),
        ]
        self.con.executemany(
            "INSERT INTO fundamentals(symbol, ts, key, value) VALUES (?, ?, ?, ?)",
            latest_rows,
        )

        hist_rows = [
            (symbol, now - timedelta(days=31), "mkt_cap", 800.0),
            (symbol, now - timedelta(days=8), "mkt_cap", 900.0),
            (symbol, now, "mkt_cap", 1000.0),
            (symbol, now - timedelta(days=8), "vol_24h", 180.0),
            (symbol, now, "vol_24h", 240.0),
            (symbol, now - timedelta(days=29), "vol_mcap_ratio", 0.10),
            (symbol, now - timedelta(days=24), "vol_mcap_ratio", 0.12),
            (symbol, now - timedelta(days=19), "vol_mcap_ratio", 0.16),
            (symbol, now - timedelta(days=14), "vol_mcap_ratio", 0.18),
            (symbol, now - timedelta(days=9), "vol_mcap_ratio", 0.22),
            (symbol, now, "vol_mcap_ratio", 0.24),
        ]
        self.con.executemany(
            "INSERT INTO fundamentals_history(symbol, ts, key, value) VALUES (?, ?, ?, ?)",
            hist_rows,
        )

        features = compute_fundamentals_features(symbol, self.con)
        self.assertIn("mkt_cap_change_7d", features)
        self.assertIn("mkt_cap_change_30d", features)
        self.assertIn("vol_24h_change_7d", features)
        self.assertIn("vol_mcap_ratio_z_30d", features)
        self.assertIn("fundamental_trend_score", features)
        self.assertIn("dominance_pct", features)
        self.assertIn("liquidity_score", features)
        self.assertIn("cg_score", features)
        self.assertIn("dev_score", features)
        self.assertIn("community_score", features)
        self.assertIn("mktcap_fdv_ratio", features)

        self.assertAlmostEqual(_to_float(features["mkt_cap_change_7d"]), (1000.0 - 900.0) / 900.0, places=6)
        self.assertAlmostEqual(_to_float(features["mkt_cap_change_30d"]), (1000.0 - 800.0) / 800.0, places=6)
        self.assertAlmostEqual(_to_float(features["vol_24h_change_7d"]), (240.0 - 180.0) / 180.0, places=6)
        self.assertIsNotNone(_to_float(features["vol_mcap_ratio_z_30d"]))
        self.assertIsNotNone(_to_float(features["fundamental_trend_score"]))


if __name__ == "__main__":
    unittest.main()
