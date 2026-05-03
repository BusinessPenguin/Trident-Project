import unittest
from copy import deepcopy
from unittest.mock import patch

from backend.features.phase4_analysis import _confidence_block, interpret_snapshot


def _base_snapshot():
    return {
        "symbol": "BTC-USD",
        "technical_features": {
            "ema_cross": "bullish",
            "trend_strength": 0.02,
            "atr_pct": 0.01,
            "ret_3d": 0.01,
            "ret_7d": 0.02,
            "rsi14": 55.0,
            "macd_hist": 10.0,
            "vol_regime": "normal",
        },
        "news_features": {
            "direction": "neutral",
            "intensity": 0.10,
            "weighted_bullish": 6.0,
            "weighted_bearish": 5.0,
            "article_count": 30,
            "window_hours": 48,
        },
        "fundamental_features": {
            "pct_change_24h": 0.2,
            "mcap_change_1d": 0.2,
        },
        "meta_features": {
            "agreement_score": 0.80,
            "freshness": {
                "tech_avg_age_hours": 1.0,
                "news_avg_age_hours": 1.0,
                "fundamentals_avg_age_hours": 1.0,
                "news_count_used": 30,
            },
        },
        "macro_calendar": {},
        "macro_calendar_raw": {},
        "macro_sentiment": {
            "fear_greed": {"value": 50, "label": "Neutral", "age_hours": 1.0}
        },
        "macro_liquidity": {},
        "news_summaries": [],
        "macro_summaries": [],
    }


def _base_scenarios():
    return {
        "best_case": {"likelihood": 0.60},
        "base_case": {"likelihood": 0.30},
        "worst_case": {"likelihood": 0.10},
    }


class TestPhase4HorizonAlignment(unittest.TestCase):
    def test_confidence_agreement_blends_with_horizon_alignment(self):
        snapshot = _base_snapshot()
        snapshot["technical_features"]["multi_horizon"] = {
            "windows": ["6h", "2d", "14d"],
            "horizon_signals": {
                "6h": {"trend_bias": "bullish", "composite_strength": 0.94},
                "2d": {"trend_bias": "bullish", "composite_strength": 0.715},
                "14d": {"trend_bias": "bearish", "composite_strength": 0.285},
            },
        }
        conf = _confidence_block(snapshot, _base_scenarios(), freshness_score=0.80)
        # alignment_raw = abs(0.94 + 0.715 - 0.285) / (0.94 + 0.715 + 0.285) = 1.37 / 1.94
        alignment_raw = 1.37 / 1.94
        expected = round(min(1.0, max(0.0, 0.70 * 0.80 + 0.30 * alignment_raw)), 3)
        self.assertAlmostEqual(conf["components"]["agreement"], expected, places=3)
        self.assertGreaterEqual(conf["components"]["agreement"], 0.0)
        self.assertLessEqual(conf["components"]["agreement"], 1.0)

    def test_conflict_adds_horizon_conflict_penalty(self):
        snapshot = _base_snapshot()
        snapshot["technical_features"]["multi_horizon"] = {
            "windows": ["6h", "2d", "14d"],
            "horizon_signals": {
                "6h": {"trend_bias": "bullish", "composite_strength": 1.0},
                "2d": {"trend_bias": "bearish", "composite_strength": 1.0},
                "14d": {"trend_bias": "neutral", "composite_strength": 1.0},
            },
        }
        conf = _confidence_block(snapshot, _base_scenarios(), freshness_score=0.80)
        penalties = conf.get("penalties", {}).get("signal_consistency", [])
        codes = [p.get("code") for p in penalties]
        self.assertIn("HORIZON_CONFLICT", codes)

    @patch("backend.features.phase4_analysis._call_openai_for_scenarios", return_value={})
    def test_interpret_snapshot_present_vs_missing_multi_horizon(self, _mock_openai):
        snapshot_with = _base_snapshot()
        snapshot_with["technical_features"]["multi_horizon"] = {
            "windows": ["6h", "2d", "14d"],
            "horizon_signals": {
                "6h": {"bars": 6, "trend_bias": "bullish", "composite_strength": 0.94},
                "2d": {"bars": 48, "trend_bias": "bullish", "composite_strength": 0.715},
                "14d": {"bars": 336, "trend_bias": "bearish", "composite_strength": 0.285},
            },
        }
        out_with = interpret_snapshot(deepcopy(snapshot_with), con=None)
        ad_with = out_with.get("agreement_detail", {})
        shadow_with = out_with.get("shadow_intelligence", {}) or {}
        self.assertIn("horizon_alignment_score", ad_with)
        self.assertIn("horizon_alignment_label", ad_with)
        self.assertIn("horizon_dominant_bias", ad_with)
        self.assertIn("multi_horizon", ad_with)
        self.assertIn("windows", ad_with["multi_horizon"])
        self.assertIn("horizon_signals", ad_with["multi_horizon"])
        self.assertGreaterEqual(float(ad_with["horizon_alignment_score"]), 0.0)
        self.assertLessEqual(float(ad_with["horizon_alignment_score"]), 1.0)
        self.assertEqual(shadow_with.get("stage"), "shadow")
        self.assertIn(shadow_with.get("status"), {"ok", "unavailable", "error"})

        snapshot_without = _base_snapshot()
        out_without = interpret_snapshot(deepcopy(snapshot_without), con=None)
        ad_without = out_without.get("agreement_detail", {})
        shadow_without = out_without.get("shadow_intelligence", {}) or {}
        self.assertNotIn("horizon_alignment_score", ad_without)
        self.assertNotIn("horizon_alignment_label", ad_without)
        self.assertNotIn("horizon_dominant_bias", ad_without)
        self.assertNotIn("multi_horizon", ad_without)
        self.assertEqual(shadow_without.get("stage"), "shadow")
        self.assertIn(shadow_without.get("status"), {"ok", "unavailable", "error"})
        penalties_without = (
            out_without.get("confidence", {})
            .get("penalties", {})
            .get("signal_consistency", [])
        )
        self.assertFalse(any(p.get("code") == "HORIZON_CONFLICT" for p in penalties_without))

    def test_confidence_adds_new_market_structure_penalties(self):
        snapshot = _base_snapshot()
        snapshot["technical_features"]["breadth_score"] = 0.30
        snapshot["technical_features"]["rs_vs_btc_7d"] = -0.04
        snapshot["technical_features"]["squeeze_on"] = True
        snapshot["technical_features"]["vwap_distance_pct"] = 0.002
        conf = _confidence_block(snapshot, _base_scenarios(), freshness_score=0.80)
        market_penalties = conf.get("penalties", {}).get("market_structure", [])
        codes = [p.get("code") for p in market_penalties]
        self.assertIn("WEAK_BREADTH", codes)
        self.assertIn("RELATIVE_WEAKNESS", codes)
        self.assertIn("PRE_BREAKOUT_COMPRESSION", codes)


if __name__ == "__main__":
    unittest.main()
