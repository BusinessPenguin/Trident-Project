import unittest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import duckdb

from backend.decide.gates import compute_gate_thresholds
from backend.decide.trident_decision import (
    ALLOWED_ACTIONS,
    ALLOWED_REGIMES,
    _build_hold_escalation_state,
    _select_hypothetical_action,
    _validity_hours,
    build_decision_output,
    classify_regime,
    compute_deadband,
)


def _snapshot() -> dict:
    return {
        "symbol": "BTC-USD",
        "technical_features": {
            "ema_cross": "bearish",
            "trend_strength": 0.05,
            "adx14": 30.0,
            "plus_di14": 12.0,
            "minus_di14": 27.0,
            "vol_regime": "high",
            "atr_pct": 0.015,
            "rv_24h": 0.010,
            "rv_7d": 0.009,
            "bb_bandwidth_20_2": 0.10,
            "multi_horizon": {
                "windows": ["6h", "2d", "14d"],
                "horizon_signals": {
                    "6h": {"bars": 6, "trend_bias": "bearish", "momentum_state": "bearish", "vol_state": "high", "composite_strength": 0.80},
                    "2d": {"bars": 48, "trend_bias": "bearish", "momentum_state": "neutral", "vol_state": "high", "composite_strength": 0.70},
                    "14d": {"bars": 336, "trend_bias": "bearish", "momentum_state": "neutral", "vol_state": "normal", "composite_strength": 0.55},
                },
            },
        },
        "news_features": {
            "direction": "bearish",
            "intensity": 0.35,
            "article_count": 30,
            "weighted_bullish": 5.0,
            "weighted_bearish": 20.0,
            "window_hours": 48,
        },
        "fundamental_features": {
            "pct_change_24h": -1.2,
            "mcap_change_1d": -1.2,
            "vol_mcap_ratio": 0.07,
        },
        "meta_features": {
            "agreement_score": 0.62,
            "freshness": {
                "tech_avg_age_hours": 2.0,
                "news_avg_age_hours": 8.0,
                "fundamentals_avg_age_hours": 2.0,
                "news_count_used": 30,
            },
        },
        "macro_calendar": {
            "next_high_event_in_hours": 8.0,
            "next_high_event_title": "Fed Balance Sheet",
            "directional_evidence": {"derived_bias": "neutral"},
        },
        "macro_calendar_raw": {
            "top_recent": [],
            "top_upcoming": [],
        },
        "macro_sentiment": {"fear_greed": {"value": 35, "label": "Fear", "age_hours": 1.0}},
        "macro_liquidity": {},
        "news_summaries": [],
        "macro_summaries": [],
    }


class TestPhase5Decision(unittest.TestCase):
    def test_validity_hours_swing_mapping(self):
        self.assertEqual(_validity_hours("high", "range", "normal"), 24)
        self.assertEqual(_validity_hours("medium", "range", "normal"), 48)
        self.assertEqual(_validity_hours("low", "range", "normal"), 96)
        self.assertEqual(_validity_hours("high", "bull_trend", "elevated"), 24)
        self.assertGreaterEqual(_validity_hours("low", "high_vol_chop", "elevated"), 24)
        self.assertLessEqual(_validity_hours("low", "high_vol_chop", "elevated"), 168)

    def test_contextual_14d_soft_conflict_in_range_demotes_when_weak(self):
        thresholds = compute_gate_thresholds("range")
        action = _select_hypothetical_action(
            dominant_bias="bullish",
            longer_bias="bearish",
            regime_label="range",
            best_likelihood=0.31,
            worst_likelihood=0.29,
            agreement_score=0.53,
            margin=0.061,
            thresholds=thresholds,
        )
        self.assertEqual(action, "NO_TRADE")

    def test_contextual_14d_hard_veto_in_trend(self):
        thresholds = compute_gate_thresholds("bull_trend")
        action = _select_hypothetical_action(
            dominant_bias="bullish",
            longer_bias="bearish",
            regime_label="bull_trend",
            best_likelihood=0.80,
            worst_likelihood=0.10,
            agreement_score=0.90,
            margin=0.50,
            thresholds=thresholds,
        )
        self.assertEqual(action, "NO_TRADE")

    def test_neutral_fallback_gate_coupled(self):
        thresholds = compute_gate_thresholds("range")
        action = _select_hypothetical_action(
            dominant_bias="neutral",
            longer_bias="neutral",
            regime_label="range",
            best_likelihood=0.40,
            worst_likelihood=0.20,
            agreement_score=0.50,
            margin=0.10,
            thresholds=thresholds,
        )
        self.assertEqual(action, "NO_TRADE")

    def test_classify_regime_bounds(self):
        snap = _snapshot()
        tech = snap["technical_features"]
        agreement_detail = {"agreement_score": 0.60, "horizon_alignment_score": 0.75, "horizon_dominant_bias": "bearish"}
        regime = classify_regime(tech, agreement_detail, tech.get("multi_horizon"))
        self.assertIn(regime["label"], ALLOWED_REGIMES)
        self.assertGreaterEqual(float(regime["confidence"]), 0.0)
        self.assertLessEqual(float(regime["confidence"]), 1.0)

    def test_deadband_hard_blocker_forces_active(self):
        deadband = compute_deadband(
            confidence_overall=0.20,
            agreement_score=0.95,
            horizon_alignment_score=0.95,
            margin=0.20,
            event_risk="normal",
            regime={"label": "range", "is_high_vol": False},
        )
        self.assertTrue(deadband["active"])
        self.assertEqual(deadband.get("activation_basis"), "hard_blocker")
        codes = [r.get("code") for r in deadband.get("hard_blockers", [])]
        self.assertIn("CRITICAL_LOW_CONFIDENCE", codes)

    def test_deadband_weighted_threshold_triggers_without_hard_blockers(self):
        deadband = compute_deadband(
            confidence_overall=0.37,
            agreement_score=0.20,
            horizon_alignment_score=0.00,
            margin=0.00,
            event_risk="elevated",
            regime={"label": "high_vol_chop", "is_high_vol": True},
            gate_overrides={"penalty_threshold": 0.40},
        )
        self.assertTrue(deadband["active"])
        self.assertEqual(deadband.get("hard_blockers"), [])
        self.assertEqual(deadband.get("activation_basis"), "penalty_threshold")
        self.assertGreaterEqual(
            float(deadband.get("penalty_total") or 0.0),
            float(deadband.get("penalty_threshold") or 0.0),
        )

    def test_deadband_weighted_below_threshold_is_inactive(self):
        deadband = compute_deadband(
            confidence_overall=0.32,
            agreement_score=0.51,
            horizon_alignment_score=0.59,
            margin=0.055,
            event_risk="normal",
            regime={"label": "range", "is_high_vol": False},
        )
        self.assertFalse(deadband["active"])
        self.assertEqual(deadband.get("hard_blockers"), [])
        self.assertEqual(deadband.get("activation_basis"), "none")
        self.assertLess(
            float(deadband.get("penalty_total") or 0.0),
            float(deadband.get("penalty_threshold") or 0.0),
        )
        self.assertGreater(len(deadband.get("reasons", [])), 0)

    def test_deadband_threshold_boundaries_pass_at_exact_thresholds(self):
        deadband = compute_deadband(
            confidence_overall=0.33,
            agreement_score=0.52,
            horizon_alignment_score=0.60,
            margin=0.06,
            event_risk="normal",
            regime={"label": "range", "is_high_vol": False},
        )
        self.assertFalse(deadband["active"])
        self.assertEqual(deadband.get("reasons", []), [])
        self.assertEqual(deadband.get("effective_confidence"), 0.33)

    def test_gate_threshold_override_hook(self):
        deadband = compute_deadband(
            confidence_overall=0.39,
            agreement_score=0.90,
            horizon_alignment_score=0.90,
            margin=0.20,
            event_risk="normal",
            regime={"label": "range", "is_high_vol": False},
            gate_overrides={"min_confidence": 0.40},
        )
        self.assertFalse(deadband["active"])
        self.assertEqual(deadband["thresholds"]["min_confidence"], 0.4)
        self.assertEqual(deadband.get("activation_basis"), "none")
        reasons = [r.get("code") for r in deadband.get("reasons", [])]
        self.assertIn("LOW_CONFIDENCE", reasons)

    def test_deadband_context_reasons_can_trigger(self):
        deadband = compute_deadband(
            confidence_overall=0.60,
            agreement_score=0.90,
            horizon_alignment_score=0.90,
            margin=0.20,
            event_risk="normal",
            regime={"label": "range", "is_high_vol": False},
            context={
                "breadth_score": 0.30,
                "rs_vs_btc_7d": -0.05,
                "squeeze_on": True,
                "vwap_distance_pct": 0.001,
            },
        )
        diag_codes = [r.get("code") for r in deadband.get("diagnostic_only_checks", [])]
        self.assertIn("LOW_BREADTH", diag_codes)
        self.assertIn("RELATIVE_WEAKNESS", diag_codes)
        self.assertIn("PRE_BREAKOUT_COMPRESSION", diag_codes)

    def test_deadband_required_modality_failure_is_hard_blocker(self):
        deadband = compute_deadband(
            confidence_overall=0.60,
            agreement_score=0.90,
            horizon_alignment_score=0.90,
            margin=0.20,
            event_risk="normal",
            regime={"label": "range", "is_high_vol": False},
            context={
                "required_modality_ok": False,
                "required_modality_failures": [{"modality": "technicals", "code": "STALE"}],
            },
        )
        self.assertTrue(deadband["active"])
        hard_codes = [r.get("code") for r in deadband.get("hard_blockers", [])]
        self.assertIn("CRITICAL_STALE_REQUIRED_MODALITY", hard_codes)

    def test_reason_ordering_prioritizes_hard_blockers_then_weighted(self):
        deadband = compute_deadband(
            confidence_overall=0.20,
            agreement_score=0.10,
            horizon_alignment_score=0.00,
            margin=0.00,
            event_risk="elevated",
            regime={"label": "high_vol_chop", "is_high_vol": True},
        )
        hard = deadband.get("hard_blockers", [])
        weighted = deadband.get("weighted_reasons", [])
        reasons = deadband.get("reasons", [])
        self.assertGreaterEqual(len(hard), 1)
        self.assertGreaterEqual(len(weighted), 1)
        hard_codes = [r.get("code") for r in hard]
        reason_head_codes = [r.get("code") for r in reasons[: len(hard)]]
        self.assertEqual(reason_head_codes, hard_codes)
        weighted_codes = [r.get("code") for r in weighted]
        reason_tail_codes = [r.get("code") for r in reasons[len(hard) : len(hard) + len(weighted)]]
        self.assertEqual(reason_tail_codes, weighted_codes)

    def test_alignment_penalty_is_capped_and_explicit(self):
        deadband = compute_deadband(
            confidence_overall=0.50,
            agreement_score=0.90,
            horizon_alignment_score=0.00,
            margin=0.20,
            event_risk="normal",
            regime={"label": "high_vol_chop", "is_high_vol": False},
        )
        # deficit = 0.60, slope=0.25 -> 0.15, but cap is 0.10
        self.assertEqual(deadband.get("effective_confidence"), 0.4)
        qa = deadband.get("quality_adjustments", [])
        self.assertTrue(qa and qa[0].get("code") == "LOW_HORIZON_ALIGNMENT_QUALITY")

    def test_build_output_schema_and_domains(self):
        out = build_decision_output("BTC-USD", _snapshot(), use_gpt=False, gpt_model="gpt-5.2")

        self.assertIn(out["decision"]["action"], ALLOWED_ACTIONS)
        self.assertIn(out["strategy_inputs"]["regime_classification"]["market_regime"], ALLOWED_REGIMES)
        self.assertGreaterEqual(float(out["strategy_inputs"]["regime_classification"]["confidence"]), 0.0)
        self.assertLessEqual(float(out["strategy_inputs"]["regime_classification"]["confidence"]), 1.0)
        self.assertGreaterEqual(int(out["decision"]["validity_hours"]), 24)
        self.assertLessEqual(int(out["decision"]["validity_hours"]), 168)

        self.assertIn("symbol", out)
        self.assertIn("price", out)
        self.assertIn("position_state", out)
        self.assertIn("next_position_state", out)
        self.assertIn("asof", out)
        self.assertIn("decision", out)
        self.assertIn("decision_logic_version", out)
        self.assertIn("strategy_inputs", out)
        self.assertIn("no_trade_gate", out)
        self.assertIn("paper_trade_preview", out)
        self.assertIn("trade_plan", out)
        self.assertIn("trade_explanation", out)
        self.assertIn("explain_like_human", out)
        self.assertIn("evidence_audit", out)
        gate = out.get("no_trade_gate") or {}
        self.assertIn("gate_model", gate)
        self.assertIn("hard_blockers", gate)
        self.assertIn("weighted_reasons", gate)
        self.assertIn("penalty_total", gate)
        self.assertIn("penalty_threshold", gate)
        self.assertIn("penalty_ratio", gate)
        self.assertIn("passed_checks", gate)
        self.assertIn("activation_basis", gate)
        self.assertIn("reasons", gate)
        self.assertIn("quality_adjustments", gate)
        self.assertEqual(gate.get("gate_model"), "hybrid_weighted_v1")
        thresholds = (out.get("evidence_audit") or {}).get("thresholds") or {}
        self.assertIn("min_confidence", thresholds)
        self.assertIn("min_agreement", thresholds)
        self.assertIn("min_horizon_alignment", thresholds)

        explain = out["explain_like_human"]
        self.assertIsInstance(explain.get("summary"), str)
        self.assertIsInstance(explain.get("what_would_change"), list)
        self.assertIsInstance(explain.get("top_risks"), list)
        self.assertIn(explain.get("source"), {"deterministic_fallback", "fallback_after_gpt_failure", "gpt"})

        # Hypothetical action can now be neutral/no-trade in weak mixed states.
        self.assertIn(out["paper_trade_preview"]["hypothetical_action"], {"LONG", "SHORT", "NO_TRADE"})
        self.assertIsInstance(out["paper_trade_preview"]["would_trade_if_allowed"], bool)
        self.assertIn("blocked_by", out["paper_trade_preview"])
        self.assertIn("quality_flags", out["paper_trade_preview"])
        self.assertIn("effective_confidence", out["strategy_inputs"]["signal_quality"])
        self.assertIn("technical_context", out["strategy_inputs"])
        self.assertIn("fundamental_context", out["strategy_inputs"])
        self.assertIn("shadow_intelligence", out["strategy_inputs"])
        shadow = out["strategy_inputs"]["shadow_intelligence"] or {}
        self.assertEqual(shadow.get("stage"), "shadow")
        self.assertIn(shadow.get("status"), {"ok", "unavailable", "error"})
        self.assertIn("prediction", shadow)
        self.assertIn("patterns", shadow)
        self.assertIn("breadth_score", out["strategy_inputs"]["technical_context"])
        self.assertIn("fundamental_trend_score", out["strategy_inputs"]["fundamental_context"])
        self.assertEqual(
            out["strategy_inputs"]["signal_quality"].get("hold_window_source"),
            "deterministic_swing_v1",
        )

        trade_explanation = out.get("trade_explanation") or {}
        self.assertIsInstance(trade_explanation.get("rationale"), str)
        self.assertIsInstance(trade_explanation.get("risk_context"), str)
        self.assertIsInstance(trade_explanation.get("positioning_logic"), str)
        self.assertIn(
            trade_explanation.get("source"),
            {"deterministic_fallback", "fallback_after_gpt_failure", "gpt"},
        )

        if out["no_trade_gate"]["active"]:
            self.assertEqual(out["decision"]["action"], "NO_TRADE")
            self.assertGreaterEqual(len(out["no_trade_gate"].get("reasons", [])), 1)
            self.assertIn(out["decision"]["action"], {"NO_TRADE", "HOLD"})
            if out["decision"]["action"] == "NO_TRADE":
                self.assertIsNone(out.get("trade_plan"))
            else:
                self.assertIsInstance(out.get("trade_plan"), dict)
        else:
            tp = out.get("trade_plan") or {}
            self.assertIn("risk_management", tp)
            self.assertIn("target_plan", tp)
            self.assertIn("exit_conditions", tp)
            if out["decision"]["action"] in {"LONG", "SHORT"}:
                self.assertIn("position_sizing", tp)
                self.assertIn("entry_plan", tp)
                rm = tp.get("risk_management", {}) or {}
                self.assertTrue((rm.get("validations", {}) or {}).get("stop_distance_positive"))
                self.assertTrue((rm.get("validations", {}) or {}).get("risk_dollars_within_cap"))
                exit_conditions = tp.get("exit_conditions", {}) or {}
                self.assertIn("breakeven_trigger_r", exit_conditions)
                self.assertIn("partial_take_profit", exit_conditions)
                self.assertIn("trailing_stop_atr", exit_conditions)

        # JSON serializable contract
        payload = json.dumps(out, sort_keys=True)
        self.assertIsInstance(payload, str)

    def test_open_position_uses_hold_not_no_trade(self):
        out = build_decision_output(
            "BTC-USD",
            _snapshot(),
            use_gpt=False,
            gpt_model="gpt-5.2",
            current_position_state="LONG",
            con=None,
        )
        self.assertEqual(out.get("position_state"), "LONG")
        self.assertEqual(out["decision"]["action"], "HOLD")
        tp = out.get("trade_plan") or {}
        self.assertEqual(tp.get("mode"), "position_management")
        self.assertIn(tp.get("source"), {"entry_anchored", "derived_fallback"})
        preview = out.get("paper_trade_preview", {}) or {}
        self.assertIn("flip_alert", preview)
        self.assertIn("state_transition", preview)
        self.assertIn("quality_flags", preview)

    def test_hold_plan_populated_with_anchor_tighten_only(self):
        con = duckdb.connect(":memory:")
        now_utc = datetime.now(timezone.utc)
        anchor_dt = now_utc - timedelta(hours=24)
        anchor_iso = anchor_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        anchor_sql = anchor_dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
        con.execute(
            """
            CREATE TABLE trident_decisions (
                asof_utc TIMESTAMP,
                created_at_utc TIMESTAMP,
                symbol VARCHAR,
                payload_json VARCHAR
            )
            """
        )
        anchor_payload = {
            "asof": anchor_iso,
            "decision": {"action": "LONG"},
            "trade_plan": {
                "risk_management": {
                    "entry_price_estimate": 65000.0,
                    "stop_price_estimate": 62000.0,
                },
                "target_plan": {"target_price_estimate": 70000.0},
            },
        }
        con.execute(
            """
            INSERT INTO trident_decisions (asof_utc, created_at_utc, symbol, payload_json)
            VALUES (CAST(? AS TIMESTAMP), NOW(), ?, ?)
            """,
            [anchor_sql, "BTC-USD", json.dumps(anchor_payload)],
        )

        out = build_decision_output(
            "BTC-USD",
            _snapshot(),
            use_gpt=False,
            gpt_model="gpt-5.2",
            current_position_state="LONG",
            con=con,
        )
        self.assertEqual(out["decision"]["action"], "HOLD")
        tp = out.get("trade_plan") or {}
        self.assertEqual(tp.get("mode"), "position_management")
        self.assertEqual(tp.get("source"), "entry_anchored")
        self.assertEqual(tp.get("position_side"), "LONG")
        anchor = tp.get("anchor") or {}
        self.assertEqual(anchor.get("entry_price"), 65000.0)
        self.assertEqual(anchor.get("stop_price"), 62000.0)
        rm = tp.get("risk_management") or {}
        self.assertIn(rm.get("stop_source"), {"entry_anchored", "recomputed"})
        self.assertIn(rm.get("atr_mode"), {"applied", "not_applied"})
        self.assertIn("computed_stop_candidate", rm)
        self.assertIn("anchor_age_hours", rm)
        self.assertIn("anchor_stale", rm)
        self.assertFalse(bool(rm.get("anchor_stale")))
        self.assertLessEqual(float(rm.get("anchor_age_hours") or 0.0), 72.0)
        self.assertGreaterEqual(float(rm.get("stop_price_estimate") or 0.0), 62000.0)
        self.assertTrue(((rm.get("validations") or {}).get("stop_not_widened")))
        if rm.get("atr_mode") == "not_applied":
            self.assertIsNone(rm.get("atr_used"))
            self.assertIsNone(rm.get("stop_multiple_atr"))
        else:
            self.assertIsInstance(rm.get("atr_used"), float)
            self.assertIsInstance(rm.get("stop_multiple_atr"), float)
        ex = (tp.get("exit_conditions") or {}).get("escalation_state") or {}
        self.assertIn("recommended_step", ex)
        self.assertIn(ex.get("recommended_step"), {"MAINTAIN", "REDUCE_OR_EXIT"})

    def test_hold_plan_derived_fallback_when_no_anchor(self):
        out = build_decision_output(
            "BTC-USD",
            _snapshot(),
            use_gpt=False,
            gpt_model="gpt-5.2",
            current_position_state="SHORT",
            con=None,
        )
        self.assertEqual(out["decision"]["action"], "HOLD")
        tp = out.get("trade_plan") or {}
        self.assertEqual(tp.get("mode"), "position_management")
        self.assertEqual(tp.get("source"), "derived_fallback")
        rm = tp.get("risk_management") or {}
        self.assertEqual(rm.get("stop_source"), "recomputed")
        self.assertEqual(rm.get("atr_mode"), "applied")
        self.assertIsInstance(rm.get("atr_used"), float)
        self.assertIsInstance(rm.get("stop_multiple_atr"), float)
        self.assertTrue((rm.get("validations") or {}).get("stop_distance_positive"))
        self.assertTrue((rm.get("validations") or {}).get("position_side_matches_state"))
        ex = (tp.get("exit_conditions") or {}).get("escalation_state") or {}
        self.assertIn("recommended_step", ex)

    def test_stale_anchor_forces_derived_fallback(self):
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE trident_decisions (
                asof_utc TIMESTAMP,
                created_at_utc TIMESTAMP,
                symbol VARCHAR,
                payload_json VARCHAR
            )
            """
        )
        anchor_payload = {
            "asof": "2026-02-10T00:00:00Z",
            "decision": {"action": "SHORT"},
            "position_state": "SHORT",
            "strategy_inputs": {"signal_quality": {"agreement_score": 0.60, "effective_confidence": 0.60}},
            "trade_plan": {
                "risk_management": {
                    "entry_price_estimate": 65000.0,
                    "stop_price_estimate": 67000.0,
                },
                "target_plan": {"target_price_estimate": 62000.0},
            },
        }
        con.execute(
            """
            INSERT INTO trident_decisions (asof_utc, created_at_utc, symbol, payload_json)
            VALUES (CAST(? AS TIMESTAMP), NOW(), ?, ?)
            """,
            ["2026-02-10 00:00:00", "BTC-USD", json.dumps(anchor_payload)],
        )
        out = build_decision_output(
            "BTC-USD",
            _snapshot(),
            use_gpt=False,
            gpt_model="gpt-5.2",
            current_position_state="SHORT",
            con=con,
        )
        tp = out.get("trade_plan") or {}
        self.assertEqual(tp.get("source"), "derived_fallback")
        rm = tp.get("risk_management") or {}
        self.assertTrue(bool(rm.get("anchor_stale")))
        self.assertGreater(float(rm.get("anchor_age_hours") or 0.0), 72.0)

    def test_blocked_by_vs_quality_flags_semantics(self):
        # Gate inactive with quality issue -> blocked_by empty, quality_flags populated.
        snap = _snapshot()
        snap["technical_features"]["multi_horizon"]["horizon_signals"]["6h"]["trend_bias"] = "bearish"
        out = build_decision_output("BTC-USD", snap, use_gpt=False, gpt_model="gpt-5.2")
        gate = out.get("no_trade_gate") or {}
        preview = out.get("paper_trade_preview") or {}
        if not bool(gate.get("active")):
            self.assertEqual(preview.get("blocked_by"), [])
            self.assertIsInstance(preview.get("quality_flags"), list)

    def test_blocked_by_populated_when_gate_active(self):
        forced_deadband = {
            "active": True,
            "gate_model": "hybrid_weighted_v1",
            "hard_blockers": [],
            "weighted_reasons": [],
            "penalty_total": 0.5,
            "penalty_threshold": 0.45,
            "penalty_ratio": 1.111,
            "passed_checks": [],
            "activation_basis": "penalty_threshold",
            "reasons": [{"code": "LOW_CONFIDENCE", "detail": "forced"}],
            "quality_adjustments": [],
            "effective_confidence": 0.2,
            "thresholds": compute_gate_thresholds("range"),
        }
        with patch("backend.decide.trident_decision.compute_deadband", return_value=forced_deadband):
            out = build_decision_output("BTC-USD", _snapshot(), use_gpt=False, gpt_model="gpt-5.2")
        preview = out.get("paper_trade_preview") or {}
        self.assertIn("LOW_CONFIDENCE", preview.get("blocked_by") or [])

    def test_hold_escalation_triggers(self):
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE trident_decisions (
                asof_utc TIMESTAMP,
                created_at_utc TIMESTAMP,
                symbol VARCHAR,
                payload_json VARCHAR
            )
            """
        )
        p1 = {
            "asof": "2026-02-16T10:00:00Z",
            "position_state": "SHORT",
            "decision": {"action": "HOLD"},
            "strategy_inputs": {"signal_quality": {"agreement_score": 0.45, "effective_confidence": 0.34}},
            "trade_plan": {
                "risk_management": {"entry_price_estimate": 65000.0, "stop_price_estimate": 67000.0},
                "target_plan": {"target_price_estimate": 62000.0},
            },
        }
        con.execute(
            """
            INSERT INTO trident_decisions (asof_utc, created_at_utc, symbol, payload_json)
            VALUES (CAST(? AS TIMESTAMP), NOW(), ?, ?)
            """,
            ["2026-02-16 10:00:00", "BTC-USD", json.dumps(p1)],
        )
        ex = _build_hold_escalation_state(
            con=con,
            symbol="BTC-USD",
            side="SHORT",
            gate_active=False,
            agreement_score=0.45,
            effective_confidence=0.34,
        )
        self.assertGreaterEqual(int(ex.get("agreement_breach_runs") or 0), 2)
        self.assertGreaterEqual(int(ex.get("confidence_breach_runs") or 0), 2)
        self.assertTrue(bool(ex.get("triggered")))
        self.assertEqual(ex.get("recommended_step"), "REDUCE_OR_EXIT")

    def test_hold_escalation_gate_active_immediate(self):
        ex = _build_hold_escalation_state(
            con=None,
            symbol="BTC-USD",
            side="SHORT",
            gate_active=True,
            agreement_score=0.70,
            effective_confidence=0.70,
        )
        self.assertTrue(bool(ex.get("gate_active_trigger")))
        self.assertTrue(bool(ex.get("triggered")))
        self.assertEqual(ex.get("recommended_step"), "REDUCE_OR_EXIT")


if __name__ == "__main__":
    unittest.main()
