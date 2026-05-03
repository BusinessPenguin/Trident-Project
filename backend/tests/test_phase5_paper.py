import json
import unittest
from datetime import datetime, timedelta, timezone
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb

from backend.db.paper_repo import (
    compute_equity_snapshot,
    create_run,
    get_active_paper_config,
    list_open_positions,
    record_candidate,
    record_decision,
    reset_paper_ledger,
    upsert_paper_config,
)
from backend.db.schema import apply_core_schema
from backend.services.paper_engine import (
    apply_intelligence,
    build_pattern_snapshot,
    build_prediction_snapshot,
    compute_position_plan,
    evaluate_exit,
    evaluate_entry_eligibility,
    extract_candidate_from_decision,
    fallback_aggression_profile,
    mark_open_positions,
    replay_open_positions,
    resolve_intel_stage,
    run_paper_cycle,
    simulate_fill,
)
from backend.services.paper_learning import (
    classify_failures,
    load_closed_trades,
    propose_parameter_updates,
)
from backend.services.policy_ai import render_paper_report_summary_gpt52
from backend.cli import paper_mark_cmd, paper_report_cmd, paper_run_cmd


def _base_config() -> dict:
    return {
        "starting_equity": 10000.0,
        "symbols": ["BTC-USD", "ETH-USD"],
        "fee_bps": 5.0,
        "slippage_bps": 8.0,
        "max_trades_per_run": 1,
        "max_open_positions": 5,
        "risk_limits": {
            "max_risk_per_trade_pct": 0.01,
            "max_total_exposure_pct": 0.60,
            "max_open_positions_per_symbol": 1,
            "replay_interval": "15m",
            "replay_lookback_bars": 672,
            "min_confidence": 0.33,
            "stop_distance_atr_mult": 1.0,
            "stop_distance_pct_fallback": 0.015,
            "entry_min_score": 0.18,
            "entry_min_effective_confidence": 0.40,
            "entry_min_agreement": 0.60,
            "entry_min_margin": 0.06,
            "quality_veto_enabled": True,
            "intelligence_mode": "auto",
            "intelligence_bootstrap_trades": 25,
            "intelligence_promotion_trades": 50,
            "prediction_enabled": True,
            "pattern_enabled": True,
            "prediction_weight_soft": 0.15,
            "pattern_weight_soft": 0.10,
            "prediction_weight_hard": 0.20,
            "pattern_weight_hard": 0.10,
            "allow_weighted_gate_override": False,
        },
        "learning_policy": {
            "penalty_high_vol_chop": 0.9,
            "penalty_elevated_event_risk": 0.9,
            "gpt_learn_max_influence": 0.30,
            "gpt_learn_min_trades": 25,
            "learn_bootstrap_stop_only": True,
            "learn_apply_cooldown_hours": 24,
            "gate_overrides": {"min_confidence": 0.33},
            "aggression_baseline": {
                "risk_mult": 1.0,
                "stop_mult": 1.0,
                "hold_mult": 1.0,
                "exposure_cap": 0.12,
            },
        },
    }


def _decision(action: str = "HOLD", gate_active: bool = False, hyp: str = "LONG") -> dict:
    return {
        "symbol": "BTC-USD",
        "asof": "2026-02-18T00:00:00Z",
        "decision": {"action": action, "confidence": 0.5, "validity_hours": 8, "conviction_label": "medium"},
        "strategy_inputs": {
            "regime_classification": {"market_regime": "range", "confidence": 0.5},
            "scenario_snapshot": {"top_scenario": "base_case", "margin_vs_second": 0.1},
            "signal_quality": {
                "agreement_score": 0.7,
                "freshness_score": 0.8,
                "effective_confidence": 0.6,
                "event_risk": "normal",
            },
        },
        "no_trade_gate": {
            "active": gate_active,
            "reasons": [{"code": "LOW_CONFIDENCE", "detail": "x"}] if gate_active else [],
            "activation_basis": "penalty_threshold" if gate_active else "none",
            "hard_blockers": [],
            "penalty_ratio": 0.1,
        },
        "paper_trade_preview": {"hypothetical_action": hyp, "blocked_by": [], "quality_flags": []},
    }


def _last_json(stdout_text: str) -> dict:
    lines = stdout_text.splitlines()
    starts = [idx for idx, line in enumerate(lines) if line.strip() == "{"]
    for idx in starts:
        blob = "\n".join(lines[idx:]).strip()
        try:
            parsed = json.loads(blob)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    for idx, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        blob = "\n".join(lines[idx:]).strip()
        try:
            parsed = json.loads(blob)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


class TestPhase5Paper(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect(":memory:")
        apply_core_schema(self.con)

    def tearDown(self) -> None:
        self.con.close()

    def test_schema_has_paper_tables(self):
        rows = self.con.execute("SHOW TABLES").fetchall()
        names = {r[0] for r in rows}
        for t in [
            "paper_config",
            "paper_runs",
            "paper_decisions",
            "paper_candidates",
            "paper_positions",
            "paper_fills",
            "paper_marks",
            "paper_learning_events",
            "paper_replay_events",
            "paper_signal_audit",
        ]:
            self.assertIn(t, names)

    def test_candidate_extracts_hold_from_hypothetical(self):
        c = extract_candidate_from_decision(_decision(action="HOLD", gate_active=False, hyp="SHORT"))
        self.assertEqual(c["side"], "SHORT")
        self.assertGreater(c["candidate_score"], 0.0)

    def test_candidate_gate_active_forces_no_trade(self):
        c = extract_candidate_from_decision(_decision(action="LONG", gate_active=True, hyp="LONG"))
        self.assertEqual(c["side"], "NO_TRADE")
        self.assertEqual(c["candidate_score"], 0.0)
        self.assertTrue(c["gate_active"])
        self.assertEqual(c["gates_blocking"], ["LOW_CONFIDENCE"])
        self.assertEqual(c["quality_flags"], [])

    def test_candidate_gate_inactive_uses_quality_flags(self):
        d = _decision(action="LONG", gate_active=False, hyp="LONG")
        d["no_trade_gate"]["reasons"] = [{"code": "LOW_HORIZON_ALIGNMENT", "detail": "x"}]
        c = extract_candidate_from_decision(d)
        self.assertFalse(c["gate_active"])
        self.assertEqual(c["gates_blocking"], [])
        self.assertEqual(c["quality_flags"], ["LOW_HORIZON_ALIGNMENT"])

    def test_candidate_score_adjusts_for_quality_flags_and_context(self):
        base = _decision(action="LONG", gate_active=False, hyp="LONG")
        base["strategy_inputs"]["technical_context"] = {
            "breadth_score": 0.80,
            "rs_vs_btc_7d": 0.03,
            "vwap_distance_pct": 0.02,
            "squeeze_on": False,
        }
        base_candidate = extract_candidate_from_decision(base)

        weak = _decision(action="LONG", gate_active=False, hyp="LONG")
        weak["no_trade_gate"]["reasons"] = [
            {"code": "LOW_BREADTH", "detail": "x"},
            {"code": "RELATIVE_WEAKNESS", "detail": "x"},
            {"code": "PRE_BREAKOUT_COMPRESSION", "detail": "x"},
        ]
        weak["strategy_inputs"]["technical_context"] = {
            "breadth_score": 0.30,
            "rs_vs_btc_7d": -0.04,
            "vwap_distance_pct": 0.001,
            "squeeze_on": True,
        }
        weak_candidate = extract_candidate_from_decision(weak)
        self.assertGreater(base_candidate["candidate_score"], weak_candidate["candidate_score"])

    def test_entry_eligibility_applies_floor_and_quality_veto(self):
        d = _decision(action="LONG", gate_active=False, hyp="LONG")
        d["strategy_inputs"]["scenario_snapshot"]["margin_vs_second"] = 0.08
        c = extract_candidate_from_decision(d)
        c["quality_flags"] = ["MODEL_EDGE_WEAK", "LOW_CONFIDENCE"]
        ok, blockers = evaluate_entry_eligibility(c, d, _base_config()["risk_limits"])
        self.assertFalse(ok)
        self.assertIn("QUALITY_VETO_LOW_CONFIDENCE_MODEL_EDGE_WEAK", blockers)

        c2 = dict(c)
        c2["quality_flags"] = ["MODEL_EDGE_WEAK", "LOW_BREADTH"]
        c2["effective_confidence"] = 0.60
        c2["agreement_score"] = 0.85
        c2["candidate_score"] = 0.30
        ok2, blockers2 = evaluate_entry_eligibility(c2, d, _base_config()["risk_limits"])
        self.assertTrue(ok2)
        self.assertNotIn("QUALITY_VETO_LOW_CONFIDENCE_MODEL_EDGE_WEAK", blockers2)

    def test_resolve_intel_stage_auto_bootstrap(self):
        s1 = resolve_intel_stage(closed_trades=9, mode="auto", bootstrap=25, promotion=50)
        self.assertEqual(s1["stage"], "shadow")
        self.assertTrue(s1["bootstrap_guard_active"])
        s2 = resolve_intel_stage(closed_trades=30, mode="auto", bootstrap=25, promotion=50)
        self.assertEqual(s2["stage"], "soft")
        s3 = resolve_intel_stage(closed_trades=55, mode="auto", bootstrap=25, promotion=50)
        self.assertEqual(s3["stage"], "hard")

    def test_prediction_and_pattern_snapshots_are_deterministic(self):
        now = datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc)
        for i in range(140):
            ts = now - timedelta(minutes=(139 - i) * 15)
            close = 100.0 + (i * 0.05)
            self.con.execute(
                """
                INSERT INTO candles (symbol, interval, ts, open, high, low, close, volume)
                VALUES ('BTC-USD','15m', ?, ?, ?, ?, ?, 1)
                """,
                [ts, close - 0.1, close + 0.2, close - 0.3, close],
            )
        d = _decision(action="SHORT", gate_active=False, hyp="SHORT")
        c = extract_candidate_from_decision(d)
        p1 = build_prediction_snapshot(self.con, "BTC-USD", d, c, interval="15m")
        p2 = build_prediction_snapshot(self.con, "BTC-USD", d, c, interval="15m")
        self.assertEqual(p1, p2)
        pat1 = build_pattern_snapshot(self.con, "BTC-USD", d, interval="15m")
        pat2 = build_pattern_snapshot(self.con, "BTC-USD", d, interval="15m")
        self.assertEqual(pat1, pat2)
        self.assertIn("version", p1)
        self.assertIn("version", pat1)

    def test_apply_intelligence_shadow_is_non_blocking(self):
        candidate = {
            "symbol": "BTC-USD",
            "side": "SHORT",
            "candidate_score": 0.25,
        }
        intel = apply_intelligence(
            candidate=candidate,
            prediction={"direction_24h": "LONG", "confidence": 0.9, "ev_r": -0.2},
            patterns={"support_long": 0.9, "support_short": 0.1, "conflict_ratio": 0.2},
            stage_cfg={"stage": "shadow", "prediction_weight": 0.20, "pattern_weight": 0.10},
        )
        self.assertFalse(intel["intel_used_for_entry"])
        self.assertEqual(float(intel["intel_score_delta"]), 0.0)
        self.assertEqual(list(intel["intel_blockers"]), [])

    def test_replay_stop_recovery_closes_position(self):
        cfg = upsert_paper_config(self.con, _base_config(), set_active=True)
        entry_ts = datetime(2026, 2, 20, 0, 0, tzinfo=timezone.utc)
        self.con.execute(
            """
            INSERT INTO paper_positions (
                position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            ) VALUES ('pos_replay_1','BTC-USD','LONG',1.0,?,?,?,?,?,'OPEN',NULL,NULL,NULL,'run1')
            """,
            [entry_ts, 100.0, 95.0, 110.0, entry_ts + timedelta(hours=24)],
        )
        self.con.execute(
            """
            INSERT INTO candles (symbol, interval, ts, open, high, low, close, volume)
            VALUES ('BTC-USD','15m', ?, 99, 100, 94, 95, 1)
            """,
            [entry_ts + timedelta(minutes=15)],
        )
        out = replay_open_positions(
            conn=self.con,
            positions=list_open_positions(self.con),
            config=cfg,
            command="paper:run",
            run_id="run_r1",
            interval="15m",
            now_utc=entry_ts + timedelta(minutes=20),
        )
        self.assertEqual(int(out.get("exits_replayed") or 0), 1)
        pos = self.con.execute(
            "SELECT status, exit_reason FROM paper_positions WHERE position_id='pos_replay_1'"
        ).fetchone()
        self.assertEqual(pos[0], "CLOSED")
        self.assertEqual(pos[1], "STOP_HIT_REPLAY")
        n_events = self.con.execute("SELECT COUNT(*) FROM paper_replay_events").fetchone()[0]
        self.assertEqual(n_events, 1)

    def test_replay_ambiguous_bar_uses_stop_first(self):
        cfg = upsert_paper_config(self.con, _base_config(), set_active=True)
        entry_ts = datetime(2026, 2, 20, 1, 0, tzinfo=timezone.utc)
        self.con.execute(
            """
            INSERT INTO paper_positions (
                position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            ) VALUES ('pos_replay_2','ETH-USD','LONG',1.0,?,?,?,?,?,'OPEN',NULL,NULL,NULL,'run2')
            """,
            [entry_ts, 100.0, 95.0, 105.0, entry_ts + timedelta(hours=24)],
        )
        self.con.execute(
            """
            INSERT INTO candles (symbol, interval, ts, open, high, low, close, volume)
            VALUES ('ETH-USD','15m', ?, 100, 106, 94, 100, 1)
            """,
            [entry_ts + timedelta(minutes=15)],
        )
        out = replay_open_positions(
            conn=self.con,
            positions=list_open_positions(self.con),
            config=cfg,
            command="paper:mark",
            interval="15m",
            now_utc=entry_ts + timedelta(minutes=20),
        )
        self.assertEqual(int(out.get("exits_replayed") or 0), 1)
        row = self.con.execute(
            "SELECT exit_reason FROM paper_positions WHERE position_id='pos_replay_2'"
        ).fetchone()
        self.assertEqual(row[0], "STOP_HIT_REPLAY")

    def test_replay_respects_window_from_last_activity(self):
        cfg = upsert_paper_config(self.con, _base_config(), set_active=True)
        now = datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc)
        entry_ts = now - timedelta(hours=6)
        self.con.execute(
            """
            INSERT INTO paper_positions (
                position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            ) VALUES ('pos_replay_window','BTC-USD','LONG',1.0,?,?,?,?,?,'OPEN',NULL,NULL,NULL,'run3')
            """,
            [entry_ts, 100.0, 95.0, 110.0, now + timedelta(hours=12)],
        )
        # Last activity (mark) is newer than the stop-touch bar below.
        self.con.execute(
            """
            INSERT INTO paper_marks (mark_id, ts, symbol, mid_price, position_id, unrealized_pnl_usd, equity, drawdown_pct)
            VALUES ('mark_last', ?, 'BTC-USD', 101.0, 'pos_replay_window', 1.0, 10001.0, 0.0)
            """,
            [now - timedelta(hours=1)],
        )
        self.con.execute(
            """
            INSERT INTO candles (symbol, interval, ts, open, high, low, close, volume)
            VALUES ('BTC-USD','15m', ?, 100, 101, 94, 95, 1)
            """,
            [now - timedelta(hours=2)],
        )
        out = replay_open_positions(
            conn=self.con,
            positions=list_open_positions(self.con),
            config=cfg,
            command="paper:run",
            run_id="run_window",
            interval="15m",
            now_utc=now,
        )
        self.assertEqual(int(out.get("exits_replayed") or 0), 0)
        pos = self.con.execute(
            "SELECT status FROM paper_positions WHERE position_id='pos_replay_window'"
        ).fetchone()
        self.assertEqual(pos[0], "OPEN")

    def test_run_cycle_selects_best_candidate(self):
        d1 = _decision(action="HOLD", gate_active=False, hyp="LONG")
        d2 = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        d2["symbol"] = "ETH-USD"
        d2["strategy_inputs"]["signal_quality"]["effective_confidence"] = 0.3
        out = run_paper_cycle([d1, d2], max_trades=1)
        self.assertIsNotNone(out["selected"])
        self.assertEqual(out["selected"]["symbol"], "BTC-USD")

    def test_simulate_fill_is_deterministic(self):
        f1 = simulate_fill(100.0, "LONG", 1.0, fee_bps=5.0, slippage_bps=8.0, fill_type="ENTRY")
        f2 = simulate_fill(100.0, "LONG", 1.0, fee_bps=5.0, slippage_bps=8.0, fill_type="ENTRY")
        self.assertEqual(f1, f2)
        self.assertGreater(f1["fill_price"], 100.0)

    def test_compute_position_plan_respects_exposure_cap(self):
        plan = compute_position_plan(
            entry_price=100.0,
            side="LONG",
            equity=10000.0,
            risk_cfg={
                "max_risk_per_trade_pct": 0.01,
                "max_total_exposure_pct": 0.10,
                "stop_distance_atr_mult": 1.0,
                "stop_distance_pct_fallback": 0.015,
            },
            atr_pct=0.02,
            validity_hours=8,
        )
        self.assertGreater(plan["qty"], 0.0)
        self.assertLessEqual(plan["qty"] * 100.0, 10000.0 * 0.10 + 1e-9)

    def test_compute_position_plan_applies_aggression_knobs(self):
        plan = compute_position_plan(
            entry_price=100.0,
            side="LONG",
            equity=10000.0,
            risk_cfg={
                "max_risk_per_trade_pct": 0.02,
                "max_total_exposure_pct": 0.60,
                "stop_distance_atr_mult": 1.0,
                "stop_distance_pct_fallback": 0.015,
            },
            atr_pct=0.02,
            validity_hours=48,
            aggression_knobs={
                "risk_mult": 2.0,
                "stop_mult": 0.8,
                "hold_mult": 0.75,
                "notional_cap": 0.10,
            },
        )
        self.assertLessEqual(plan["risk_usd"], 10000.0 * 0.02 + 1e-9)
        self.assertLessEqual(plan["qty"] * 100.0, 10000.0 * 0.10 + 1e-9)
        self.assertGreaterEqual(int(plan["validity_hours_applied"]), 24)
        self.assertLessEqual(int(plan["validity_hours_applied"]), 168)

    def test_candidate_override_weighted_gate_allowed(self):
        d = _decision(action="NO_TRADE", gate_active=True, hyp="LONG")
        d["no_trade_gate"]["hard_blockers"] = []
        d["no_trade_gate"]["activation_basis"] = "penalty_threshold"
        d["strategy_inputs"]["signal_quality"]["effective_confidence"] = 0.62
        d["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.70
        d["strategy_inputs"]["scenario_snapshot"]["margin_vs_second"] = 0.09
        c = extract_candidate_from_decision(
            d,
            aggression_profile={
                "tier": "assertive",
                "source": "gpt",
                "override_weighted_gate": True,
                "quick_exit_bias": "none",
                "confidence": 0.8,
                "reasoning_summary": "strong structure",
                "knobs_applied": fallback_aggression_profile()["knobs_applied"],
            },
        )
        self.assertTrue(c["override_weighted_gate"])
        self.assertEqual(c["side"], "LONG")
        self.assertGreater(c["candidate_score"], 0.0)
        self.assertEqual(c["gates_blocking"], [])

    def test_candidate_override_rejected_when_hard_blocker_present(self):
        d = _decision(action="NO_TRADE", gate_active=True, hyp="SHORT")
        d["no_trade_gate"]["hard_blockers"] = [{"code": "CRITICAL_LOW_CONFIDENCE", "detail": "x"}]
        d["strategy_inputs"]["signal_quality"]["effective_confidence"] = 0.70
        d["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.80
        d["strategy_inputs"]["scenario_snapshot"]["margin_vs_second"] = 0.12
        c = extract_candidate_from_decision(
            d,
            aggression_profile={
                "tier": "very_aggressive",
                "source": "gpt",
                "override_weighted_gate": True,
                "quick_exit_bias": "none",
                "confidence": 0.9,
                "reasoning_summary": "force",
                "knobs_applied": fallback_aggression_profile()["knobs_applied"],
            },
        )
        self.assertFalse(c["override_weighted_gate"])
        self.assertEqual(c["side"], "NO_TRADE")
        self.assertEqual(c["candidate_score"], 0.0)

    def test_evaluate_exit_rules(self):
        now = datetime.now(timezone.utc)
        pos = {
            "side": "LONG",
            "stop_price": 95.0,
            "take_profit_price": 110.0,
            "time_stop_ts": now + timedelta(hours=1),
        }
        self.assertEqual(evaluate_exit(pos, 94.0, now), "STOP_HIT")
        self.assertEqual(evaluate_exit(pos, 111.0, now), "TP_HIT")
        self.assertEqual(evaluate_exit(pos, 100.0, now + timedelta(hours=2)), "TIME_STOP")

    def test_repo_config_run_decision_candidate_roundtrip(self):
        cfg = upsert_paper_config(self.con, _base_config(), set_active=True)
        self.assertTrue(cfg["active"])
        active = get_active_paper_config(self.con)
        self.assertEqual(active["config_id"], cfg["config_id"])
        run_id = create_run(
            self.con,
            "paper:run",
            {"symbols_requested": ["BTC-USD"], "refresh_mode": "smart_refresh", "dry_run": True},
            cfg["config_id"],
        )
        d = _decision(action="LONG", gate_active=False, hyp="LONG")
        record_decision(self.con, run_id, "BTC-USD", d)
        c = extract_candidate_from_decision(d)
        record_candidate(self.con, run_id, c)
        n_dec = self.con.execute("SELECT COUNT(*) FROM paper_decisions").fetchone()[0]
        n_can = self.con.execute("SELECT COUNT(*) FROM paper_candidates").fetchone()[0]
        self.assertEqual(n_dec, 1)
        self.assertEqual(n_can, 1)

    def test_mark_open_positions_closes_stop(self):
        cfg = upsert_paper_config(self.con, _base_config(), set_active=True)
        now = datetime.now(timezone.utc)
        self.con.execute(
            """
            INSERT INTO candles (symbol, interval, ts, open, high, low, close, volume)
            VALUES ('BTC-USD','1h', ?, 100, 101, 90, 90, 1)
            """,
            [now],
        )
        self.con.execute(
            """
            INSERT INTO paper_positions (
                position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', NULL, NULL, NULL, ?)
            """,
            [
                "pos1",
                "BTC-USD",
                "LONG",
                1.0,
                now - timedelta(hours=1),
                100.0,
                95.0,
                120.0,
                now + timedelta(hours=4),
                "run1",
            ],
        )
        out = mark_open_positions(
            conn=self.con,
            positions=list_open_positions(self.con),
            config=cfg,
            now_utc=now,
            exit_on_flip=False,
        )
        self.assertEqual(len(out["exits_triggered"]), 1)
        status = self.con.execute("SELECT status FROM paper_positions WHERE position_id='pos1'").fetchone()[0]
        self.assertEqual(status, "CLOSED")

    def test_learning_updates_are_bounded(self):
        now = datetime.now(timezone.utc)
        # Create one closed losing trade.
        self.con.execute(
            """
            INSERT INTO paper_positions (
                position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED', ?, ?, ?, ?)
            """,
            [
                "pos_l1",
                "BTC-USD",
                "LONG",
                1.0,
                now - timedelta(hours=4),
                100.0,
                95.0,
                110.0,
                now + timedelta(hours=8),
                now - timedelta(hours=1),
                94.0,
                "STOP_HIT",
                "runL",
            ],
        )
        self.con.execute(
            """
            INSERT INTO paper_decisions (run_id, symbol, asof_utc, decision_json, decision_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                "runL",
                "BTC-USD",
                now - timedelta(hours=5),
                json.dumps(_decision(action="LONG", gate_active=False, hyp="LONG")),
                "h1",
            ],
        )
        trades = load_closed_trades(self.con, last_n=10)
        counts, _ = classify_failures(trades)
        changes, risk_new, policy_new, _, _, _ = propose_parameter_updates(
            risk_limits=_base_config()["risk_limits"],
            learning_policy=_base_config()["learning_policy"],
            counts=counts,
        )
        self.assertGreaterEqual(changes["min_confidence"]["ema_new"], 0.25)
        self.assertLessEqual(changes["min_confidence"]["ema_new"], 0.70)
        self.assertGreaterEqual(risk_new["stop_distance_atr_mult"], 0.75)
        self.assertLessEqual(risk_new["stop_distance_atr_mult"], 2.5)
        self.assertIn("gate_overrides", policy_new)
        self.assertIn("min_confidence", policy_new["gate_overrides"])

    def test_learning_gpt_influence_zero_below_10_trades(self):
        changes, _, _, _, gpt_strategy, arbiter = propose_parameter_updates(
            risk_limits=_base_config()["risk_limits"],
            learning_policy=_base_config()["learning_policy"],
            counts={},
            trades=[{"gross_pnl": -1.0}] * 2,
            gpt_policy_proposal={
                "deltas": {"min_confidence": 0.02},
                "confidence": 0.9,
                "rationale": "test",
                "focus_labels": [],
            },
            max_gpt_influence=0.30,
        )
        self.assertEqual(float(gpt_strategy.get("influence_weight") or 0.0), 0.0)
        self.assertIn("gpt_influence_disabled:closed_trades_below_10", list(arbiter.get("rejected_or_clamped_reasons") or []))
        self.assertAlmostEqual(float(changes["min_confidence"]["proposed"]), 0.33, places=6)

    def test_learning_gpt_influence_applies_at_10_trades(self):
        changes, _, _, _, gpt_strategy, arbiter = propose_parameter_updates(
            risk_limits=_base_config()["risk_limits"],
            learning_policy=_base_config()["learning_policy"],
            counts={},
            trades=[{"gross_pnl": -1.0}] * 10,
            gpt_policy_proposal={
                "deltas": {"min_confidence": 0.02},
                "confidence": 0.9,
                "rationale": "test",
                "focus_labels": [],
            },
            max_gpt_influence=0.30,
        )
        self.assertAlmostEqual(float(gpt_strategy.get("influence_weight") or 0.0), 0.2, places=6)
        self.assertGreater(float(changes["min_confidence"]["proposed"]), 0.33)
        self.assertIn("merged_proposal", arbiter)

    def test_learning_stop_only_scope_freezes_non_stop(self):
        changes, risk_new, policy_new, _, _, arbiter = propose_parameter_updates(
            risk_limits=_base_config()["risk_limits"],
            learning_policy=_base_config()["learning_policy"],
            counts={"STOP_TOO_TIGHT": 3},
            trades=[{"gross_pnl": -1.0}] * 9,
            learn_scope="stop_only",
            freeze_aggression_baseline=True,
            min_trades_for_gpt_influence=25,
        )
        self.assertGreater(float(changes["stop_distance_atr_mult"]["ema_new"]), 1.0)
        self.assertAlmostEqual(float(risk_new["entry_min_score"]), 0.18, places=6)
        self.assertAlmostEqual(float(risk_new["entry_min_effective_confidence"]), 0.4, places=6)
        self.assertAlmostEqual(float(risk_new["entry_min_agreement"]), 0.6, places=6)
        self.assertAlmostEqual(float(risk_new["entry_min_margin"]), 0.06, places=6)
        aggr = (policy_new.get("aggression_baseline") or {})
        self.assertAlmostEqual(float(aggr.get("risk_mult") or 0.0), 1.0, places=6)
        self.assertIn("frozen_params", arbiter)
        self.assertIn("aggression_risk_mult", list(arbiter.get("frozen_params") or []))

    def test_reset_paper_ledger_clears_tables(self):
        cfg = upsert_paper_config(self.con, _base_config(), set_active=True)
        run_id = create_run(
            self.con,
            "paper:run",
            {"symbols_requested": ["BTC-USD"], "refresh_mode": "smart_refresh", "dry_run": True},
            cfg["config_id"],
        )
        record_decision(self.con, run_id, "BTC-USD", _decision(action="LONG", gate_active=False, hyp="LONG"))
        reset_paper_ledger(self.con)
        n_cfg = self.con.execute("SELECT COUNT(*) FROM paper_config").fetchone()[0]
        n_runs = self.con.execute("SELECT COUNT(*) FROM paper_runs").fetchone()[0]
        n_dec = self.con.execute("SELECT COUNT(*) FROM paper_decisions").fetchone()[0]
        self.assertEqual(n_cfg, 0)
        self.assertEqual(n_runs, 0)
        self.assertEqual(n_dec, 0)
        eq = compute_equity_snapshot(self.con, starting_equity=10000.0)
        self.assertEqual(eq["equity"], 10000.0)

    def test_paper_mark_refreshes_candles_only_with_replay_interval_and_lookback(self):
        config = _base_config()
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_mark_test.duckdb"),
            paper_symbols=list(config["symbols"]),
        )
        marked = {
            "exits_triggered": [],
            "marks_written": 0,
            "equity": {"equity": 10000.0, "drawdown_pct": 0.0},
        }
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
            patch("backend.services.paper_engine.mark_open_positions", return_value=marked),
            patch("backend.cli.crypto_backfill") as mock_backfill,
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_mark_cmd(symbols=None, exit_on_flip=False)
        for sym in config["symbols"]:
            mock_backfill.assert_any_call(symbol=sym, interval="15m", lookback=672, plain=True)
        out = _last_json(buf.getvalue())
        self.assertEqual(out.get("command"), "paper:mark")
        self.assertIn("candle_refresh", out)
        self.assertIn("replay", out)
        self.assertEqual(int((out.get("candle_refresh") or {}).get("ok_count", 0)), len(config["symbols"]))
        self.assertIn("aggression", out)
        self.assertEqual((out.get("aggression") or {}).get("source"), "deterministic_fallback")

    def test_paper_run_retries_news_pull_and_continues(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_retry"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_retry.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=list(config["symbols"]),
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
        )
        decision = {
            "symbol": "BTC-USD",
            "decision": {"action": "SHORT", "confidence": 0.6, "validity_hours": 8, "conviction_label": "medium"},
            "strategy_inputs": {
                "regime_classification": {"market_regime": "range", "confidence": 0.5},
                "signal_quality": {
                    "agreement_score": 0.7,
                    "freshness_score": 0.8,
                    "effective_confidence": 0.6,
                    "event_risk": "normal",
                },
            },
            "no_trade_gate": {"active": False, "reasons": []},
            "paper_trade_preview": {"hypothetical_action": "SHORT"},
        }
        calls = {"n": 0}

        def _news_pull(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError('TransactionContext Error: Failed to commit: write-write conflict on key: "u"')
            return None

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.fundamentals_pull"),
            patch("backend.cli.news_pull", side_effect=_news_pull),
            patch("backend.cli.fed_pull"),
            patch("backend.cli.sentiment_pull"),
            patch("backend.cli.calendar_pull"),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
            patch("backend.cli.time.sleep"),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=True, mark_first=False)
        self.assertEqual(calls["n"], 3)
        captured = buf.getvalue()
        out = _last_json(captured)
        self.assertEqual(out.get("command"), "paper:run", msg=captured)
        refresh = out.get("refresh") or {}
        self.assertEqual((refresh.get("status_by_step") or {}).get("news:pull"), "ok")
        self.assertEqual(int((refresh.get("retry_counts") or {}).get("news:pull", -1)), 2)
        self.assertIn("aggression", out)

    def test_paper_run_retry_exhausted_marks_step_error(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_retry_error"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_retry_err.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=list(config["symbols"]),
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
        )
        decision = {
            "symbol": "BTC-USD",
            "decision": {"action": "SHORT", "confidence": 0.6, "validity_hours": 8, "conviction_label": "medium"},
            "strategy_inputs": {
                "regime_classification": {"market_regime": "range", "confidence": 0.5},
                "signal_quality": {
                    "agreement_score": 0.7,
                    "freshness_score": 0.8,
                    "effective_confidence": 0.6,
                    "event_risk": "normal",
                },
            },
            "no_trade_gate": {"active": False, "reasons": []},
            "paper_trade_preview": {"hypothetical_action": "SHORT"},
        }

        def _news_pull(*args, **kwargs):
            raise RuntimeError('TransactionContext Error: Failed to commit: write-write conflict on key: "u"')

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.fundamentals_pull"),
            patch("backend.cli.news_pull", side_effect=_news_pull),
            patch("backend.cli.fed_pull"),
            patch("backend.cli.sentiment_pull"),
            patch("backend.cli.calendar_pull"),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
            patch("backend.cli.time.sleep"),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=True, mark_first=False)
        captured = buf.getvalue()
        out = _last_json(captured)
        self.assertEqual(out.get("command"), "paper:run", msg=captured)
        refresh = out.get("refresh") or {}
        self.assertEqual((refresh.get("status_by_step") or {}).get("news:pull"), "error")
        self.assertEqual(int((refresh.get("retry_counts") or {}).get("news:pull", -1)), 3)

    def test_paper_run_skips_news_pull_within_min_interval(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_news_skip_interval"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_news_skip_interval.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=list(config["symbols"]),
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = {
            "symbol": "BTC-USD",
            "decision": {"action": "SHORT", "confidence": 0.6, "validity_hours": 8, "conviction_label": "medium"},
            "strategy_inputs": {
                "regime_classification": {"market_regime": "range", "confidence": 0.5},
                "signal_quality": {
                    "agreement_score": 0.7,
                    "freshness_score": 0.8,
                    "effective_confidence": 0.6,
                    "event_risk": "normal",
                },
            },
            "no_trade_gate": {"active": False, "reasons": []},
            "paper_trade_preview": {"hypothetical_action": "SHORT"},
        }
        now = datetime.now(timezone.utc)
        self.con.execute("INSERT INTO news_pull_log (pulled_at_utc) VALUES (?)", [now])

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.fundamentals_pull"),
            patch("backend.cli.news_pull") as mock_news_pull,
            patch("backend.cli.fed_pull"),
            patch("backend.cli.sentiment_pull"),
            patch("backend.cli.calendar_pull"),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=True, mark_first=False)
        out = _last_json(buf.getvalue())
        refresh = out.get("refresh") or {}
        self.assertEqual((refresh.get("status_by_step") or {}).get("news:pull"), "skipped")
        self.assertEqual(int((refresh.get("retry_counts") or {}).get("news:pull", -1)), 0)
        self.assertFalse(mock_news_pull.called)
        self.assertIn("intelligence", out)

    def test_paper_run_output_includes_aggression_block(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_aggr"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_aggr.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        self.assertEqual(out.get("command"), "paper:run")
        self.assertIn("replay", out)
        aggr = out.get("aggression") or {}
        self.assertIn(aggr.get("tier"), {"very_defensive", "defensive", "balanced", "assertive", "very_aggressive"})
        self.assertIn(aggr.get("source"), {"gpt", "deterministic_fallback"})
        self.assertIn("intelligence", out)
        candidates = list(out.get("candidates") or [])
        self.assertGreaterEqual(len(candidates), 1)
        self.assertIn("entry_eligible", candidates[0])
        self.assertIn("entry_blockers", candidates[0])
        self.assertIn("prediction", candidates[0])
        self.assertIn("patterns", candidates[0])
        self.assertIn("intel_score_delta", candidates[0])
        self.assertIn("intel_blockers", candidates[0])
        self.assertIn("intel_used_for_entry", candidates[0])

    def test_paper_run_output_includes_replay_summary(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_replay_summary"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_replay_summary.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=5,
            paper_max_open_positions_per_symbol=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        replay_stub = {
            "interval": "15m",
            "window_from": "2026-02-20T00:00:00Z",
            "window_to": "2026-02-20T01:00:00Z",
            "positions_checked": 1,
            "exits_replayed": 1,
            "events": [{"symbol": "BTC-USD", "trigger_type": "TP_HIT_REPLAY"}],
            "exits_triggered": [{"position_id": "p1", "symbol": "BTC-USD", "reason": "TP_HIT_REPLAY"}],
        }
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.services.paper_engine.replay_open_positions", return_value=replay_stub),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        self.assertEqual(int((out.get("replay") or {}).get("exits_replayed") or 0), 1)
        self.assertEqual(int((out.get("run_summary") or {}).get("replay_exits_count") or 0), 1)
        self.assertIn("intelligence", out)

    def test_paper_run_no_eligible_candidates_sets_reason_and_summary(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_no_eligible"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_no_eligible.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD", "ETH-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decisions = [_decision(action="NO_TRADE", gate_active=True, hyp="SHORT"), _decision(action="NO_TRADE", gate_active=True, hyp="SHORT")]
        decisions[1]["symbol"] = "ETH-USD"
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", side_effect=decisions),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD,ETH-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        self.assertIn("no_eligible_candidates", list(out.get("reasons") or []))
        summary = out.get("run_summary") or {}
        self.assertEqual(summary.get("selection_status"), "none")
        self.assertEqual(int(summary.get("eligible_candidates") or 0), 0)
        self.assertGreaterEqual(int(summary.get("blocked_candidates") or 0), 1)
        self.assertIsNone(out.get("selected"))

    def test_paper_run_rescue_lane_selects_candidate_when_standard_filters_fail(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_rescue_lane"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_rescue_lane.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        decision["decision"]["confidence"] = 0.37
        decision["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.58

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        run_summary = out.get("run_summary") or {}
        self.assertEqual(run_summary.get("selection_status"), "rescue_selected")
        self.assertTrue(bool(run_summary.get("rescue_lane_applicable")))
        self.assertEqual(str(run_summary.get("rescue_selected") or ""), "BTC-USD")
        self.assertEqual(int(run_summary.get("eligible_after_entry_filters") or 0), 0)
        self.assertEqual(int(len(run_summary.get("rescue_candidates") or [])), 1)
        self.assertFalse(bool(out.get("placed_trade")))
        selected = out.get("selected") or {}
        self.assertEqual(str(selected.get("entry_mode") or ""), "rescue")
        self.assertTrue(bool(selected.get("entry_mode_score")))

    def test_paper_run_rescue_not_applicable_when_gate_active(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_rescue_none"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_rescue_none.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=True, hyp="SHORT")
        decision["decision"]["confidence"] = 0.37
        decision["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.58
        decision["no_trade_gate"]["hard_blockers"] = ["CRITICAL_LOW_CONFIDENCE"]
        decision["no_trade_gate"]["reasons"] = [{"code": "LOW_CONFIDENCE", "detail": "blocked"}]

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        run_summary = out.get("run_summary") or {}
        self.assertEqual(run_summary.get("selection_status"), "none")
        self.assertFalse(bool(run_summary.get("rescue_lane_applicable")))
        self.assertIsNone(run_summary.get("rescue_selected"))
        self.assertFalse(out.get("placed_trade"))

    def test_paper_run_rescue_blocked_on_weekend_with_guard(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_rescue_weekend_guard"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_rescue_weekend_guard.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
            paper_weekend_rescue_guard=True,
            paper_weekend_rescue_notional_cap=0.03,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        decision["decision"]["confidence"] = 0.37
        decision["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.58

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
            patch("backend.cli._is_weekend_utc", return_value=True),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        run_summary = out.get("run_summary") or {}
        self.assertEqual(run_summary.get("selection_status"), "none")
        self.assertTrue(bool(run_summary.get("is_weekend_utc")))
        self.assertTrue(bool(run_summary.get("weekend_rescue_guard_active")))
        self.assertFalse(bool(run_summary.get("rescue_lane_applicable")))
        self.assertIsNone(run_summary.get("rescue_selected"))
        self.assertFalse(out.get("placed_trade"))

    def test_paper_run_rescue_uses_conservative_knobs(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_rescue_knobs"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_rescue_knobs.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        decision["decision"]["confidence"] = 0.37
        decision["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.58

        captured: dict = {}
        expected_plan = {
            "entry_price": 100.0,
            "stop_price": 99.0,
            "take_profit_price": 103.0,
            "time_stop_ts": datetime.now(timezone.utc),
            "qty": 1.0,
            "risk_distance": 1.0,
            "risk_usd": 100.0,
            "stop_distance_pct": 0.001,
            "stop_method": "fallback_pct",
            "validity_hours_applied": 24,
            "aggression_knobs": {},
        }

        def _mock_plan(**kwargs):
            captured["aggression_knobs"] = dict(kwargs.get("aggression_knobs") or {})
            return dict(expected_plan)

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
                patch("backend.services.paper_engine.get_latest_mid_price", return_value=(100.0, None)),
            patch("backend.services.paper_engine.compute_position_plan", side_effect=_mock_plan),
            patch("backend.services.paper_engine.simulate_fill", return_value={
                "fill_price": 100.0,
                "fees_usd": 0.1,
                "slippage_usd": 0.2,
                "qty": 1.0,
                "type": "ENTRY",
            }),
            patch("backend.db.paper_repo.insert_position_and_entry_fill"),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=False, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        trade = out.get("trade") or {}
        self.assertEqual(str(trade.get("entry_mode") or ""), "rescue")
        self.assertTrue(bool(captured.get("aggression_knobs")))
        self.assertLessEqual(float(captured["aggression_knobs"].get("risk_mult") or 1.0), 0.70 + 1e-9)
        self.assertGreaterEqual(float(captured["aggression_knobs"].get("stop_mult") or 0.0), 1.30 - 1e-9)
        self.assertLessEqual(float(captured["aggression_knobs"].get("hold_mult") or 1.0), 0.90 + 1e-9)
        self.assertLessEqual(float(captured["aggression_knobs"].get("notional_cap") or 1.0), 0.06 + 1e-9)

    def test_paper_run_rescue_weekend_notional_cap_is_respected(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_rescue_weekend_cap"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_rescue_weekend_cap.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
            paper_weekend_rescue_guard=False,
            paper_weekend_rescue_notional_cap=0.03,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        decision["decision"]["confidence"] = 0.37
        decision["strategy_inputs"]["signal_quality"]["agreement_score"] = 0.58

        captured: dict = {}
        expected_plan = {
            "entry_price": 100.0,
            "stop_price": 99.0,
            "take_profit_price": 103.0,
            "time_stop_ts": datetime.now(timezone.utc),
            "qty": 1.0,
            "risk_distance": 1.0,
            "risk_usd": 100.0,
            "stop_distance_pct": 0.001,
            "stop_method": "fallback_pct",
            "validity_hours_applied": 24,
            "aggression_knobs": {},
        }

        def _mock_plan(**kwargs):
            captured["aggression_knobs"] = dict(kwargs.get("aggression_knobs") or {})
            return dict(expected_plan)

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
            patch("backend.cli._is_weekend_utc", return_value=True),
            patch("backend.services.paper_engine.get_latest_mid_price", return_value=(100.0, None)),
            patch("backend.services.paper_engine.compute_position_plan", side_effect=_mock_plan),
            patch("backend.services.paper_engine.simulate_fill", return_value={
                "fill_price": 100.0,
                "fees_usd": 0.1,
                "slippage_usd": 0.2,
                "qty": 1.0,
                "type": "ENTRY",
            }),
            patch("backend.db.paper_repo.insert_position_and_entry_fill"),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=False, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        trade = out.get("trade") or {}
        self.assertEqual(str(trade.get("entry_mode") or ""), "rescue")
        self.assertTrue(bool(captured.get("aggression_knobs")))
        self.assertLessEqual(float(captured["aggression_knobs"].get("notional_cap") or 1.0), 0.03 + 1e-9)

    def test_paper_run_enforces_symbol_position_cap(self):
        config = _base_config()
        config["config_id"] = "cfg_test_symbol_cap"
        config["max_open_positions"] = 5
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_symbol_cap.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=5,
            paper_max_open_positions_per_symbol=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        open_pos = [
            {
                "position_id": "p1",
                "symbol": "BTC-USD",
                "side": "SHORT",
                "qty": 1.0,
                "entry_ts": datetime.now(timezone.utc),
                "entry_price": 100.0,
                "stop_price": 101.0,
                "take_profit_price": 98.0,
                "time_stop_ts": datetime.now(timezone.utc) + timedelta(hours=12),
                "status": "OPEN",
                "exit_ts": None,
                "exit_price": None,
                "exit_reason": None,
                "linked_run_id": "r0",
            }
        ]

        def _list_open_positions(_conn, symbol=None):
            if symbol is None:
                return list(open_pos)
            return list(open_pos) if symbol == "BTC-USD" else []

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.services.paper_engine.replay_open_positions", return_value={"interval": "15m", "window_from": None, "window_to": None, "positions_checked": 1, "exits_replayed": 0, "events": [], "exits_triggered": []}),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", side_effect=_list_open_positions),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=False, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        self.assertFalse(bool(out.get("placed_trade")))
        self.assertIn("symbol_position_cap_reached", list(out.get("reasons") or []))

    def test_paper_run_no_selection_uses_conservative_aggregate_aggression(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_no_select_aggr"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_no_select_aggr.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key="test-key",
            paper_symbols=["BTC-USD", "ETH-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decisions = [_decision(action="NO_TRADE", gate_active=True, hyp="SHORT"), _decision(action="NO_TRADE", gate_active=True, hyp="SHORT")]
        decisions[1]["symbol"] = "ETH-USD"
        gpt_profiles = [
            {"tier": "assertive", "override_weighted_gate": False, "quick_exit_bias": "none", "reasoning_summary": "assertive", "confidence": 0.8},
            {"tier": "very_defensive", "override_weighted_gate": False, "quick_exit_bias": "cut_loss", "reasoning_summary": "defensive", "confidence": 0.8},
        ]
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", side_effect=decisions),
            patch("backend.services.policy_ai.render_aggression_profile_gpt52", side_effect=gpt_profiles),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD,ETH-USD", max_trades=1, dry_run=True, refresh=False, mark_first=False)
        out = _last_json(buf.getvalue())
        aggr = out.get("aggression") or {}
        self.assertEqual(aggr.get("tier"), "very_defensive")
        self.assertIn("aggregate conservative posture", str(aggr.get("rationale") or ""))

    def test_paper_run_verbose_outputs_full_sections(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_verbose"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_verbose.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=["BTC-USD"],
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = _decision(action="HOLD", gate_active=False, hyp="SHORT")
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(
                    symbols="BTC-USD",
                    max_trades=1,
                    dry_run=True,
                    refresh=False,
                    mark_first=False,
                    verbose=True,
                )
        out = _last_json(buf.getvalue())
        self.assertIn("candidates_verbose", out)
        self.assertIn("selected_verbose", out)
        candidates = list(out.get("candidates") or [])
        self.assertGreaterEqual(len(candidates), 1)
        self.assertIn("aggression", candidates[0])
        self.assertIn("rationale_full", (candidates[0].get("aggression") or {}))
        self.assertIn("explanation", out)
        self.assertIn("position_timeline", out)
        self.assertIn("latest_position_run_age_runs", out.get("position_timeline") or {})

    def test_paper_run_skips_news_pull_on_daily_cap(self):
        config = _base_config()
        config["config_id"] = "cfg_test_run_news_skip_cap"
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_run_news_skip_cap.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_symbols=list(config["symbols"]),
            paper_starting_equity=10000.0,
            paper_fee_bps=5.0,
            paper_slippage_bps=8.0,
            paper_max_trades_per_run=1,
            paper_max_open_positions=1,
            paper_max_risk_per_trade_pct=0.01,
            paper_max_total_exposure_pct=0.60,
            paper_news_min_interval_minutes=60,
            paper_news_max_pulls_per_day=10,
        )
        decision = {
            "symbol": "BTC-USD",
            "decision": {"action": "SHORT", "confidence": 0.6, "validity_hours": 8, "conviction_label": "medium"},
            "strategy_inputs": {
                "regime_classification": {"market_regime": "range", "confidence": 0.5},
                "signal_quality": {
                    "agreement_score": 0.7,
                    "freshness_score": 0.8,
                    "effective_confidence": 0.6,
                    "event_risk": "normal",
                },
            },
            "no_trade_gate": {"active": False, "reasons": []},
            "paper_trade_preview": {"hypothetical_action": "SHORT"},
        }
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        for i in range(10):
            self.con.execute(
                "INSERT INTO news_pull_log (pulled_at_utc) VALUES (?)",
                [now - timedelta(minutes=i * 5)],
            )

        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_engine.smart_refresh_symbol", return_value=[]),
            patch("backend.cli.fundamentals_pull"),
            patch("backend.cli.news_pull") as mock_news_pull,
            patch("backend.cli.fed_pull"),
            patch("backend.cli.sentiment_pull"),
            patch("backend.cli.calendar_pull"),
            patch("backend.cli.run_decision_trident", return_value=decision),
            patch("backend.db.paper_repo.list_open_positions", return_value=[]),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_run_cmd(symbols="BTC-USD", max_trades=1, dry_run=True, refresh=True, mark_first=False)
        out = _last_json(buf.getvalue())
        refresh = out.get("refresh") or {}
        self.assertEqual((refresh.get("status_by_step") or {}).get("news:pull"), "skipped")
        self.assertEqual(int((refresh.get("retry_counts") or {}).get("news:pull", -1)), 0)
        self.assertFalse(mock_news_pull.called)

    def test_paper_report_fallback_includes_gpt_error(self):
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_report_test.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key="test-key",
        )
        report = {
            "window": "daily",
            "asof": "2026-02-19T00:00:00Z",
            "equity": {
                "starting_equity": 10000.0,
                "realized_gross": 0.0,
                "unrealized_gross": 0.0,
                "fees_total": 0.0,
                "equity": 10000.0,
                "peak_equity": 10000.0,
                "drawdown_pct": 0.0,
                "open_positions": 0,
                "closed_positions": 0,
            },
            "open_positions": [],
            "closed_trades": [],
            "per_symbol_stats": [],
            "gate_frequency": [],
        }
        config = _base_config()
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
            patch("backend.services.paper_reporting.build_paper_report", return_value=report),
            patch("backend.services.policy_ai.render_paper_report_summary_gpt52", return_value={"error": "payload_not_json_serializable"}),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                paper_report_cmd(daily=True, weekly=False, last=20, by_symbol=None, use_gpt=True)
        out = _last_json(buf.getvalue())
        self.assertEqual(out.get("command"), "paper:report")
        narrative = out.get("narrative") or {}
        self.assertEqual(narrative.get("source"), "deterministic_fallback")
        self.assertEqual(narrative.get("gpt_error"), "payload_not_json_serializable")

    def test_policy_ai_report_summary_accepts_datetime_payload(self):
        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"ok"}'))]
        )
        with patch("backend.services.policy_ai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = fake_resp
            out = render_paper_report_summary_gpt52(
                {"asof": datetime(2026, 2, 19, 12, 0, tzinfo=timezone.utc), "equity": {"equity": 10000.0}},
                model="gpt-5.2",
            )
        self.assertEqual(out.get("summary"), "ok")

    def test_paper_learn_output_includes_gpt_strategy_and_arbiter(self):
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_learn_test.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_gpt_learn_max_influence=0.30,
        )
        config = _base_config()
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                from backend.cli import paper_learn_cmd

                paper_learn_cmd(since=None, last=10, apply=False, explain=False, use_gpt=False)
        out = _last_json(buf.getvalue())
        self.assertEqual(out.get("command"), "paper:learn")
        self.assertIn("gpt_strategy", out)
        self.assertIn("arbiter", out)
        self.assertIn("stability_guardrails", out)
        self.assertIn("prediction_pattern_diagnostics", out)
        self.assertIn("smart_adjustment_report", out)
        self.assertIn("policy_candidates", out)
        self.assertIn("selected_policy_candidate", out)
        self.assertIn("projected_success_fail_ratio", out)
        self.assertIn("killswitch_state", out)
        self.assertIn("adjustment_run_id", out)

    def test_paper_learn_apply_respects_cooldown(self):
        now = datetime.now(timezone.utc)
        self.con.execute(
            """
            INSERT INTO paper_learning_events (
                learn_id, ts, scope, summary, changes_json, applied, diff_text, source_model
            ) VALUES ('learn_prev', ?, 'last_10', 'x', '{}', TRUE, 'd', NULL)
            """,
            [now],
        )
        settings = SimpleNamespace(
            database_path=Path("/tmp/paper_learn_cooldown.duckdb"),
            trident_gpt_model="gpt-5.2",
            openai_api_key=None,
            paper_gpt_learn_max_influence=0.30,
            paper_learn_apply_cooldown_hours=24,
            paper_gpt_learn_min_trades=25,
        )
        config = _base_config()
        with (
            patch("backend.cli.get_settings", return_value=settings),
            patch("backend.cli.get_connection", return_value=self.con),
            patch("backend.cli.apply_all_migrations"),
            patch("backend.db.paper_repo.get_active_paper_config", return_value=config),
        ):
            buf = StringIO()
            with redirect_stdout(buf):
                from backend.cli import paper_learn_cmd

                paper_learn_cmd(since=None, last=10, apply=True, explain=False, use_gpt=False)
        out = _last_json(buf.getvalue())
        self.assertEqual(out.get("command"), "paper:learn")
        self.assertFalse(bool(out.get("applied")))
        sg = out.get("stability_guardrails") or {}
        self.assertFalse(bool(sg.get("apply_cooldown_passed")))
        self.assertEqual(sg.get("apply_block_reason"), "cooldown_active")
        report = out.get("smart_adjustment_report") or {}
        apply_result = report.get("apply_result") or {}
        self.assertTrue(bool(apply_result.get("requested")))
        self.assertFalse(bool(apply_result.get("applied")))
        self.assertEqual(apply_result.get("apply_block_reason"), "cooldown_active")


if __name__ == "__main__":
    unittest.main()
