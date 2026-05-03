"""DuckDB repository helpers for paper trading ledger."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    s = str(value or "").strip()
    if not s:
        return _utcnow()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return _utcnow()


def _parse_ts_nullable(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return _parse_ts(value)


@contextmanager
def _transaction(conn):
    conn.execute("BEGIN TRANSACTION")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def _row_to_config(row: Optional[tuple]) -> Dict[str, Any]:
    if not row:
        return {}
    (
        config_id,
        created_at,
        active,
        starting_equity,
        symbols_json,
        fee_bps,
        slippage_bps,
        max_trades_per_run,
        max_open_positions,
        risk_limits_json,
        learning_policy_json,
    ) = row
    return {
        "config_id": str(config_id),
        "created_at": created_at,
        "active": bool(active),
        "starting_equity": float(starting_equity or 0.0),
        "symbols": _json_loads(symbols_json, []),
        "fee_bps": float(fee_bps or 0.0),
        "slippage_bps": float(slippage_bps or 0.0),
        "max_trades_per_run": int(max_trades_per_run or 0),
        "max_open_positions": int(max_open_positions or 0),
        "risk_limits": _json_loads(risk_limits_json, {}),
        "learning_policy": _json_loads(learning_policy_json, {}),
    }


def get_active_paper_config(conn) -> Dict[str, Any]:
    row = conn.execute(
        """
        SELECT config_id, created_at, active, starting_equity, symbols_json, fee_bps, slippage_bps,
               max_trades_per_run, max_open_positions, risk_limits_json, learning_policy_json
        FROM paper_config
        WHERE active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    return _row_to_config(row)


def upsert_paper_config(conn, config_dict: Dict[str, Any], set_active: bool = True) -> Dict[str, Any]:
    config_id = str(config_dict.get("config_id") or str(uuid.uuid4()))
    created_at = _parse_ts(config_dict.get("created_at") or _utcnow())
    starting_equity = float(config_dict.get("starting_equity") or 0.0)
    symbols = config_dict.get("symbols") or []
    fee_bps = float(config_dict.get("fee_bps") or 0.0)
    slippage_bps = float(config_dict.get("slippage_bps") or 0.0)
    max_trades_per_run = int(config_dict.get("max_trades_per_run") or 0)
    max_open_positions = int(config_dict.get("max_open_positions") or 0)
    risk_limits = config_dict.get("risk_limits") or {}
    learning_policy = config_dict.get("learning_policy") or {}
    active = bool(config_dict.get("active", True if set_active else False))

    if set_active:
        conn.execute("UPDATE paper_config SET active = FALSE WHERE active = TRUE")
        active = True

    conn.execute(
        """
        INSERT OR REPLACE INTO paper_config (
            config_id, created_at, active, starting_equity, symbols_json, fee_bps, slippage_bps,
            max_trades_per_run, max_open_positions, risk_limits_json, learning_policy_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            config_id,
            created_at,
            active,
            starting_equity,
            _json_dumps(symbols),
            fee_bps,
            slippage_bps,
            max_trades_per_run,
            max_open_positions,
            _json_dumps(risk_limits),
            _json_dumps(learning_policy),
        ],
    )
    row = conn.execute(
        """
        SELECT config_id, created_at, active, starting_equity, symbols_json, fee_bps, slippage_bps,
               max_trades_per_run, max_open_positions, risk_limits_json, learning_policy_json
        FROM paper_config
        WHERE config_id = ?
        LIMIT 1
        """,
        [config_id],
    ).fetchone()
    return _row_to_config(row)


def create_run(conn, command: str, flags_dict: Dict[str, Any], config_id: str) -> str:
    now = _utcnow()
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    symbols_requested = flags_dict.get("symbols_requested")
    if isinstance(symbols_requested, list):
        symbols_requested = ",".join(str(x) for x in symbols_requested)
    conn.execute(
        """
        INSERT INTO paper_runs (
            run_id, ts, config_id, command, symbols_requested, refresh_mode, dry_run, notes, status, error_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            now,
            config_id,
            command,
            str(symbols_requested or ""),
            str(flags_dict.get("refresh_mode") or ""),
            bool(flags_dict.get("dry_run", False)),
            str(flags_dict.get("notes") or ""),
            "running",
            None,
        ],
    )
    return run_id


def set_run_status(
    conn,
    run_id: str,
    status: str,
    error_text: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE paper_runs
        SET status = ?, error_text = ?, notes = COALESCE(?, notes)
        WHERE run_id = ?
        """,
        [status, error_text, notes, run_id],
    )


def record_error_run(conn, run_id: str, error_text: str) -> None:
    set_run_status(conn, run_id=run_id, status="error", error_text=str(error_text))


def record_decision(conn, run_id: str, symbol: str, decision_json: Dict[str, Any]) -> None:
    payload = _json_dumps(decision_json)
    decision_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    asof = _parse_ts(decision_json.get("asof"))
    conn.execute(
        """
        INSERT INTO paper_decisions (run_id, symbol, asof_utc, decision_json, decision_hash)
        VALUES (?, ?, ?, ?, ?)
        """,
        [run_id, symbol, asof, payload, decision_hash],
    )


def record_candidate(conn, run_id: str, candidate_dict: Dict[str, Any]) -> None:
    gates = candidate_dict.get("gates_blocking") or []
    conn.execute(
        """
        INSERT INTO paper_candidates (
            run_id, symbol, candidate_score, side, confidence, effective_confidence,
            agreement_score, freshness_score, gates_blocking_json, candidate_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            str(candidate_dict.get("symbol") or ""),
            float(candidate_dict.get("candidate_score") or 0.0),
            str(candidate_dict.get("side") or "NO_TRADE"),
            float(candidate_dict.get("confidence") or 0.0),
            float(candidate_dict.get("effective_confidence") or 0.0),
            float(candidate_dict.get("agreement_score") or 0.0),
            float(candidate_dict.get("freshness_score") or 0.0),
            _json_dumps(gates),
            _json_dumps(candidate_dict),
        ],
    )


def insert_position_and_entry_fill(conn, position_dict: Dict[str, Any], fill_dict: Dict[str, Any]) -> None:
    with _transaction(conn):
        conn.execute(
            """
            INSERT INTO paper_positions (
                position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                position_dict["position_id"],
                position_dict["symbol"],
                position_dict["side"],
                float(position_dict["qty"]),
                _parse_ts(position_dict["entry_ts"]),
                float(position_dict["entry_price"]),
                float(position_dict["stop_price"]),
                float(position_dict["take_profit_price"]) if position_dict.get("take_profit_price") is not None else None,
                _parse_ts(position_dict["time_stop_ts"]) if position_dict.get("time_stop_ts") is not None else None,
                str(position_dict.get("status") or "OPEN"),
                None,
                None,
                None,
                str(position_dict.get("linked_run_id") or ""),
            ],
        )
        conn.execute(
            """
            INSERT INTO paper_fills (
                fill_id, position_id, ts, fill_price, fees_usd, slippage_usd, qty, type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fill_dict["fill_id"],
                position_dict["position_id"],
                _parse_ts(fill_dict["ts"]),
                float(fill_dict["fill_price"]),
                float(fill_dict.get("fees_usd") or 0.0),
                float(fill_dict.get("slippage_usd") or 0.0),
                float(fill_dict["qty"]),
                str(fill_dict.get("type") or "ENTRY"),
            ],
        )


def close_position_and_exit_fill(
    conn,
    position_id: str,
    exit_dict: Dict[str, Any],
    fill_dict: Dict[str, Any],
) -> None:
    with _transaction(conn):
        conn.execute(
            """
            UPDATE paper_positions
            SET status = 'CLOSED',
                exit_ts = ?,
                exit_price = ?,
                exit_reason = ?
            WHERE position_id = ?
            """,
            [
                _parse_ts(exit_dict["exit_ts"]),
                float(exit_dict["exit_price"]),
                str(exit_dict.get("exit_reason") or ""),
                position_id,
            ],
        )
        conn.execute(
            """
            INSERT INTO paper_fills (
                fill_id, position_id, ts, fill_price, fees_usd, slippage_usd, qty, type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fill_dict["fill_id"],
                position_id,
                _parse_ts(fill_dict["ts"]),
                float(fill_dict["fill_price"]),
                float(fill_dict.get("fees_usd") or 0.0),
                float(fill_dict.get("slippage_usd") or 0.0),
                float(fill_dict["qty"]),
                str(fill_dict.get("type") or "EXIT"),
            ],
        )


def insert_mark(conn, mark_dict: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO paper_marks (
            mark_id, ts, symbol, mid_price, position_id, unrealized_pnl_usd, equity, drawdown_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(mark_dict["mark_id"]),
            _parse_ts(mark_dict["ts"]),
            str(mark_dict["symbol"]),
            float(mark_dict["mid_price"]),
            str(mark_dict["position_id"]),
            float(mark_dict.get("unrealized_pnl_usd") or 0.0),
            float(mark_dict.get("equity") or 0.0),
            float(mark_dict.get("drawdown_pct") or 0.0),
        ],
    )


def get_last_position_activity_ts(conn, position_id: str) -> Optional[datetime]:
    row = conn.execute(
        """
        SELECT MAX(ts) AS last_ts
        FROM (
            SELECT ts FROM paper_marks WHERE position_id = ?
            UNION ALL
            SELECT ts FROM paper_replay_events WHERE position_id = ?
        ) t
        """,
        [position_id, position_id],
    ).fetchone()
    if not row:
        return None
    return _parse_ts_nullable(row[0])


def record_replay_event(conn, event_dict: Dict[str, Any]) -> str:
    event_id = str(event_dict.get("event_id") or f"replay_{uuid.uuid4().hex}")
    conn.execute(
        """
        INSERT INTO paper_replay_events (
            event_id, ts, command, run_id, position_id, symbol, side, interval,
            window_from, window_to, bar_ts, trigger_type, trigger_price, fill_price,
            gap_fill, ambiguous_bar, resolution_rule, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            event_id,
            _parse_ts(event_dict.get("ts") or _utcnow()),
            str(event_dict.get("command") or ""),
            str(event_dict.get("run_id") or "") or None,
            str(event_dict.get("position_id") or ""),
            str(event_dict.get("symbol") or ""),
            str(event_dict.get("side") or ""),
            str(event_dict.get("interval") or ""),
            _parse_ts_nullable(event_dict.get("window_from")),
            _parse_ts_nullable(event_dict.get("window_to")),
            _parse_ts_nullable(event_dict.get("bar_ts")),
            str(event_dict.get("trigger_type") or ""),
            float(event_dict.get("trigger_price") or 0.0),
            float(event_dict.get("fill_price") or 0.0),
            bool(event_dict.get("gap_fill")),
            bool(event_dict.get("ambiguous_bar")),
            str(event_dict.get("resolution_rule") or ""),
            str(event_dict.get("notes") or ""),
        ],
    )
    return event_id


def record_signal_audit(conn, row_dict: Dict[str, Any]) -> str:
    audit_id = str(row_dict.get("audit_id") or f"sig_{uuid.uuid4().hex}")
    conn.execute(
        """
        INSERT INTO paper_signal_audit (
            audit_id, ts, run_id, symbol, mode, used_for_entry,
            prediction_json, patterns_json, intel_score_delta, intel_blockers_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            audit_id,
            _parse_ts(row_dict.get("ts") or _utcnow()),
            str(row_dict.get("run_id") or "") or None,
            str(row_dict.get("symbol") or ""),
            str(row_dict.get("mode") or ""),
            bool(row_dict.get("used_for_entry")),
            _json_dumps(row_dict.get("prediction") or {}),
            _json_dumps(row_dict.get("patterns") or {}),
            float(row_dict.get("intel_score_delta") or 0.0),
            _json_dumps(row_dict.get("intel_blockers") or []),
        ],
    )
    return audit_id


def list_signal_audit(
    conn,
    since_hours: Optional[int] = None,
    last_n: Optional[int] = None,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: List[Any] = []
    sql = """
        SELECT audit_id, ts, run_id, symbol, mode, used_for_entry,
               prediction_json, patterns_json, intel_score_delta, intel_blockers_json
        FROM paper_signal_audit
        WHERE 1=1
    """
    if symbol:
        sql += " AND symbol = ?"
        params.append(str(symbol))
    if since_hours is not None and since_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))
        sql += " AND ts >= ?"
        params.append(cutoff)
    sql += " ORDER BY ts DESC"
    if last_n is not None and last_n > 0:
        sql += " LIMIT ?"
        params.append(int(last_n))

    rows = conn.execute(sql, params).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "audit_id": str(row[0]),
                "ts": row[1],
                "run_id": row[2],
                "symbol": row[3],
                "mode": row[4],
                "used_for_entry": bool(row[5]),
                "prediction": _json_loads(row[6], {}),
                "patterns": _json_loads(row[7], {}),
                "intel_score_delta": float(row[8] or 0.0),
                "intel_blockers": _json_loads(row[9], []),
            }
        )
    return out


def get_last_applied_learning_event_ts(conn) -> Optional[datetime]:
    row = conn.execute(
        """
        SELECT MAX(ts)
        FROM paper_learning_events
        WHERE applied = TRUE
        """
    ).fetchone()
    if not row:
        return None
    return _parse_ts_nullable(row[0])


def record_adjustment_run(conn, row_dict: Dict[str, Any]) -> str:
    adjustment_run_id = str(row_dict.get("adjustment_run_id") or f"adj_{uuid.uuid4().hex}")
    conn.execute(
        """
        INSERT INTO paper_adjustment_runs (
            adjustment_run_id, ts, scope, selected_candidate_id, selected_score, selected_confidence,
            apply_requested, applied, apply_block_reason, kill_switch_active, rollback_reference,
            prior_config_id, applied_config_id, summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            adjustment_run_id,
            _parse_ts(row_dict.get("ts") or _utcnow()),
            str(row_dict.get("scope") or ""),
            str(row_dict.get("selected_candidate_id") or "") or None,
            float(row_dict.get("selected_score") or 0.0),
            float(row_dict.get("selected_confidence") or 0.0),
            bool(row_dict.get("apply_requested")),
            bool(row_dict.get("applied")),
            str(row_dict.get("apply_block_reason") or "") or None,
            bool(row_dict.get("kill_switch_active")),
            str(row_dict.get("rollback_reference") or "") or None,
            str(row_dict.get("prior_config_id") or "") or None,
            str(row_dict.get("applied_config_id") or "") or None,
            _json_dumps(row_dict.get("summary") or {}),
        ],
    )
    return adjustment_run_id


def record_adjustment_candidates(
    conn,
    adjustment_run_id: str,
    candidates: List[Dict[str, Any]],
) -> List[str]:
    ids: List[str] = []
    for idx, cand in enumerate(candidates or [], start=1):
        base_candidate_id = str(cand.get("candidate_id") or f"cand_{uuid.uuid4().hex}")
        candidate_id = f"{str(adjustment_run_id)}::{base_candidate_id}"
        ids.append(candidate_id)
        conn.execute(
            """
            INSERT INTO paper_adjustment_candidates (
                candidate_id, adjustment_run_id, rank, candidate_type, score, confidence,
                projected_success_ratio, projected_fail_ratio, reasoning,
                recommended_risk_json, recommended_learning_json, gate_policy_changes_json,
                selected, applied
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                candidate_id,
                str(adjustment_run_id),
                int(cand.get("rank") or idx),
                str(cand.get("candidate_type") or ""),
                float(cand.get("score") or 0.0),
                float(cand.get("confidence") or 0.0),
                float(cand.get("projected_success_ratio") or 0.0),
                float(cand.get("projected_fail_ratio") or 0.0),
                str(cand.get("reasoning") or ""),
                _json_dumps(cand.get("risk_limits") or {}),
                _json_dumps(cand.get("learning_policy") or {}),
                _json_dumps(cand.get("gate_policy_changes") or {}),
                bool(cand.get("selected")),
                bool(cand.get("applied")),
            ],
        )
    return ids


def record_adjustment_rollback(
    conn,
    adjustment_run_id: str,
    from_config_id: Optional[str],
    to_config_id: Optional[str],
    reason: str,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    rollback_id = f"rollback_{uuid.uuid4().hex}"
    conn.execute(
        """
        INSERT INTO paper_adjustment_rollbacks (
            rollback_id, ts, adjustment_run_id, from_config_id, to_config_id, reason, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            rollback_id,
            _utcnow(),
            str(adjustment_run_id or ""),
            str(from_config_id or "") or None,
            str(to_config_id or "") or None,
            str(reason or ""),
            _json_dumps(details or {}),
        ],
    )
    return rollback_id


def get_last_stable_paper_config(conn) -> Dict[str, Any]:
    """
    Return the most recent applied non-killswitched config if available.
    Falls back to current active config.
    """
    row = conn.execute(
        """
        SELECT applied_config_id
        FROM paper_adjustment_runs
        WHERE applied = TRUE
          AND kill_switch_active = FALSE
          AND applied_config_id IS NOT NULL
          AND applied_config_id <> ''
        ORDER BY ts DESC
        LIMIT 1
        """
    ).fetchone()
    if not row or not row[0]:
        return get_active_paper_config(conn)
    cfg_row = conn.execute(
        """
        SELECT config_id, created_at, active, starting_equity, symbols_json, fee_bps, slippage_bps,
               max_trades_per_run, max_open_positions, risk_limits_json, learning_policy_json
        FROM paper_config
        WHERE config_id = ?
        LIMIT 1
        """,
        [str(row[0])],
    ).fetchone()
    if not cfg_row:
        return get_active_paper_config(conn)
    return _row_to_config(cfg_row)


def list_open_positions(conn, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    if symbol:
        rows = conn.execute(
            """
            SELECT position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                   time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            FROM paper_positions
            WHERE status = 'OPEN' AND symbol = ?
            ORDER BY entry_ts ASC
            """,
            [symbol],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT position_id, symbol, side, qty, entry_ts, entry_price, stop_price, take_profit_price,
                   time_stop_ts, status, exit_ts, exit_price, exit_reason, linked_run_id
            FROM paper_positions
            WHERE status = 'OPEN'
            ORDER BY entry_ts ASC
            """
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "position_id": row[0],
                "symbol": row[1],
                "side": row[2],
                "qty": float(row[3] or 0.0),
                "entry_ts": row[4],
                "entry_price": float(row[5] or 0.0),
                "stop_price": float(row[6] or 0.0),
                "take_profit_price": float(row[7]) if row[7] is not None else None,
                "time_stop_ts": row[8],
                "status": row[9],
                "exit_ts": row[10],
                "exit_price": float(row[11]) if row[11] is not None else None,
                "exit_reason": row[12],
                "linked_run_id": row[13],
            }
        )
    return out


def compute_equity_snapshot(conn, starting_equity: float) -> Dict[str, Any]:
    realized = conn.execute(
        """
        SELECT COALESCE(SUM(
            CASE
                WHEN side = 'LONG' THEN (exit_price - entry_price) * qty
                WHEN side = 'SHORT' THEN (entry_price - exit_price) * qty
                ELSE 0
            END
        ), 0.0)
        FROM paper_positions
        WHERE status = 'CLOSED' AND exit_price IS NOT NULL
        """
    ).fetchone()[0]
    fees_total = conn.execute("SELECT COALESCE(SUM(fees_usd), 0.0) FROM paper_fills").fetchone()[0]
    unrealized = conn.execute(
        """
        SELECT COALESCE(SUM(m.unrealized_pnl_usd), 0.0)
        FROM paper_marks m
        INNER JOIN (
            SELECT position_id, MAX(ts) AS max_ts
            FROM paper_marks
            GROUP BY position_id
        ) latest
        ON m.position_id = latest.position_id AND m.ts = latest.max_ts
        INNER JOIN paper_positions p ON p.position_id = m.position_id
        WHERE p.status = 'OPEN'
        """
    ).fetchone()[0]
    realized = float(realized or 0.0)
    fees_total = float(fees_total or 0.0)
    unrealized = float(unrealized or 0.0)
    equity = float(starting_equity) + realized + unrealized - fees_total
    peak_mark = conn.execute("SELECT COALESCE(MAX(equity), ?) FROM paper_marks", [float(starting_equity)]).fetchone()[0]
    peak_equity = max(float(starting_equity), float(peak_mark or starting_equity), equity)
    drawdown_pct = 0.0
    if peak_equity > 0:
        drawdown_pct = max(0.0, (peak_equity - equity) / peak_equity)
    open_count = conn.execute("SELECT COUNT(*) FROM paper_positions WHERE status = 'OPEN'").fetchone()[0]
    closed_count = conn.execute("SELECT COUNT(*) FROM paper_positions WHERE status = 'CLOSED'").fetchone()[0]
    return {
        "starting_equity": float(starting_equity),
        "realized_gross": round(realized, 6),
        "unrealized_gross": round(unrealized, 6),
        "fees_total": round(fees_total, 6),
        "equity": round(equity, 6),
        "peak_equity": round(peak_equity, 6),
        "drawdown_pct": round(drawdown_pct, 6),
        "open_positions": int(open_count or 0),
        "closed_positions": int(closed_count or 0),
    }


def reset_paper_ledger(conn) -> List[str]:
    tables = [
        "paper_runs",
        "paper_decisions",
        "paper_candidates",
        "paper_positions",
        "paper_fills",
        "paper_marks",
        "paper_replay_events",
        "paper_signal_audit",
        "paper_learning_events",
        "paper_config",
    ]
    for table in tables:
        conn.execute(f"DELETE FROM {table}")
    return tables
