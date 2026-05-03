"""Reporting helpers for paper trading ledger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from backend.db.paper_repo import compute_equity_snapshot, get_active_paper_config, list_open_positions


def _json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _window_start(daily: bool, weekly: bool, since_hours: Optional[int] = None) -> datetime:
    now = datetime.now(timezone.utc)
    if since_hours is not None and since_hours > 0:
        return now - timedelta(hours=int(since_hours))
    if weekly:
        return now - timedelta(days=7)
    if daily:
        return now - timedelta(days=1)
    return now - timedelta(days=7)


def _gate_frequency(conn, since_ts: datetime, by_symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    if by_symbol:
        rows = conn.execute(
            """
            SELECT c.gates_blocking_json
            FROM paper_candidates c
            INNER JOIN paper_runs r ON r.run_id = c.run_id
            WHERE c.symbol = ? AND r.ts >= ?
            """,
            [by_symbol, since_ts],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT c.gates_blocking_json
            FROM paper_candidates c
            INNER JOIN paper_runs r ON r.run_id = c.run_id
            WHERE r.ts >= ?
            """,
            [since_ts],
        ).fetchall()

    counts: Dict[str, int] = {}
    for row in rows:
        gates = _json_loads(row[0], [])
        for code in gates:
            key = str(code or "").strip()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    out = [{"code": code, "count": count} for code, count in counts.items()]
    out.sort(key=lambda x: (-int(x["count"]), str(x["code"])))
    return out


def _recent_closed_trades(conn, since_ts: datetime, last: int, by_symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    params: List[Any] = [since_ts, int(last)]
    sql = """
        SELECT position_id, symbol, side, qty, entry_ts, entry_price, exit_ts, exit_price, exit_reason
        FROM paper_positions
        WHERE status = 'CLOSED' AND exit_ts >= ?
    """
    if by_symbol:
        sql += " AND symbol = ?"
        params = [since_ts, by_symbol, int(last)]
    sql += " ORDER BY exit_ts DESC LIMIT ?"
    rows = conn.execute(sql, params).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        side = str(row[2] or "").upper()
        qty = float(row[3] or 0.0)
        entry = float(row[5] or 0.0)
        exit_px = float(row[7] or 0.0)
        if side == "LONG":
            pnl = (exit_px - entry) * qty
        else:
            pnl = (entry - exit_px) * qty
        out.append(
            {
                "position_id": row[0],
                "symbol": row[1],
                "side": side,
                "qty": qty,
                "entry_ts": row[4],
                "entry_price": entry,
                "exit_ts": row[6],
                "exit_price": exit_px,
                "exit_reason": row[8],
                "pnl_gross": round(pnl, 6),
            }
        )
    return out


def _per_symbol_stats(conn, since_ts: datetime) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT symbol, side, qty, entry_price, exit_price, status
        FROM paper_positions
        WHERE (entry_ts >= ? OR exit_ts >= ?)
        """,
        [since_ts, since_ts],
    ).fetchall()
    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        symbol = str(row[0] or "")
        side = str(row[1] or "").upper()
        qty = float(row[2] or 0.0)
        entry = float(row[3] or 0.0)
        exit_px = float(row[4]) if row[4] is not None else None
        status = str(row[5] or "")
        if symbol not in stats:
            stats[symbol] = {
                "symbol": symbol,
                "trades": 0,
                "closed_trades": 0,
                "open_trades": 0,
                "gross_pnl": 0.0,
                "wins": 0,
                "losses": 0,
            }
        s = stats[symbol]
        s["trades"] += 1
        if status == "OPEN":
            s["open_trades"] += 1
            continue
        s["closed_trades"] += 1
        if exit_px is None:
            continue
        pnl = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
        s["gross_pnl"] += pnl
        if pnl > 0:
            s["wins"] += 1
        elif pnl < 0:
            s["losses"] += 1

    out = list(stats.values())
    out.sort(key=lambda x: str(x["symbol"]))
    for row in out:
        row["gross_pnl"] = round(float(row["gross_pnl"]), 6)
    return out


def build_paper_report(
    conn,
    config: Dict[str, Any],
    *,
    daily: bool = False,
    weekly: bool = False,
    last: int = 20,
    by_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    since_ts = _window_start(daily=daily, weekly=weekly)
    equity = compute_equity_snapshot(conn, float(config.get("starting_equity") or 10000.0))
    open_positions = list_open_positions(conn, symbol=by_symbol)
    closed_trades = _recent_closed_trades(conn, since_ts=since_ts, last=last, by_symbol=by_symbol)
    gate_frequency = _gate_frequency(conn, since_ts=since_ts, by_symbol=by_symbol)
    per_symbol = _per_symbol_stats(conn, since_ts=since_ts)
    window = "daily" if daily else "weekly" if weekly else f"last_{last}"
    return {
        "window": window,
        "asof": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "equity": equity,
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "per_symbol_stats": per_symbol,
        "gate_frequency": gate_frequency,
    }


def build_paper_status(conn) -> Dict[str, Any]:
    config = get_active_paper_config(conn)
    if not config:
        return {
            "asof": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "has_active_config": False,
        }
    equity = compute_equity_snapshot(conn, float(config.get("starting_equity") or 10000.0))
    open_positions = list_open_positions(conn)
    last_run = conn.execute(
        """
        SELECT run_id, ts, command, status, error_text
        FROM paper_runs
        ORDER BY ts DESC
        LIMIT 1
        """
    ).fetchone()
    return {
        "asof": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "has_active_config": True,
        "config_id": config.get("config_id"),
        "equity": equity,
        "open_positions": open_positions,
        "last_run": (
            {
                "run_id": last_run[0],
                "ts": last_run[1],
                "command": last_run[2],
                "status": last_run[3],
                "error_text": last_run[4],
            }
            if last_run
            else None
        ),
    }
