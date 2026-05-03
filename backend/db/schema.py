"""DuckDB schema management for Project Trident."""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import duckdb

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]


_WRITER_LOCK_STATE = {"depth": {}, "handles": {}}


def get_connection(database_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection at the given path. Create parent dirs if needed.
    """
    resolved_path = database_path.expanduser()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved_path))


@contextmanager
def writer_lock(
    database_path: Path,
    timeout_seconds: float = 120.0,
    poll_seconds: float = 0.20,
):
    """
    Global single-writer coordinator for DuckDB write paths.
    Uses a file lock and supports re-entrant usage in-process.
    """
    resolved = str(database_path.expanduser())
    depth = int(_WRITER_LOCK_STATE["depth"].get(resolved, 0))
    if depth > 0:
        _WRITER_LOCK_STATE["depth"][resolved] = depth + 1
        try:
            yield
        finally:
            next_depth = int(_WRITER_LOCK_STATE["depth"].get(resolved, 1)) - 1
            if next_depth <= 0:
                _WRITER_LOCK_STATE["depth"].pop(resolved, None)
            else:
                _WRITER_LOCK_STATE["depth"][resolved] = next_depth
        return

    lock_path = f"{resolved}.writer.lock"
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    acquired = False
    started = time.monotonic()
    try:
        if fcntl is not None:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if (time.monotonic() - started) >= float(timeout_seconds):
                        raise TimeoutError(
                            f"Timed out waiting for writer lock ({lock_path}) after {timeout_seconds:.1f}s"
                        )
                    time.sleep(max(0.01, float(poll_seconds)))
        else:
            # Fallback for environments lacking fcntl; still keeps in-process reentrancy.
            acquired = True
        _WRITER_LOCK_STATE["depth"][resolved] = 1
        _WRITER_LOCK_STATE["handles"][resolved] = handle
        yield
    finally:
        _WRITER_LOCK_STATE["depth"].pop(resolved, None)
        _WRITER_LOCK_STATE["handles"].pop(resolved, None)
        try:
            if acquired and fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass


def _migration_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"mig_{now}_{uuid.uuid4().hex[:8]}"


def _record_migration_step(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    step: str,
    required: bool,
    status: str,
    error_text: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO migration_journal
            (run_id, step, required, status, error_text, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [run_id, step, bool(required), status, error_text, datetime.now(timezone.utc)],
    )


def _run_migration_step(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    step: str,
    required: bool,
    fn: Callable[[], None],
) -> None:
    try:
        fn()
        _record_migration_step(conn, run_id, step, required, "ok", None)
    except Exception as exc:
        msg = str(exc)
        _record_migration_step(conn, run_id, step, required, "error", msg)
        if required:
            raise RuntimeError(f"Required migration failed: {step}: {msg}") from exc
        print(f"[db:migrate][warn] optional migration failed step={step} error={msg}", file=sys.stderr)


def apply_core_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create core tables (candles, fundamentals, news_items) if they do not exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol VARCHAR,
            interval VARCHAR,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY(symbol, interval, ts)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol VARCHAR,
            ts TIMESTAMP,
            key VARCHAR,
            value DOUBLE,
            PRIMARY KEY(symbol, ts, key)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals_history (
            symbol VARCHAR,
            ts TIMESTAMP,
            key VARCHAR,
            value DOUBLE,
            PRIMARY KEY(symbol, ts, key)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_items (
            symbol VARCHAR,
            published_at TIMESTAMP,
            source VARCHAR,
            url VARCHAR,
            title VARCHAR,
            polarity DOUBLE,
            ai_meta JSON,
            lane VARCHAR DEFAULT 'symbol',
            macro_tag VARCHAR,
            PRIMARY KEY(url)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_pull_log (
            pulled_at_utc TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fear_greed_index (
            ts_utc TIMESTAMP,
            value INTEGER,
            label VARCHAR,
            source VARCHAR,
            fetched_at_utc TIMESTAMP,
            PRIMARY KEY(ts_utc, source)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_ts_utc TIMESTAMP,
            symbol VARCHAR,
            run_id VARCHAR,
            snapshot_id VARCHAR,
            input_hash VARCHAR,
            confidence_overall DOUBLE,
            scenario_conf_best DOUBLE,
            scenario_conf_base DOUBLE,
            scenario_conf_worst DOUBLE,
            scenario_like_best DOUBLE,
            scenario_like_base DOUBLE,
            scenario_like_worst DOUBLE,
            scenario_intensity_best DOUBLE,
            scenario_intensity_base DOUBLE,
            scenario_intensity_worst DOUBLE,
            PRIMARY KEY(run_ts_utc, symbol)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS migration_journal (
            run_id VARCHAR,
            step VARCHAR,
            required BOOLEAN,
            status VARCHAR,
            error_text VARCHAR,
            created_at_utc TIMESTAMP,
            PRIMARY KEY(run_id, step)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_provenance (
            run_id VARCHAR PRIMARY KEY,
            snapshot_id VARCHAR,
            input_hash VARCHAR,
            command VARCHAR,
            symbol VARCHAR,
            asof_utc TIMESTAMP,
            status VARCHAR,
            notes VARCHAR,
            created_at_utc TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trident_decisions (
            asof_utc TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            action VARCHAR NOT NULL,
            confidence DOUBLE,
            conviction_label VARCHAR,
            validity_hours INTEGER,
            hypothetical_action VARCHAR,
            deadband_active BOOLEAN,
            blocked_by_json VARCHAR,
            regime_label VARCHAR,
            regime_confidence DOUBLE,
            top_scenario VARCHAR,
            top_likelihood DOUBLE,
            likelihood_margin DOUBLE,
            logic_version VARCHAR,
            run_id VARCHAR,
            snapshot_id VARCHAR,
            input_hash VARCHAR,
            persistence_status VARCHAR,
            payload_json VARCHAR NOT NULL,
            created_at_utc TIMESTAMP NOT NULL,
            PRIMARY KEY(asof_utc, symbol)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_fred_series (
            series_id VARCHAR NOT NULL,
            obs_date DATE NOT NULL,
            value DOUBLE,
            value_raw DOUBLE,
            value_norm DOUBLE,
            unit VARCHAR,
            multiplier DOUBLE,
            series VARCHAR,
            ts_utc TIMESTAMP,
            fetched_at_utc TIMESTAMP NOT NULL,
            source VARCHAR NOT NULL DEFAULT 'fred',
            PRIMARY KEY(series_id, obs_date)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS economic_calendar_events (
            provider VARCHAR NOT NULL,
            event_ts_utc TIMESTAMP NOT NULL,
            country VARCHAR,
            title VARCHAR NOT NULL,
            category VARCHAR,
            impact VARCHAR,
            macro_tags VARCHAR,
            actual DOUBLE,
            forecast DOUBLE,
            previous DOUBLE,
            unit VARCHAR,
            raw_json VARCHAR NOT NULL,
            fetched_at_utc TIMESTAMP NOT NULL,
            PRIMARY KEY(provider, event_ts_utc, country, title)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS economic_calendar_events_archive (
            provider VARCHAR NOT NULL,
            event_ts_utc TIMESTAMP NOT NULL,
            country VARCHAR,
            title VARCHAR NOT NULL,
            category VARCHAR,
            impact VARCHAR,
            macro_tags VARCHAR,
            actual DOUBLE,
            forecast DOUBLE,
            previous DOUBLE,
            unit VARCHAR,
            raw_json VARCHAR NOT NULL,
            fetched_at_utc TIMESTAMP NOT NULL,
            archived_at_utc TIMESTAMP NOT NULL,
            PRIMARY KEY(provider, event_ts_utc, country, title)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS economic_calendar_explainers (
            event_family VARCHAR NOT NULL,
            country VARCHAR,
            category VARCHAR,
            explanation VARCHAR NOT NULL,
            model VARCHAR,
            source VARCHAR,
            updated_at_utc TIMESTAMP NOT NULL,
            PRIMARY KEY(event_family, country, category)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_config (
            config_id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            active BOOLEAN,
            starting_equity DOUBLE,
            symbols_json VARCHAR,
            fee_bps DOUBLE,
            slippage_bps DOUBLE,
            max_trades_per_run INTEGER,
            max_open_positions INTEGER,
            risk_limits_json VARCHAR,
            learning_policy_json VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_runs (
            run_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            config_id VARCHAR,
            command VARCHAR,
            symbols_requested VARCHAR,
            refresh_mode VARCHAR,
            dry_run BOOLEAN,
            notes VARCHAR,
            status VARCHAR,
            error_text VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_decisions (
            run_id VARCHAR,
            symbol VARCHAR,
            asof_utc TIMESTAMP,
            decision_json VARCHAR,
            decision_hash VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_candidates (
            run_id VARCHAR,
            symbol VARCHAR,
            candidate_score DOUBLE,
            side VARCHAR,
            confidence DOUBLE,
            effective_confidence DOUBLE,
            agreement_score DOUBLE,
            freshness_score DOUBLE,
            gates_blocking_json VARCHAR,
            candidate_json VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_positions (
            position_id VARCHAR PRIMARY KEY,
            symbol VARCHAR,
            side VARCHAR,
            qty DOUBLE,
            entry_ts TIMESTAMP,
            entry_price DOUBLE,
            stop_price DOUBLE,
            take_profit_price DOUBLE,
            time_stop_ts TIMESTAMP,
            status VARCHAR,
            exit_ts TIMESTAMP,
            exit_price DOUBLE,
            exit_reason VARCHAR,
            linked_run_id VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_fills (
            fill_id VARCHAR PRIMARY KEY,
            position_id VARCHAR,
            ts TIMESTAMP,
            fill_price DOUBLE,
            fees_usd DOUBLE,
            slippage_usd DOUBLE,
            qty DOUBLE,
            type VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_marks (
            mark_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            symbol VARCHAR,
            mid_price DOUBLE,
            position_id VARCHAR,
            unrealized_pnl_usd DOUBLE,
            equity DOUBLE,
            drawdown_pct DOUBLE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_learning_events (
            learn_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            scope VARCHAR,
            summary VARCHAR,
            changes_json VARCHAR,
            applied BOOLEAN,
            diff_text VARCHAR,
            source_model VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_replay_events (
            event_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            command VARCHAR,
            run_id VARCHAR,
            position_id VARCHAR,
            symbol VARCHAR,
            side VARCHAR,
            interval VARCHAR,
            window_from TIMESTAMP,
            window_to TIMESTAMP,
            bar_ts TIMESTAMP,
            trigger_type VARCHAR,
            trigger_price DOUBLE,
            fill_price DOUBLE,
            gap_fill BOOLEAN,
            ambiguous_bar BOOLEAN,
            resolution_rule VARCHAR,
            notes VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_signal_audit (
            audit_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            run_id VARCHAR,
            symbol VARCHAR,
            mode VARCHAR,
            used_for_entry BOOLEAN,
            prediction_json VARCHAR,
            patterns_json VARCHAR,
            intel_score_delta DOUBLE,
            intel_blockers_json VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_adjustment_runs (
            adjustment_run_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            scope VARCHAR,
            selected_candidate_id VARCHAR,
            selected_score DOUBLE,
            selected_confidence DOUBLE,
            apply_requested BOOLEAN,
            applied BOOLEAN,
            apply_block_reason VARCHAR,
            kill_switch_active BOOLEAN,
            rollback_reference VARCHAR,
            prior_config_id VARCHAR,
            applied_config_id VARCHAR,
            summary_json VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_adjustment_candidates (
            candidate_id VARCHAR PRIMARY KEY,
            adjustment_run_id VARCHAR,
            rank INTEGER,
            candidate_type VARCHAR,
            score DOUBLE,
            confidence DOUBLE,
            projected_success_ratio DOUBLE,
            projected_fail_ratio DOUBLE,
            reasoning VARCHAR,
            recommended_risk_json VARCHAR,
            recommended_learning_json VARCHAR,
            gate_policy_changes_json VARCHAR,
            selected BOOLEAN,
            applied BOOLEAN
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_adjustment_rollbacks (
            rollback_id VARCHAR PRIMARY KEY,
            ts TIMESTAMP,
            adjustment_run_id VARCHAR,
            from_config_id VARCHAR,
            to_config_id VARCHAR,
            reason VARCHAR,
            details_json VARCHAR
        );
        """
    )


def apply_all_migrations(
    conn: duckdb.DuckDBPyConnection,
    skip_data_backfills: bool = False,
) -> None:
    """
    Apply all migrations with required/optional classification.
    Required failures raise and abort; optional failures are journaled and warned.
    """
    run_id = _migration_run_id()
    apply_core_schema(conn)
    _record_migration_step(conn, run_id, "core_schema", True, "ok", None)

    def _m001() -> None:
        conn.execute("ALTER TABLE candles ADD COLUMN IF NOT EXISTS interval VARCHAR;")
        if not skip_data_backfills:
            conn.execute("UPDATE candles SET interval = '1h' WHERE interval IS NULL;")

    def _m002() -> None:
        conn.execute("ALTER TABLE news_items ADD COLUMN IF NOT EXISTS ai_meta JSON;")
        conn.execute("ALTER TABLE news_items ADD COLUMN IF NOT EXISTS lane VARCHAR;")
        conn.execute("ALTER TABLE news_items ADD COLUMN IF NOT EXISTS macro_tag VARCHAR;")
        if not skip_data_backfills:
            conn.execute("UPDATE news_items SET lane = 'symbol' WHERE lane IS NULL;")

    def _m003_optional_macro_backfill() -> None:
        if skip_data_backfills:
            return
        import json
        from backend.services.news_api import classify_macro_tag

        rows = conn.execute(
            """
            SELECT url, title, ai_meta
            FROM news_items
            WHERE lane = 'macro' AND (macro_tag IS NULL OR macro_tag = '')
            """
        ).fetchall()
        for url, title, ai_meta in rows:
            summary = ""
            if ai_meta:
                try:
                    meta = ai_meta
                    if isinstance(ai_meta, str):
                        meta = json.loads(ai_meta)
                    summary = (meta or {}).get("summary") or ""
                except Exception:
                    summary = ""
            tag = classify_macro_tag(title or "", summary)
            if tag:
                conn.execute(
                    "UPDATE news_items SET macro_tag = ? WHERE url = ?",
                    [tag, url],
                )

    def _m004_required_calendar_cols() -> None:
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS category VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS impact VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS macro_tags VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS actual DOUBLE;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS forecast DOUBLE;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS previous DOUBLE;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS unit VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS raw_json VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events ADD COLUMN IF NOT EXISTS fetched_at_utc TIMESTAMP;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS category VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS impact VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS macro_tags VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS actual DOUBLE;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS forecast DOUBLE;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS previous DOUBLE;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS unit VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS raw_json VARCHAR;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS fetched_at_utc TIMESTAMP;")
        conn.execute("ALTER TABLE economic_calendar_events_archive ADD COLUMN IF NOT EXISTS archived_at_utc TIMESTAMP;")

    def _m005_required_analysis_cols() -> None:
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS run_id VARCHAR;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS snapshot_id VARCHAR;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS input_hash VARCHAR;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_like_best DOUBLE;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_like_base DOUBLE;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_like_worst DOUBLE;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_intensity_best DOUBLE;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_intensity_base DOUBLE;")
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS scenario_intensity_worst DOUBLE;")

    def _m006_required_decision_cols() -> None:
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS asof_utc TIMESTAMP;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS symbol VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS action VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS confidence DOUBLE;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS conviction_label VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS validity_hours INTEGER;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS hypothetical_action VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS deadband_active BOOLEAN;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS blocked_by_json VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS regime_label VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS regime_confidence DOUBLE;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS top_scenario VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS top_likelihood DOUBLE;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS likelihood_margin DOUBLE;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS logic_version VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS run_id VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS snapshot_id VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS input_hash VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS persistence_status VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS payload_json VARCHAR;")
        conn.execute("ALTER TABLE trident_decisions ADD COLUMN IF NOT EXISTS created_at_utc TIMESTAMP;")

    def _m007_optional_macro_fred_cols() -> None:
        conn.execute("ALTER TABLE macro_fred_series ADD COLUMN IF NOT EXISTS series VARCHAR;")
        conn.execute("ALTER TABLE macro_fred_series ADD COLUMN IF NOT EXISTS ts_utc TIMESTAMP;")
        conn.execute("ALTER TABLE macro_fred_series ADD COLUMN IF NOT EXISTS value_raw DOUBLE;")
        conn.execute("ALTER TABLE macro_fred_series ADD COLUMN IF NOT EXISTS value_norm DOUBLE;")
        conn.execute("ALTER TABLE macro_fred_series ADD COLUMN IF NOT EXISTS unit VARCHAR;")
        conn.execute("ALTER TABLE macro_fred_series ADD COLUMN IF NOT EXISTS multiplier DOUBLE;")
        conn.execute("UPDATE macro_fred_series SET series = series_id WHERE series IS NULL;")
        conn.execute("UPDATE macro_fred_series SET ts_utc = CAST(obs_date AS TIMESTAMP) WHERE ts_utc IS NULL;")
        conn.execute("UPDATE macro_fred_series SET value_raw = value WHERE value_raw IS NULL;")
        conn.execute(
            """
            UPDATE macro_fred_series
            SET multiplier = CASE
                WHEN series_id IN ('WALCL','WRESBAL') THEN 1e6
                WHEN series_id = 'RRPONTSYD' THEN 1e9
                ELSE multiplier
            END
            WHERE multiplier IS NULL;
            """
        )
        conn.execute("UPDATE macro_fred_series SET unit = 'USD' WHERE unit IS NULL;")
        conn.execute(
            """
            UPDATE macro_fred_series
            SET value_norm = value_raw * multiplier
            WHERE value_norm IS NULL
              AND value_raw IS NOT NULL
              AND multiplier IS NOT NULL;
            """
        )

    def _m008_optional_indexes() -> None:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_items_lane_published ON news_items (lane, published_at);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_items_lane_symbol_published ON news_items (lane, symbol, published_at);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fundamentals_history_symbol_key_ts ON fundamentals_history (symbol, key, ts);"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_migration_journal_created ON migration_journal (created_at_utc);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_runs_ts_status ON paper_runs (ts, command, status);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_decisions_symbol_asof ON paper_decisions (symbol, asof_utc);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_candidates_symbol_score ON paper_candidates (symbol, candidate_score);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_positions_symbol_status_entry ON paper_positions (symbol, status, entry_ts);"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_fills_position_ts_type ON paper_fills (position_id, ts, type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_marks_symbol_ts ON paper_marks (symbol, ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_learning_events_ts_scope ON paper_learning_events (ts, scope);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_replay_events_symbol_ts ON paper_replay_events (symbol, ts);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_replay_events_position_ts ON paper_replay_events (position_id, ts);"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_replay_events_command_ts ON paper_replay_events (command, ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_signal_audit_run_symbol ON paper_signal_audit (run_id, symbol);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_signal_audit_symbol_ts ON paper_signal_audit (symbol, ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_signal_audit_mode_ts ON paper_signal_audit (mode, ts);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_adjustment_runs_ts ON paper_adjustment_runs (ts, applied, kill_switch_active);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_adjustment_candidates_run_rank ON paper_adjustment_candidates (adjustment_run_id, rank);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_adjustment_rollbacks_ts ON paper_adjustment_rollbacks (ts, adjustment_run_id);"
        )

    _run_migration_step(conn, run_id, "m001_candles_interval", True, _m001)
    _run_migration_step(conn, run_id, "m002_news_columns", True, _m002)
    _run_migration_step(conn, run_id, "m003_news_macro_tag_backfill", False, _m003_optional_macro_backfill)
    _run_migration_step(conn, run_id, "m004_calendar_columns", True, _m004_required_calendar_cols)
    _run_migration_step(conn, run_id, "m005_analysis_columns", True, _m005_required_analysis_cols)
    _run_migration_step(conn, run_id, "m006_trident_decisions_columns", True, _m006_required_decision_cols)
    _run_migration_step(conn, run_id, "m007_macro_fred_columns", False, _m007_optional_macro_fred_cols)
    _run_migration_step(conn, run_id, "m008_indexes", False, _m008_optional_indexes)
