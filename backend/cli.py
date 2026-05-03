"""Project Trident CLI entrypoint."""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import calendar as _calendar
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config.env import Settings, get_settings
from backend.db.schema import apply_all_migrations, get_connection, writer_lock
from backend.decide.trident_decision import run_decision_trident
from backend.features.news_features import compute_news_features
from backend.features.signal_fusion import compute_signal_fusion
from backend.features.phase4_analysis import (
    build_phase4_snapshot,
    interpret_snapshot,
    evaluate_required_modality_status,
)
from backend.features.tech_features import compute_tech_features
from backend.features.fundamentals_features import compute_fundamentals_features

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Project Trident CLI")
SYSTEM_PHASE_VERSION = "6A.2"

EXPECTED_KEYS = [
    "ENV",
    "FEATURE_EQUITIES",
    "CRYPTO_SYMBOLS",
    "DATABASE_PATH",
    "DATA_VENDOR",
    "DISABLE_YF",
    "NEWS_VENDOR",
    "ONCHAIN_VENDOR",
    "AI_VENDOR",
    "AI_MODEL",
    "TRIDENT_USE_GPT",
    "TRIDENT_GPT_MODEL",
    "NEWSAPI_AI_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
    "COINGECKO_BASE",
    "KRAKEN_BASE",
    "OPENAI_API_KEY",
    "PAPER_STARTING_EQUITY",
    "PAPER_SYMBOLS",
    "PAPER_FEE_BPS",
    "PAPER_SLIPPAGE_BPS",
    "PAPER_MAX_TRADES_PER_RUN",
    "PAPER_MAX_OPEN_POSITIONS",
    "PAPER_MAX_OPEN_POSITIONS_PER_SYMBOL",
    "PAPER_MAX_RISK_PER_TRADE_PCT",
    "PAPER_MAX_TOTAL_EXPOSURE_PCT",
    "PAPER_REPLAY_INTERVAL",
    "PAPER_REPLAY_LOOKBACK_BARS",
    "PAPER_ENTRY_MIN_SCORE",
    "PAPER_ENTRY_MIN_EFFECTIVE_CONFIDENCE",
    "PAPER_ENTRY_MIN_AGREEMENT",
    "PAPER_ENTRY_MIN_MARGIN",
    "PAPER_QUALITY_VETO_ENABLED",
    "PAPER_GPT_LEARN_MAX_INFLUENCE",
    "PAPER_INTEL_MODE",
    "PAPER_INTEL_BOOTSTRAP_TRADES",
    "PAPER_INTEL_PROMOTION_TRADES",
    "PAPER_PREDICTION_ENABLED",
    "PAPER_PATTERN_ENABLED",
    "PAPER_PREDICTION_WEIGHT_SOFT",
    "PAPER_PATTERN_WEIGHT_SOFT",
    "PAPER_PREDICTION_WEIGHT_HARD",
    "PAPER_PATTERN_WEIGHT_HARD",
    "PAPER_ALLOW_WEIGHTED_GATE_OVERRIDE",
    "PAPER_LEARN_BOOTSTRAP_STOP_ONLY",
    "PAPER_LEARN_APPLY_COOLDOWN_HOURS",
    "PAPER_GPT_LEARN_MIN_TRADES",
    "PAPER_WEEKEND_RESCUE_GUARD",
    "PAPER_WEEKEND_RESCUE_NOTIONAL_CAP",
    "PAPER_ENTRY_RESCUE_ENABLED",
    "PAPER_ENTRY_RESCUE_MAX_PER_RUN",
    "PAPER_ENTRY_RESCUE_MIN_SCORE",
    "PAPER_ENTRY_RESCUE_MIN_EFFECTIVE_CONFIDENCE",
    "PAPER_ENTRY_RESCUE_MIN_AGREEMENT",
    "PAPER_ENTRY_RESCUE_MIN_MARGIN",
    "PAPER_ENTRY_RESCUE_NOTIONAL_CAP",
    "PAPER_ENTRY_RESCUE_RISK_MULT",
    "PAPER_ENTRY_RESCUE_STOP_MULT",
    "PAPER_ENTRY_RESCUE_HOLD_MULT",
    "PAPER_NEWS_MIN_INTERVAL_MINUTES",
    "PAPER_NEWS_MAX_PULLS_PER_DAY",
    "PAPER_DB_SCHEMA_VERSION",
]


def _mask_secret(value: Optional[str]) -> str:
    """Mask sensitive values for display."""
    if value is None:
        return "<missing>"
    if value == "":
        return "<empty>"
    visible = value[:2]
    masked_length = max(len(value) - 2, 4)
    return f"{visible}{'*' * masked_length}"


def _split_symbols(raw: str) -> List[str]:
    """Split comma-separated symbols provided to CLI flags."""
    return [sym.strip() for sym in raw.split(",") if sym.strip()]


def _plus_one_month_iso(from_iso: str) -> str:
    dt = datetime.strptime(from_iso, "%Y-%m-%d")
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    day = min(dt.day, _calendar.monthrange(year, month)[1])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _paper_run_ts(conn, run_id: str) -> Optional[datetime]:
    if not run_id:
        return None
    row = conn.execute("SELECT ts FROM paper_runs WHERE run_id = ?", [str(run_id)]).fetchone()
    return _to_utc_datetime(row[0]) if row else None


def _count_paper_run_steps_between(conn, since_ts: datetime, upto_ts: datetime) -> int:
    if not since_ts or not upto_ts:
        return 0
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM paper_runs
        WHERE command = 'paper:run'
          AND status IS NOT NULL
          AND status != 'error'
          AND ts >= ?
          AND ts <= ?
        """,
        [since_ts, upto_ts],
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _is_weekend_utc(now: Optional[datetime] = None) -> bool:
    """Return True when the provided UTC time is Saturday/Sunday."""
    ts = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).weekday() >= 5


def _paper_default_config(settings: Settings) -> dict:
    min_conf = 0.33
    return {
        "starting_equity": float(settings.paper_starting_equity),
        "symbols": list(settings.paper_symbols or ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "SUI-USD"]),
        "fee_bps": float(settings.paper_fee_bps),
        "slippage_bps": float(settings.paper_slippage_bps),
        "max_trades_per_run": int(settings.paper_max_trades_per_run),
        "max_open_positions": int(settings.paper_max_open_positions),
        "risk_limits": {
            "max_risk_per_trade_pct": float(settings.paper_max_risk_per_trade_pct),
            "max_total_exposure_pct": float(settings.paper_max_total_exposure_pct),
            "max_open_positions_per_symbol": int(settings.paper_max_open_positions_per_symbol),
            "replay_interval": str(settings.paper_replay_interval or "15m"),
            "replay_lookback_bars": int(settings.paper_replay_lookback_bars),
            "min_confidence": float(min_conf),
            "stop_distance_atr_mult": 1.0,
            "stop_distance_pct_fallback": 0.015,
            "entry_min_score": float(settings.paper_entry_min_score),
            "entry_min_effective_confidence": float(settings.paper_entry_min_effective_confidence),
            "entry_min_agreement": float(settings.paper_entry_min_agreement),
            "entry_min_margin": float(settings.paper_entry_min_margin),
            "quality_veto_enabled": bool(settings.paper_quality_veto_enabled),
            "intelligence_mode": str(settings.paper_intel_mode or "auto"),
            "intelligence_bootstrap_trades": int(settings.paper_intel_bootstrap_trades),
            "intelligence_promotion_trades": int(settings.paper_intel_promotion_trades),
            "prediction_enabled": bool(settings.paper_prediction_enabled),
            "pattern_enabled": bool(settings.paper_pattern_enabled),
            "prediction_weight_soft": float(settings.paper_prediction_weight_soft),
            "pattern_weight_soft": float(settings.paper_pattern_weight_soft),
            "prediction_weight_hard": float(settings.paper_prediction_weight_hard),
            "pattern_weight_hard": float(settings.paper_pattern_weight_hard),
            "allow_weighted_gate_override": bool(settings.paper_allow_weighted_gate_override),
            "entry_rescue_enabled": bool(settings.paper_entry_rescue_enabled),
            "entry_rescue_max_per_run": int(settings.paper_entry_rescue_max_per_run),
            "entry_rescue_min_score": float(settings.paper_entry_rescue_min_score),
            "entry_rescue_min_effective_confidence": float(settings.paper_entry_rescue_min_effective_confidence),
            "entry_rescue_min_agreement": float(settings.paper_entry_rescue_min_agreement),
            "entry_rescue_min_margin": float(settings.paper_entry_rescue_min_margin),
            "entry_rescue_notional_cap": float(settings.paper_entry_rescue_notional_cap),
            "entry_rescue_risk_mult": float(settings.paper_entry_rescue_risk_mult),
            "entry_rescue_stop_mult": float(settings.paper_entry_rescue_stop_mult),
            "entry_rescue_hold_mult": float(settings.paper_entry_rescue_hold_mult),
            "weekend_rescue_guard": bool(settings.paper_weekend_rescue_guard),
            "weekend_rescue_notional_cap": float(settings.paper_weekend_rescue_notional_cap),
        },
        "learning_policy": {
            "penalty_high_vol_chop": 0.9,
            "penalty_elevated_event_risk": 0.9,
            "gpt_learn_max_influence": float(settings.paper_gpt_learn_max_influence),
            "gpt_learn_min_trades": int(settings.paper_gpt_learn_min_trades),
            "learn_bootstrap_stop_only": bool(settings.paper_learn_bootstrap_stop_only),
            "learn_apply_cooldown_hours": int(settings.paper_learn_apply_cooldown_hours),
            "gate_overrides": {
                "min_confidence": float(min_conf),
                "gate_policy": {
                    "policy_version": "smart_adjust_v1_baseline",
                    "enabled_hard_checks": [
                        "CRITICAL_STALE_REQUIRED_MODALITY",
                        "CRITICAL_LOW_CONFIDENCE",
                    ],
                    "enabled_weighted_checks": [
                        "LOW_CONFIDENCE",
                        "LOW_HORIZON_ALIGNMENT",
                        "MODEL_EDGE_WEAK",
                        "RISK_CLUSTER_LOW_CONF",
                    ],
                    "diagnostic_only_checks": [
                        "LOW_AGREEMENT",
                        "LOW_BREADTH",
                        "RELATIVE_WEAKNESS",
                        "PRE_BREAKOUT_COMPRESSION",
                    ],
                    "merged_overlap_mode": True,
                },
            },
            "min_cohort_trades": int(settings.paper_adjust_min_cohort_trades),
            "structural_change_budget": int(settings.paper_adjust_structural_change_budget),
            "score_weights": {
                "expectancy": float(settings.paper_adjust_weight_expectancy),
                "win_rate": float(settings.paper_adjust_weight_win_rate),
                "drawdown_penalty": float(settings.paper_adjust_weight_drawdown_penalty),
                "instability_penalty": float(settings.paper_adjust_weight_instability_penalty),
            },
            "kill_switch": {
                "drawdown_spike": float(settings.paper_adjust_killswitch_drawdown_spike),
                "flip_rate": float(settings.paper_adjust_killswitch_flip_rate),
                "confidence_collapse": float(settings.paper_adjust_killswitch_conf_collapse),
            },
            "aggression_baseline": {
                "risk_mult": 1.0,
                "stop_mult": 1.0,
                "hold_mult": 1.0,
                "exposure_cap": 0.12,
            },
        },
    }


def _new_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _required_modality_gate(symbol: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    freshness = ((snapshot.get("meta_features") or {}).get("freshness") or {}) if isinstance(snapshot, dict) else {}
    status = ((snapshot.get("meta_features") or {}).get("required_modalities") or {}) if isinstance(snapshot, dict) else {}
    if not status:
        status = evaluate_required_modality_status(freshness)
    return {
        "symbol": symbol,
        "required_modality_status": status,
        "failed": not bool(status.get("ok", True)),
    }


def _required_modality_failure_payload(
    command: str,
    symbol: str,
    gate: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "command": command,
        "symbol": symbol,
        "status": "failed",
        "error": {
            "code": "STALE_REQUIRED_MODALITY",
            "message": "Required modalities are missing or stale (>=72h threshold).",
            "required_modality_status": gate.get("required_modality_status") or {},
        },
    }
    if isinstance(snapshot, dict):
        if snapshot.get("run_id") is not None:
            out["run_id"] = snapshot.get("run_id")
        if snapshot.get("snapshot_id") is not None:
            out["snapshot_id"] = snapshot.get("snapshot_id")
        if snapshot.get("input_hash") is not None:
            out["input_hash"] = snapshot.get("input_hash")
    return out


@app.command("trident:doctor")
@app.command("env:check")
def env_check(plain: bool = typer.Option(False, "--plain", help="Plain output")) -> None:
    """Validate environment configuration and display parsed values."""
    settings = get_settings()

    present_keys = [key for key in EXPECTED_KEYS if key in os.environ]
    missing_keys = [key for key in EXPECTED_KEYS if key not in os.environ]

    typer.echo("Environment status")
    typer.echo(f"- Present keys: {', '.join(present_keys) if present_keys else 'none'}")
    typer.echo(f"- Missing keys: {', '.join(missing_keys) if missing_keys else 'none'}")
    typer.echo("Parsed configuration")
    typer.echo(f"- env: {settings.env}")
    typer.echo(f"- feature_equities: {settings.feature_equities}")
    typer.echo(f"- crypto_symbols: {len(settings.crypto_symbols)} configured")
    typer.echo(f"- data_vendor: {settings.data_vendor}")
    typer.echo(f"- news_vendor: {settings.news_vendor}")
    typer.echo(f"- onchain_vendor: {settings.onchain_vendor}")
    typer.echo(f"- ai_vendor: {settings.ai_vendor}")
    typer.echo(f"- ai_model: {settings.ai_model}")
    typer.echo(f"- trident_use_gpt: {settings.trident_use_gpt}")
    typer.echo(f"- trident_gpt_model: {settings.trident_gpt_model}")
    typer.echo(f"- database_path: {settings.database_path}")
    typer.echo(f"- coingecko_base: {settings.coingecko_base}")
    typer.echo(f"- kraken_base: {settings.kraken_base}")
    typer.echo(f"- NEWSAPI_AI_KEY: {'<set>' if settings.newsapi_ai_key else '<missing>'}")
    typer.echo(f"- FINNHUB_API_KEY: {'<set>' if settings.finnhub_api_key else '<missing>'}")
    typer.echo(f"- FRED_API_KEY: {'<set>' if settings.fred_api_key else '<missing>'}")
    typer.echo(f"- OPENAI_API_KEY: {'<set>' if settings.openai_api_key else '<missing>'}")
    if plain:
        return


@app.command("trident:init")
@app.command("db:migrate")
def db_migrate(plain: bool = typer.Option(False, "--plain", help="Plain output")) -> None:
    """Prepare the DuckDB schema."""
    settings = get_settings()
    conn = None
    try:
        with writer_lock(settings.database_path):
            conn = get_connection(settings.database_path)
            apply_all_migrations(conn)
        typer.echo(f"[db:migrate] Applied core schema to {settings.database_path}")
    except Exception as exc:  # pragma: no cover - defensive logging
        typer.echo(f"[db:migrate] Error: {exc}")
        raise typer.Exit(code=1)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if plain:
        return


@app.command("trident:prime")
@app.command("backfill:all")
def backfill_all_cmd(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to backfill/pull (e.g., BTC-USD)"),
    interval: str = typer.Option("1h", "--interval", help="Candle interval for crypto backfill"),
    lookback: int = typer.Option(2000, "--lookback", help="Bars to backfill for crypto"),
    fed_lookback: int = typer.Option(336, "--fed-lookback", help="Lookback days for Fed pull"),
    calendar_from: Optional[str] = typer.Option(
        None,
        "--calendar-from",
        help="Calendar start date YYYY-MM-DD (default: current UTC date)",
    ),
    calendar_to: Optional[str] = typer.Option(
        None, "--calendar-to", help="Calendar end date YYYY-MM-DD (default: +1 month from --calendar-from)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug on downstream pull commands"),
    strict: bool = typer.Option(False, "--strict", help="Stop on first failed step"),
) -> None:
    """Run all data collection commands for one symbol in sequence."""
    started = datetime.now(timezone.utc)
    calendar_from_eff = calendar_from or started.date().isoformat()
    calendar_to_eff = calendar_to or _plus_one_month_iso(calendar_from_eff)
    steps: List[dict] = []

    def _run_step(step_name: str, fn) -> None:
        print(f"[backfill:all] step={step_name} status=running")
        try:
            fn()
            steps.append({"step": step_name, "ok": True, "error": None})
            print(f"[backfill:all] step={step_name} status=ok")
        except Exception as exc:  # pragma: no cover - defensive execution wrapper
            err = str(exc)
            steps.append({"step": step_name, "ok": False, "error": err})
            print(f"[backfill:all] step={step_name} status=error error={err}")
            if strict:
                raise

    print(
        f"[backfill:all] symbol={symbol} interval={interval} lookback={lookback} "
        f"calendar_from={calendar_from_eff} calendar_to={calendar_to_eff}"
    )

    _run_step("db:migrate", lambda: db_migrate(plain=True))
    _run_step(
        "crypto:backfill",
        lambda: crypto_backfill(symbol=symbol, interval=interval, lookback=lookback, plain=True),
    )
    _run_step(
        "fundamentals:pull",
        lambda: fundamentals_pull(symbols=symbol, plain=True),
    )
    _run_step(
        "news:pull",
        lambda: news_pull(symbols=symbol, plain=True, debug=debug),
    )
    _run_step(
        "fed:pull",
        lambda: fed_pull(lookback=fed_lookback, plain=True),
    )
    _run_step(
        "sentiment:pull",
        lambda: sentiment_pull(plain=True),
    )
    _run_step(
        "calendar:pull",
        lambda: calendar_pull(
            from_date=calendar_from_eff,
            to_date=calendar_to_eff,
            debug=debug,
            plain=True,
        ),
    )

    finished = datetime.now(timezone.utc)
    summary = {
        "command": "backfill:all",
        "symbol": symbol,
        "interval": interval,
        "lookback": lookback,
        "calendar": {"from": calendar_from_eff, "to": calendar_to_eff},
        "started_at": started.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "finished_at": finished.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ok": all(bool(s.get("ok")) for s in steps),
        "steps": steps,
    }
    print(json.dumps(summary, indent=2))


@app.command("crypto:backfill")
def crypto_backfill(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to backfill (e.g., BTC-USD)"),
    interval: str = typer.Option(
        ...,
        "--interval",
        "-i",
        help="Interval (e.g., 1h). Examples: 15m, 5m, 1d",
    ),
    lookback: int = typer.Option(..., "--lookback", "-l", help="Number of bars to backfill"),
    plain: bool = typer.Option(False, "--plain", help="Plain output"),
) -> None:
    """Backfill crypto OHLCV data from the configured vendor."""
    try:
        from backend.services.crypto_backfill import backfill_candles

        settings = get_settings()
        with writer_lock(settings.database_path):
            count = backfill_candles(symbol=symbol, interval=interval, lookback=lookback)
        typer.echo(
            f"[crypto:backfill] symbol={symbol} interval={interval} lookback={lookback} -> wrote {count} bars"
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        typer.echo(f"[crypto:backfill] Error: {exc}")
    if plain:
        return


@app.command("news:pull")
def news_pull(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated symbols, e.g. BTC-USD,ETH-USD"),
    plain: bool = typer.Option(False, "--plain", help="Plain output mode"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Pull news for one or more comma-separated symbols."""
    from backend.services.news_api import (
        fetch_news_for_symbol,
        fetch_macro_news,
        fetch_sector_news,
        ACTIVE_SYMBOLS,
    )
    from backend.services.news_ingest import ingest_news

    sym_list = [s.strip() for s in symbols.split(",") if s.strip() and s.strip() in ACTIVE_SYMBOLS]
    settings = get_settings()
    run_id = _new_run_id("news")
    con = None
    with writer_lock(settings.database_path):
        try:
            from datetime import datetime, timezone, timedelta

            con = get_connection(settings.database_path)
            apply_all_migrations(con, skip_data_backfills=True)
            now_utc = datetime.now(timezone.utc)
            cutoff_24h = now_utc - timedelta(hours=24)
            print(f"[news:pull] run_id={run_id} symbols={len(sym_list)}")

            def _to_utc(ts):
                if ts is None:
                    return None
                if getattr(ts, "tzinfo", None) is None:
                    return ts.replace(tzinfo=timezone.utc)
                return ts.astimezone(timezone.utc)

            def _lane_age_and_count(lane: str) -> tuple[float | None, int]:
                latest_ts = con.execute(
                    "SELECT MAX(published_at) FROM news_items WHERE lane = ?",
                    [lane],
                ).fetchone()[0]
                count_24h = int(
                    con.execute(
                        "SELECT COUNT(*) FROM news_items WHERE lane = ? AND published_at >= ?",
                        [lane, cutoff_24h],
                    ).fetchone()[0]
                    or 0
                )
                latest = _to_utc(latest_ts)
                if latest is None:
                    return None, count_24h
                age_h = (now_utc - latest).total_seconds() / 3600.0
                return round(age_h, 2), count_24h

            macro_age_h, macro_count_24h = _lane_age_and_count("macro")
            sector_age_h, sector_count_24h = _lane_age_and_count("sector")
            macro_stale = macro_age_h is None or macro_age_h > 18.0 or macro_count_24h < 8
            sector_stale = sector_age_h is None or sector_age_h > 24.0 or sector_count_24h < 4
            print(
                "[news:pull] lane telemetry pre "
                f"macro_age_h={macro_age_h} macro_count_24h={macro_count_24h} macro_stale={macro_stale} "
                f"sector_age_h={sector_age_h} sector_count_24h={sector_count_24h} sector_stale={sector_stale}"
            )

            overflow_good_count = 25 if con.execute(
                "SELECT 1 FROM news_pull_log WHERE DATE(pulled_at_utc) = ? LIMIT 1",
                [now_utc.date()],
            ).fetchone() else 0

            sector_stats: dict = {}
            sector_ingest_stats: dict = {}
            sector_articles = fetch_sector_news(
                settings.newsapi_ai_key,
                window_hours=120 if sector_stale else 72,
                overflow_good_count=overflow_good_count,
                quality_floor=2,
                extra_pages=1 if (overflow_good_count or sector_stale) else 0,
                stale_mode=sector_stale,
                debug=debug,
                rotation_key=run_id,
                stats=sector_stats,
            )
            sector_wrote = ingest_news(con, None, sector_articles, lane="sector", stats=sector_ingest_stats) if sector_articles else 0
            print(f"[news:pull] sector wrote={sector_wrote}")

            macro_stats: dict = {}
            macro_ingest_stats: dict = {}
            macro_articles = fetch_macro_news(
                settings.newsapi_ai_key,
                window_hours=120 if macro_stale else 72,
                overflow_good_count=overflow_good_count,
                quality_floor=2,
                extra_pages=2 if macro_stale else (1 if overflow_good_count else 0),
                stale_mode=macro_stale,
                debug=debug,
                rotation_key=run_id,
                stats=macro_stats,
            )
            macro_wrote = ingest_news(con, None, macro_articles, lane="macro", stats=macro_ingest_stats) if macro_articles else 0
            print(f"[news:pull] macro wrote={macro_wrote}")

            for sym in sym_list:
                articles = fetch_news_for_symbol(settings.newsapi_ai_key, sym, debug=debug)
                wrote = ingest_news(con, sym, articles, lane="symbol")
                print(f"[news:pull] symbol={sym} wrote={wrote}")

            con.execute("INSERT INTO news_pull_log (pulled_at_utc) VALUES (?)", [now_utc])
            print(
                "[news:pull] retries "
                f"sector_raw={sector_stats.get('raw_total', sector_stats.get('raw', 0))} "
                f"macro_raw={macro_stats.get('raw_total', macro_stats.get('raw', 0))}"
            )
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass


@app.command("sentiment:pull")
def sentiment_pull(plain: bool = typer.Option(False, "--plain", help="Plain output mode")) -> None:
    """Fetch and cache market-wide sentiment index (Fear & Greed)."""
    from backend.services.fear_greed import fetch_alternative_me_fng

    settings = get_settings()
    con = None
    with writer_lock(settings.database_path):
        con = get_connection(settings.database_path)
        try:
            apply_all_migrations(con, skip_data_backfills=True)

            row = fetch_alternative_me_fng()
            exists = con.execute(
                """
                SELECT 1
                FROM fear_greed_index
                WHERE ts_utc = ? AND source = ?
                """,
                [row["ts_utc"], row["source"]],
            ).fetchone()
            if exists:
                status = "skipped"
            else:
                con.execute(
                    """
                    INSERT INTO fear_greed_index (ts_utc, value, label, source, fetched_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [row["ts_utc"], row["value"], row["label"], row["source"], row["fetched_at_utc"]],
                )
                status = "inserted"

            print(
                "[sentiment:pull] "
                f"source={row['source']} ts={row['ts_utc'].isoformat()} "
                f"value={row['value']} label={row['label']} "
                f"fetched_at={row['fetched_at_utc'].isoformat()} ({status})"
            )
            if plain:
                return
        finally:
            try:
                con.close()
            except Exception:
                pass


@app.command("fed:pull")
def fed_pull(
    lookback: int = typer.Option(730, "--lookback", help="Lookback days"),
    plain: bool = typer.Option(False, "--plain", help="Plain output mode"),
) -> None:
    """Fetch and cache Fed liquidity series from FRED."""
    from datetime import datetime, timezone, timedelta
    from backend.services.fred import (
        fred_fetch_observations,
        upsert_fred_series,
        get_latest_fred_value,
    )

    settings = get_settings()
    if not settings.fred_api_key:
        print("[fed:pull] error: FRED_API_KEY is missing")
        raise typer.Exit(code=2)

    con = None
    with writer_lock(settings.database_path):
        con = get_connection(settings.database_path)
        try:
            apply_all_migrations(con, skip_data_backfills=True)

            start_date = (datetime.now(timezone.utc) - timedelta(days=lookback)).date()
            series_ids = ["WALCL", "WRESBAL", "RRPONTSYD"]

            def _format_value(value: float | None) -> str:
                if value is None:
                    return "n/a"
                abs_val = abs(value)
                if abs_val >= 1e12:
                    return f"{value / 1e12:.2f}T"
                if abs_val >= 1e9:
                    return f"{value / 1e9:.2f}B"
                if abs_val >= 1e6:
                    return f"{value / 1e6:.2f}M"
                return f"{value:.2f}"

            failures = 0
            for series_id in series_ids:
                try:
                    rows = fred_fetch_observations(series_id, settings.fred_api_key, start_date)
                    inserted, updated, skipped = upsert_fred_series(
                        con, series_id, rows, datetime.now(timezone.utc)
                    )
                    latest = get_latest_fred_value(con, series_id)
                    latest_date = latest.get("obs_date") if latest else None
                    latest_val = latest.get("value_norm") if latest else None
                    latest_date_txt = latest_date.isoformat() if latest_date else "n/a"
                    latest_val_txt = _format_value(latest_val)
                    print(
                        f"[fed:pull] series={series_id} inserted={inserted} updated={updated} "
                        f"skipped={skipped} latest={latest_date_txt} value={latest_val_txt}"
                    )
                except Exception as exc:
                    failures += 1
                    print(f"[fed:pull] warning: series={series_id} error={exc}")

            if failures == len(series_ids):
                raise typer.Exit(code=1)
            if plain:
                return
        finally:
            try:
                con.close()
            except Exception:
                pass


@app.command("fed:features")
def fed_features(plain: bool = typer.Option(False, "--plain", help="Plain output")) -> None:
    """Show cached Fed liquidity features."""
    from backend.features.fed_liquidity import compute_fed_liquidity_features_v2

    settings = get_settings()
    con = get_connection(settings.database_path)
    apply_all_migrations(con, skip_data_backfills=True)
    features = compute_fed_liquidity_features_v2(con)
    print(json.dumps(features, indent=2, default=str))
    if plain:
        return


@app.command("fundamentals:pull")
def fundamentals_pull(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated symbols"),
    plain: bool = typer.Option(False, "--plain", help="Plain output"),
) -> None:
    """Pull fundamentals for given symbols from CoinGecko and ingest into DuckDB."""
    from backend.services.coingecko_fundamentals import fetch_fundamentals_for_symbol
    from backend.services.fundamentals_ingest import ingest_fundamentals

    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    settings = get_settings()
    con = None
    with writer_lock(settings.database_path):
        con = get_connection(settings.database_path)
        try:
            apply_all_migrations(con, skip_data_backfills=True)
            print(f"[fundamentals:pull] Fetching fundamentals for {len(sym_list)} symbols")

            for sym in sym_list:
                try:
                    fundamentals = fetch_fundamentals_for_symbol(sym)
                    wrote = ingest_fundamentals(con, sym, fundamentals)
                    if not fundamentals:
                        print(f"[fundamentals:pull] symbol={sym} -> no data written (empty response)")
                    else:
                        print(f"[fundamentals:pull] symbol={sym} -> wrote {wrote} rows")
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"[fundamentals:pull] symbol={sym} -> error: {exc}")
        finally:
            try:
                con.close()
            except Exception:
                pass


@app.command("calendar:pull")
def calendar_pull(
    from_date: Optional[str] = typer.Option(
        None, "--from", help="Start date (YYYY-MM-DD); default: today - 7 days"
    ),
    to_date: Optional[str] = typer.Option(
        None, "--to", help="End date (YYYY-MM-DD); default: today + 14 days"
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    plain: bool = typer.Option(False, "--plain", help="Plain output mode"),
) -> None:
    """Pull and cache economic calendar events from Finnhub."""
    from datetime import datetime, timezone, timedelta

    from backend.services.economic_calendar import (
        fetch_finnhub_calendar,
        classify_category,
        macro_tags_for_event,
        classify_impact,
        get_upcoming_calendar_events,
        normalize_event_family,
        ensure_calendar_explainers,
        get_static_event_explainer,
    )

    settings = get_settings()
    if not settings.finnhub_api_key:
        print("[calendar:pull] error: FINNHUB_API_KEY is missing")
        raise SystemExit(2)

    now_utc = datetime.now(timezone.utc)
    if from_date is None:
        from_date = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = (now_utc + timedelta(days=14)).strftime("%Y-%m-%d")

    lock_ctx = writer_lock(settings.database_path)
    lock_ctx.__enter__()
    con = get_connection(settings.database_path)
    apply_all_migrations(con, skip_data_backfills=True)

    try:
        events = fetch_finnhub_calendar(settings.finnhub_api_key, from_date, to_date)
    except Exception as exc:
        print(f"[calendar:pull] error: {exc}")
        try:
            con.close()
        except Exception:
            pass
        try:
            lock_ctx.__exit__(None, None, None)
        except Exception:
            pass
        raise SystemExit(1)

    if debug and events:
        ts_list = [ev.get("event_ts_utc") for ev in events if ev.get("event_ts_utc")]
        if ts_list:
            min_ts = min(ts_list)
            max_ts = max(ts_list)
            try:
                to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                to_dt = None
            beyond_to = sum(1 for t in ts_list if to_dt and t > to_dt)
            print(
                "[calendar:pull][debug] api_range min_ts="
                f"{min_ts.isoformat()} max_ts={max_ts.isoformat()} to_date={to_date} "
                f"beyond_to={beyond_to}",
                file=sys.stderr,
            )

    if debug:
        sample = None
        for ev in events:
            title = (ev.get("title") or "").lower()
            country = (ev.get("country") or "").upper()
            if country == "US" and any(k in title for k in ["cpi", "pce", "inflation"]):
                sample = ev
                break
        if sample is None and events:
            sample = events[0]
        if sample:
            raw = sample.get("raw") or {}
            print("[calendar:pull][debug] sample keys:", list(raw.keys()), file=sys.stderr)
            print(
                "[calendar:pull][debug] values forecast="
                f"{raw.get('forecast')} estimate={raw.get('estimate')} consensus={raw.get('consensus')} "
                f"expected={raw.get('expected')} previous={raw.get('previous')} prev={raw.get('prev')} "
                f"prior={raw.get('prior')} actual={raw.get('actual')} value={raw.get('value')}",
                file=sys.stderr,
            )

    allowed_countries = settings.calendar_countries or ["US", "JP"]
    allowed_set = {c.upper() for c in allowed_countries}
    fetched_total = len(events)
    filtered_events = [
        event for event in events if (event.get("country") or "").upper() in allowed_set
    ]
    kept = len(filtered_events)
    filtered_out = fetched_total - kept

    inserted = 0
    updated = 0
    skipped = 0
    archived = 0
    purged = 0
    explainer_candidates = []
    for event in filtered_events:
        title = event.get("title") or ""
        country = event.get("country")
        category = classify_category(title)
        impact = classify_impact(country, title, category)
        family = normalize_event_family(title)
        country_key = (country or "").upper()
        static_explainer = get_static_event_explainer(title, category)
        if static_explainer is None:
            if category in {"monetary_policy", "inflation", "jobs", "liquidity", "money_supply", "growth"}:
                if impact in {"high", "medium"}:
                    explainer_candidates.append(
                        (family, country_key, category, title, event.get("event_ts_utc"), impact)
                    )
        macro_tags = macro_tags_for_event(
            title,
            category,
            impact=impact,
            event_ts_utc=event.get("event_ts_utc"),
            now_utc=now_utc,
        )
        raw_json = json.dumps(event.get("raw", {}), sort_keys=True, separators=(",", ":"))
        fetched_at = now_utc

        existing = con.execute(
            """
            SELECT actual, forecast, previous, category, impact, unit, macro_tags, raw_json
            FROM economic_calendar_events
            WHERE provider = ? AND event_ts_utc = ? AND country IS NOT DISTINCT FROM ? AND title = ?
            """,
            ["finnhub", event["event_ts_utc"], country, title],
        ).fetchone()
        if existing:
            (
                old_actual,
                old_forecast,
                old_previous,
                old_category,
                old_impact,
                old_unit,
                old_macro_tags,
                old_raw_json,
            ) = existing

            def _float_equal(a, b, tol=1e-9) -> bool:
                if a is None and b is None:
                    return True
                if a is None or b is None:
                    return False
                return abs(float(a) - float(b)) <= tol

            def _canon_json(raw_val: str) -> str:
                try:
                    return json.dumps(json.loads(raw_val), sort_keys=True, separators=(",", ":"))
                except Exception:
                    return (raw_val or "").strip()

            new_raw_json = raw_json
            old_raw_canon = _canon_json(old_raw_json or "")

            new_tags = json.dumps(macro_tags)
            try:
                old_tags_list = json.loads(old_macro_tags) if old_macro_tags else []
            except Exception:
                old_tags_list = []
            changed = False
            if event.get("actual") is not None and not _float_equal(event.get("actual"), old_actual):
                changed = True
            if event.get("forecast") is not None and not _float_equal(event.get("forecast"), old_forecast):
                changed = True
            if event.get("previous") is not None and not _float_equal(event.get("previous"), old_previous):
                changed = True
            if event.get("unit") is not None and (event.get("unit") or "") != (old_unit or ""):
                changed = True
            if category != (old_category or ""):
                changed = True
            if impact != (old_impact or ""):
                changed = True
            if sorted(macro_tags) != sorted(old_tags_list):
                changed = True
            if new_raw_json != old_raw_canon:
                changed = True

            if changed:
                con.execute(
                    """
                    UPDATE economic_calendar_events
                    SET actual = CASE WHEN ? IS NULL THEN actual ELSE ? END,
                        forecast = CASE WHEN ? IS NULL THEN forecast ELSE ? END,
                        previous = CASE WHEN ? IS NULL THEN previous ELSE ? END,
                        unit = CASE WHEN ? IS NULL THEN unit ELSE ? END,
                        category = ?,
                        impact = ?,
                        macro_tags = ?,
                        raw_json = ?,
                        fetched_at_utc = ?
                    WHERE provider = ? AND event_ts_utc = ? AND country IS NOT DISTINCT FROM ? AND title = ?
                    """,
                    [
                        event.get("actual"),
                        event.get("actual"),
                        event.get("forecast"),
                        event.get("forecast"),
                        event.get("previous"),
                        event.get("previous"),
                        event.get("unit"),
                        event.get("unit"),
                        category,
                        impact,
                        new_tags,
                        new_raw_json,
                        fetched_at,
                        "finnhub",
                        event["event_ts_utc"],
                        country,
                        title,
                    ],
                )
                updated += 1
            else:
                skipped += 1
            continue

        con.execute(
            """
            INSERT INTO economic_calendar_events (
                provider, event_ts_utc, country, title, category, impact, macro_tags,
                actual, forecast, previous, unit, raw_json, fetched_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "finnhub",
                event["event_ts_utc"],
                country,
                title,
                category,
                impact,
                json.dumps(macro_tags),
                event.get("actual"),
                event.get("forecast"),
                event.get("previous"),
                event.get("unit"),
                raw_json,
                fetched_at,
            ],
        )
        inserted += 1

    # Archive older events to keep the active table lean, while preserving deep history.
    archive_cutoff = now_utc - timedelta(days=1095)
    archive_before = con.execute(
        "SELECT COUNT(*) FROM economic_calendar_events_archive"
    ).fetchone()[0]
    purged = con.execute(
        "SELECT COUNT(*) FROM economic_calendar_events WHERE event_ts_utc < ?",
        [archive_cutoff],
    ).fetchone()[0]
    if purged:
        con.execute(
            """
            INSERT INTO economic_calendar_events_archive (
                provider, event_ts_utc, country, title, category, impact, macro_tags,
                actual, forecast, previous, unit, raw_json, fetched_at_utc, archived_at_utc
            )
            SELECT
                provider, event_ts_utc, country, title, category, impact, macro_tags,
                actual, forecast, previous, unit, raw_json, fetched_at_utc, ?
            FROM economic_calendar_events e
            WHERE event_ts_utc < ?
              AND NOT EXISTS (
                  SELECT 1 FROM economic_calendar_events_archive a
                  WHERE a.provider = e.provider
                    AND a.event_ts_utc = e.event_ts_utc
                    AND a.country IS NOT DISTINCT FROM e.country
                    AND a.title = e.title
              )
            """,
            [now_utc, archive_cutoff],
        )
        con.execute(
            "DELETE FROM economic_calendar_events WHERE event_ts_utc < ?",
            [archive_cutoff],
        )
    archive_after = con.execute(
        "SELECT COUNT(*) FROM economic_calendar_events_archive"
    ).fetchone()[0]
    archived = archive_after - archive_before

    explainers_added = 0
    if settings.openai_api_key and explainer_candidates:
        max_explainers = 50
        impact_rank = {"high": 2, "medium": 1, "low": 0}
        explainer_candidates.sort(
            key=lambda item: (
                -impact_rank.get(item[5] or "low", 0),
                abs(((item[4] or now_utc) - now_utc).total_seconds()),
                item[1],
                item[3],
            )
        )
        limited = explainer_candidates[:max_explainers]
        explainer_map = {}
        for family, country_key, category, title, _, _ in limited:
            key = (family, country_key, category)
            if key not in explainer_map:
                explainer_map[key] = title
        if debug:
            print(
                f"[calendar:pull][debug] explainer_candidates={len(explainer_candidates)} "
                f"capped={len(explainer_map)}",
                file=sys.stderr,
            )
        explainers_added = ensure_calendar_explainers(
            con,
            explainer_map,
            settings.openai_api_key,
            now_utc=now_utc,
            debug=debug,
        )
    elif debug:
        if not settings.openai_api_key:
            print("[calendar:pull][debug] explainers skipped: OPENAI_API_KEY missing", file=sys.stderr)
        elif not explainer_candidates:
            print("[calendar:pull][debug] explainers skipped: no candidates", file=sys.stderr)

    if debug and explainers_added:
        print(f"[calendar:pull][debug] explainers_added={explainers_added}", file=sys.stderr)

    countries_str = ",".join(allowed_countries)
    print(
        "[calendar:pull] countries="
        f"{countries_str} fetched={fetched_total} kept={kept} "
        f"filtered_out={filtered_out} inserted={inserted} updated={updated} skipped={skipped} "
        f"archived={archived} purged={purged}"
    )
    if kept == 0:
        print(
            f"[calendar:pull] WARNING: no events matched countries={countries_str} for the requested window"
        )
    if not plain:
        upcoming = get_upcoming_calendar_events(
            con,
            now_utc,
            hours_ahead=168,
            min_impact=None,
            limit=50,
        )
        impact_rank = {"high": 0, "medium": 1, "low": 2}
        upcoming.sort(
            key=lambda ev: (
                impact_rank.get(ev.get("impact") or "low", 3),
                ev.get("event_ts_utc") or now_utc,
            )
        )
        preferred = [ev for ev in upcoming if ev.get("impact") in {"high", "medium"}]
        preview = (preferred or upcoming)[:3]
        for ev in preview:
            ts = ev.get("event_ts_utc")
            ts_str = ts.isoformat() if ts else "n/a"
            print(f"[calendar:pull] upcoming {ts_str} {ev.get('country')} {ev.get('impact')} {ev.get('title')}")
    if plain:
        try:
            con.close()
        except Exception:
            pass
        try:
            lock_ctx.__exit__(None, None, None)
        except Exception:
            pass
        return
    try:
        con.close()
    except Exception:
        pass
    try:
        lock_ctx.__exit__(None, None, None)
    except Exception:
        pass

@app.command("calendar:features")
def calendar_features_cmd(
    lookback: int = typer.Option(168, "--lookback", help="Lookback window (hours)"),
    lookahead: int = typer.Option(168, "--lookahead", help="Lookahead window (hours)"),
    min_impact: str = typer.Option("medium", "--min-impact", help="Minimum impact: high|medium|low"),
    max_upcoming: int = typer.Option(15, "--max-upcoming", help="Max upcoming items"),
    max_recent: int = typer.Option(5, "--max-recent", help="Max recent items"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    plain: bool = typer.Option(False, "--plain", help="Plain JSON output"),
) -> None:
    """Compute calendar features from cached economic events."""
    from datetime import datetime, timezone

    from backend.features.calendar_features import compute_calendar_features

    settings = get_settings()
    con = get_connection(settings.database_path)
    apply_all_migrations(con)

    now_utc = datetime.now(timezone.utc)
    result = compute_calendar_features(
        con,
        now_utc,
        lookback_hours=lookback,
        lookahead_hours=lookahead,
        min_impact=min_impact,
        max_upcoming_items=max_upcoming,
        max_recent_items=max_recent,
        debug=debug,
    )
    if plain:
        print(json.dumps(result))
        return
    print(json.dumps(result, indent=2))

@app.command("ai:decide")
def ai_decide(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to generate a decision for"),
    plain: bool = typer.Option(False, "--plain", help="Plain output"),
) -> None:
    """Run the AI decision layer (placeholder)."""
    settings = get_settings()
    typer.echo(
        f"[ai:decide] symbol={symbol} using ai_vendor={settings.ai_vendor} model={settings.ai_model} data_vendor={settings.data_vendor}"
    )
    typer.echo(f"[ai:decide] News vendor: {settings.news_vendor} | On-chain vendor: {settings.onchain_vendor}")
    # TODO: implement decision engine, scenario builder, and renderer.
    if plain:
        return


@app.command("decision:trident")
def decision_trident_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD"),
    interval: Optional[str] = typer.Option(None, "--interval", help="Optional interval, e.g. 1h"),
    asof: Optional[str] = typer.Option(None, "--asof", help="Optional as-of timestamp (currently informational)"),
    as_json: bool = typer.Option(True, "--json/--no-json", help="Print JSON output (default: true)"),
    verbose: bool = typer.Option(False, "--verbose", help="Emit debug lines to stderr"),
    use_gpt: Optional[str] = typer.Option(
        None,
        "--use_gpt",
        help="Enable GPT narrative fields (true/false). If unset, uses TRIDENT_USE_GPT env.",
    ),
) -> None:
    """Phase 5A deterministic decision engine (regime + deadband)."""
    settings = get_settings()
    with writer_lock(settings.database_path):
        con = get_connection(settings.database_path)
        try:
            apply_all_migrations(con)

            if use_gpt is None:
                # If TRIDENT_USE_GPT is unset, default to enabled when an API key exists.
                env_toggle = os.getenv("TRIDENT_USE_GPT")
                if env_toggle is None:
                    use_gpt_effective = bool(settings.openai_api_key)
                else:
                    use_gpt_effective = settings.trident_use_gpt
            else:
                use_gpt_effective = str(use_gpt).strip().lower() in {"1", "true", "yes", "y", "on"}
            gpt_model = settings.trident_gpt_model or "gpt-5.2"

            if use_gpt_effective and not settings.openai_api_key:
                print("[decision:trident][warn] OPENAI_API_KEY missing, narrative disabled", file=sys.stderr)
                use_gpt_effective = False
            if use_gpt_effective:
                print(f">>> CALLING OPENAI {gpt_model}")

            result = run_decision_trident(
                symbol=symbol,
                con=con,
                interval=interval,
                use_gpt=use_gpt_effective,
                gpt_model=gpt_model,
            )

            if asof:
                result["asof_requested"] = asof

            if verbose:
                gate = result.get("no_trade_gate", {}) or {}
                d = result.get("decision", {}) or {}
                print(
                    f"[decision:trident][debug] symbol={symbol} action={d.get('action')} gate_active={gate.get('active')}",
                    file=sys.stderr,
                )
                for reason in gate.get("reasons") or []:
                    print(
                        f"[decision:trident][debug] gate {reason.get('code')}: {reason.get('detail')}",
                        file=sys.stderr,
                    )

            required = (
                (((result.get("strategy_inputs") or {}).get("data_quality") or {}).get("required_modalities"))
                or {}
            )
            stale_failed = not bool(required.get("ok", True))
            if stale_failed:
                fail_payload = _required_modality_failure_payload(
                    command="decision:trident",
                    symbol=symbol,
                    gate={"required_modality_status": required, "failed": True},
                    snapshot={
                        "run_id": result.get("run_id"),
                        "snapshot_id": result.get("snapshot_id"),
                        "input_hash": result.get("input_hash"),
                    },
                )
                result["required_modality_failure"] = fail_payload.get("error")

            # Default output is JSON to stdout.
            if as_json:
                print(json.dumps(result, indent=2, sort_keys=False))
            else:
                print(json.dumps(result, indent=2, sort_keys=False))

            if stale_failed:
                raise typer.Exit(code=1)
        finally:
            try:
                con.close()
            except Exception:
                pass


@app.command("paper:init")
def paper_init_cmd(
    equity: Optional[float] = typer.Option(None, "--equity", help="Starting equity"),
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    fee_bps: Optional[float] = typer.Option(None, "--fee-bps", help="Fee bps"),
    slippage_bps: Optional[float] = typer.Option(None, "--slippage-bps", help="Slippage bps"),
    max_trades: Optional[int] = typer.Option(None, "--max-trades", help="Max trades per run"),
    max_open: Optional[int] = typer.Option(None, "--max-open", help="Max open positions"),
    reset: bool = typer.Option(False, "--reset", help="Reset paper tables before init"),
) -> None:
    from backend.db.paper_repo import reset_paper_ledger, upsert_paper_config

    settings = get_settings()
    with writer_lock(settings.database_path):
        con = get_connection(settings.database_path)
        apply_all_migrations(con)

        cleared_tables: List[str] = []
        if reset:
            cleared_tables = reset_paper_ledger(con)

        cfg = _paper_default_config(settings)
        if equity is not None:
            cfg["starting_equity"] = float(equity)
        if symbols:
            cfg["symbols"] = _split_symbols(symbols)
        if fee_bps is not None:
            cfg["fee_bps"] = float(fee_bps)
        if slippage_bps is not None:
            cfg["slippage_bps"] = float(slippage_bps)
        if max_trades is not None:
            cfg["max_trades_per_run"] = int(max_trades)
        if max_open is not None:
            cfg["max_open_positions"] = int(max_open)

        active = upsert_paper_config(con, cfg, set_active=True)
        print("[paper:init] schema: ready")
        print(f"[paper:init] active_config={active.get('config_id')}")
        print(
            "[paper:init] symbols="
            f"{','.join(active.get('symbols') or [])} equity={active.get('starting_equity')} "
            f"fee_bps={active.get('fee_bps')} slippage_bps={active.get('slippage_bps')}"
        )
        print(
            json.dumps(
                {
                    "command": "paper:init",
                    "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "config_id": active.get("config_id"),
                    "active": bool(active.get("active")),
                    "starting_equity": active.get("starting_equity"),
                    "symbols": active.get("symbols"),
                    "fee_bps": active.get("fee_bps"),
                    "slippage_bps": active.get("slippage_bps"),
                    "max_trades_per_run": active.get("max_trades_per_run"),
                    "max_open_positions": active.get("max_open_positions"),
                    "reset_applied": bool(reset),
                    "tables_cleared": cleared_tables,
                },
                indent=2,
                sort_keys=False,
                default=str,
            )
        )
        try:
            con.close()
        except Exception:
            pass


@app.command("paper:reset")
def paper_reset_cmd(
    yes: bool = typer.Option(False, "--yes", help="Confirm paper ledger reset"),
) -> None:
    from backend.db.paper_repo import reset_paper_ledger

    if not yes:
        print("[paper:reset] aborted: pass --yes to confirm")
        raise typer.Exit(code=2)
    settings = get_settings()
    with writer_lock(settings.database_path):
        con = get_connection(settings.database_path)
        apply_all_migrations(con)
        tables = reset_paper_ledger(con)
        print(f"[paper:reset] truncated tables: {','.join(tables)}")
        print(
            json.dumps(
                {
                    "command": "paper:reset",
                    "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    "ok": True,
                    "tables_cleared": tables,
                },
                indent=2,
                sort_keys=False,
            )
        )
        try:
            con.close()
        except Exception:
            pass


@app.command("paper:status")
def paper_status_cmd() -> None:
    from backend.services.paper_reporting import build_paper_status

    settings = get_settings()
    con = get_connection(settings.database_path)
    apply_all_migrations(con)
    status = build_paper_status(con)
    if not status.get("has_active_config"):
        print("[paper:status] no active paper config")
        print(json.dumps({"command": "paper:status", **status}, indent=2, sort_keys=False, default=str))
        return
    eq = (status.get("equity") or {}).get("equity")
    open_positions = len(status.get("open_positions") or [])
    last_run = status.get("last_run") or {}
    print(
        f"[paper:status] active_config={status.get('config_id')} "
        f"open_positions={open_positions} last_run={last_run.get('status')}"
    )
    print(json.dumps({"command": "paper:status", **status}, indent=2, sort_keys=False, default=str))


@app.command("paper:mark")
def paper_mark_cmd(
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Optional comma-separated symbols"),
    exit_on_flip: bool = typer.Option(False, "--exit-on-flip", help="Enable decision-flip exits"),
) -> None:
    from backend.db.paper_repo import get_active_paper_config, list_open_positions
    from backend.services.paper_engine import (
        fallback_aggression_profile,
        gate_override_allowed,
        get_latest_mid_price,
        mark_open_positions,
        replay_open_positions,
        resolve_aggression_profile,
    )
    from backend.services.policy_ai import render_aggression_profile_gpt52

    settings = get_settings()
    lock_ctx = writer_lock(settings.database_path)
    lock_ctx.__enter__()
    con = get_connection(settings.database_path)
    apply_all_migrations(con)
    config = get_active_paper_config(con)
    if not config:
        print("[paper:mark] no active paper config; run paper:init first")
        try:
            con.close()
        except Exception:
            pass
        try:
            lock_ctx.__exit__(None, None, None)
        except Exception:
            pass
        raise typer.Exit(code=2)

    symbol_filter = _split_symbols(symbols) if symbols else []
    symbols_refresh = symbol_filter or list(config.get("symbols") or settings.paper_symbols or [])
    risk_limits_cfg = dict(config.get("risk_limits") or {})
    replay_interval = str(
        risk_limits_cfg.get("replay_interval") or getattr(settings, "paper_replay_interval", "15m") or "15m"
    )
    replay_lookback_bars = int(
        risk_limits_cfg.get("replay_lookback_bars") or getattr(settings, "paper_replay_lookback_bars", 672) or 672
    )
    if symbol_filter:
        positions: List[dict] = []
        for sym in symbol_filter:
            positions.extend(list_open_positions(con, symbol=sym))
    else:
        positions = list_open_positions(con)
    learning_policy = config.get("learning_policy") or {}
    aggression_baseline = dict(learning_policy.get("aggression_baseline") or {})
    gpt_model = getattr(settings, "trident_gpt_model", None) or "gpt-5.2"
    gpt_enabled = bool(getattr(settings, "openai_api_key", None))
    gpt_announced = False

    def _announce_gpt_once() -> None:
        nonlocal gpt_announced
        if gpt_enabled and not gpt_announced:
            print(f">>> CALLING OPENAI {gpt_model}")
            gpt_announced = True

    def _position_details(pos_rows: List[dict]) -> List[dict]:
        details: List[dict] = []
        for pos in pos_rows:
            symbol = str(pos.get("symbol") or "")
            side = str(pos.get("side") or "").upper()
            qty = float(pos.get("qty") or 0.0)
            entry_price = float(pos.get("entry_price") or 0.0)
            mid_price, _ = get_latest_mid_price(con, symbol=symbol, interval=replay_interval)
            if mid_price is None and replay_interval != "1h":
                mid_price, _ = get_latest_mid_price(con, symbol=symbol, interval="1h")
            unrealized_pnl = None
            if mid_price is not None and qty > 0 and entry_price > 0:
                if side == "LONG":
                    unrealized_pnl = (float(mid_price) - entry_price) * qty
                elif side == "SHORT":
                    unrealized_pnl = (entry_price - float(mid_price)) * qty
            details.append(
                {
                    "position_id": pos.get("position_id"),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "entry_ts": pos.get("entry_ts"),
                    "entry_price": entry_price,
                    "stop_price": pos.get("stop_price"),
                    "take_profit_price": pos.get("take_profit_price"),
                    "time_stop_ts": pos.get("time_stop_ts"),
                    "status": pos.get("status"),
                    "linked_run_id": pos.get("linked_run_id"),
                    "mid_price": round(float(mid_price), 8) if mid_price is not None else None,
                    "unrealized_pnl_usd": round(float(unrealized_pnl), 8) if unrealized_pnl is not None else None,
                }
            )
        return details

    position_details_before = _position_details(positions)

    candle_refresh_errors: List[str] = []
    for sym in symbols_refresh:
        try:
            crypto_backfill(symbol=sym, interval=replay_interval, lookback=replay_lookback_bars, plain=True)
        except BaseException as exc:  # pragma: no cover - defensive refresh isolation
            candle_refresh_errors.append(f"{sym}:{exc}")
            print(f"[paper:mark][warn] candle refresh failed for {sym}: {exc}")
    candle_refresh = {
        "refreshed_symbols": symbols_refresh,
        "ok_count": max(0, len(symbols_refresh) - len(candle_refresh_errors)),
        "error_count": len(candle_refresh_errors),
        "errors": candle_refresh_errors,
    }

    gate_overrides = (learning_policy.get("gate_overrides") or {})
    replay = replay_open_positions(
        conn=con,
        positions=positions,
        config=config,
        command="paper:mark",
        run_id=None,
        interval=replay_interval,
    )
    replay_exits = list(replay.get("exits_triggered") or [])
    if symbol_filter:
        positions_for_mark: List[dict] = []
        for sym in symbol_filter:
            positions_for_mark.extend(list_open_positions(con, symbol=sym))
    else:
        positions_for_mark = list_open_positions(con)
    state_map = {str(p.get("symbol")): str(p.get("side") or "FLAT") for p in positions_for_mark}
    aggression_by_symbol: Dict[str, dict] = {}

    def _flip_getter(sym: str) -> Optional[dict]:
        result = run_decision_trident(
            symbol=sym,
            con=con,
            interval="1h",
            use_gpt=False,
            current_position_state_override=state_map.get(sym, "FLAT"),
            gate_overrides=gate_overrides,
        )
        aggression_profile = fallback_aggression_profile(
            baseline=aggression_baseline,
            rationale="deterministic fallback due missing GPT aggression call",
        )
        if gpt_enabled:
            _announce_gpt_once()
            gpt_aggr = render_aggression_profile_gpt52(
                {
                    "symbol": sym,
                    "decision": result.get("decision"),
                    "signal_quality": ((result.get("strategy_inputs") or {}).get("signal_quality") or {}),
                    "scenario_snapshot": ((result.get("strategy_inputs") or {}).get("scenario_snapshot") or {}),
                    "no_trade_gate": result.get("no_trade_gate"),
                    "paper_trade_preview": result.get("paper_trade_preview"),
                },
                model=gpt_model,
            )
            aggression_profile = resolve_aggression_profile(
                decision_json=result,
                gpt_profile=gpt_aggr,
                baseline=aggression_baseline,
            )
        override_allowed, strong_pass = gate_override_allowed(result, aggression_profile)
        aggression_profile["override_weighted_gate"] = bool(override_allowed)
        aggression_profile["strong_move_structure_pass"] = bool(strong_pass)
        aggression_by_symbol[sym] = {
            "tier": aggression_profile.get("tier"),
            "source": aggression_profile.get("source"),
            "override_weighted_gate": bool(override_allowed),
            "strong_move_structure_pass": bool(strong_pass),
            "knobs_applied": aggression_profile.get("knobs_applied") or {},
            "rationale": aggression_profile.get("rationale"),
            "quick_exit_bias": aggression_profile.get("quick_exit_bias"),
        }
        sq = ((result.get("strategy_inputs") or {}).get("signal_quality") or {})
        gate = result.get("no_trade_gate") or {}
        scen = ((result.get("strategy_inputs") or {}).get("scenario_snapshot") or {})
        return {
            "action": str((result.get("paper_trade_preview") or {}).get("hypothetical_action") or ""),
            "confidence": (result.get("decision") or {}).get("confidence"),
            "agreement_score": sq.get("agreement_score"),
            "effective_confidence": sq.get("effective_confidence"),
            "margin_vs_second": scen.get("margin_vs_second"),
            "gate_active": bool(gate.get("active")),
            "hard_blockers": gate.get("hard_blockers") or [],
            "penalty_ratio": gate.get("penalty_ratio"),
            "override_weighted_gate": bool(override_allowed),
            "quick_exit_bias": aggression_profile.get("quick_exit_bias"),
        }

    marked = mark_open_positions(
        conn=con,
        positions=positions_for_mark,
        config=config,
        exit_on_flip=exit_on_flip,
        allow_quick_exit=True,
        decision_flip_getter=_flip_getter,
    )
    if symbol_filter:
        positions_after: List[dict] = []
        for sym in symbol_filter:
            positions_after.extend(list_open_positions(con, symbol=sym))
    else:
        positions_after = list_open_positions(con)
    position_details_after = _position_details(positions_after)
    mark_exits = [
        {
            **e,
            "exit_origin": "mark_live_check",
        }
        for e in (marked.get("exits_triggered") or [])
    ]
    exits_combined = replay_exits + mark_exits
    equity = marked.get("equity") or {}
    selected_aggression = fallback_aggression_profile(
        baseline=aggression_baseline,
        rationale="no open positions to evaluate",
    )
    if aggression_by_symbol:
        first_key = sorted(aggression_by_symbol.keys())[0]
        selected_aggression = dict(aggression_by_symbol.get(first_key) or selected_aggression)
    print(f"[paper:mark] open_positions={len(positions)} exits={len(exits_combined)}")
    print(
        f"[paper:mark] equity={equity.get('equity')} drawdown_pct={equity.get('drawdown_pct')} "
        f"marks_written={marked.get('marks_written')} candle_refresh_ok={candle_refresh['ok_count']}"
    )
    print(
        f"[paper:mark] aggression={selected_aggression.get('tier')} "
        f"source={selected_aggression.get('source')} "
        f"override={str(selected_aggression.get('override_weighted_gate')).lower()}"
    )
    print(
        json.dumps(
            {
                "command": "paper:mark",
                "version": SYSTEM_PHASE_VERSION,
                "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "open_positions": len(positions),
                "position_details_before": position_details_before,
                "open_positions_after": len(positions_after),
                "position_details_after": position_details_after,
                "candle_refresh": candle_refresh,
                "replay": {
                    "interval": replay.get("interval"),
                    "window_from": replay.get("window_from"),
                    "window_to": replay.get("window_to"),
                    "positions_checked": replay.get("positions_checked"),
                    "exits_replayed": replay.get("exits_replayed"),
                    "events": replay.get("events") or [],
                },
                "aggression": {
                    "tier": selected_aggression.get("tier"),
                    "source": selected_aggression.get("source"),
                    "override_weighted_gate": bool(selected_aggression.get("override_weighted_gate")),
                    "strong_move_structure_pass": bool(selected_aggression.get("strong_move_structure_pass")),
                    "knobs_applied": selected_aggression.get("knobs_applied") or {},
                    "rationale": selected_aggression.get("rationale"),
                    "by_symbol": aggression_by_symbol,
                },
                "exits_triggered": exits_combined,
                "marks_written": marked.get("marks_written"),
                "equity": equity,
            },
            indent=2,
            sort_keys=False,
            default=str,
        )
    )
    try:
        con.close()
    except Exception:
        pass
    try:
        lock_ctx.__exit__(None, None, None)
    except Exception:
        pass


@app.command("paper:run")
def paper_run_cmd(
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Optional comma-separated symbols"),
    max_trades: Optional[int] = typer.Option(None, "--max-trades", help="Max trades this run"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not place simulated entry"),
    refresh: bool = typer.Option(False, "--refresh/--smart-refresh", help="Force refresh or use smart refresh"),
    mark_first: bool = typer.Option(False, "--mark-first", help="Mark open positions before running"),
    verbose: bool = typer.Option(False, "--verbose", help="Emit verbose per-symbol payload fields"),
) -> None:
    from backend.db.paper_repo import (
        compute_equity_snapshot,
        create_run,
        get_active_paper_config,
        insert_position_and_entry_fill,
        list_open_positions,
        record_candidate,
        record_decision,
        record_error_run,
        record_signal_audit,
        set_run_status,
        upsert_paper_config,
    )
    from backend.services.paper_engine import (
        apply_intelligence,
        build_pattern_snapshot,
        build_prediction_snapshot,
        compute_position_plan,
        evaluate_entry_eligibility,
        extract_candidate_from_decision,
        fallback_aggression_profile,
        get_latest_mid_price,
        mark_open_positions,
        replay_open_positions,
        resolve_intel_stage,
        resolve_aggression_profile,
        simulate_fill,
        smart_refresh_symbol,
    )
    from backend.services.policy_ai import (
        render_aggression_profile_gpt52,
        render_paper_run_explanation_gpt52,
    )

    settings = get_settings()
    gpt_model = settings.trident_gpt_model or "gpt-5.2"
    gpt_enabled = bool(settings.openai_api_key)
    lock_ctx = writer_lock(settings.database_path)
    lock_ctx.__enter__()
    con = get_connection(settings.database_path)
    apply_all_migrations(con, skip_data_backfills=True)
    config = get_active_paper_config(con)
    if not config:
        config = upsert_paper_config(con, _paper_default_config(settings), set_active=True)

    symbols_eff = _split_symbols(symbols) if symbols else list(config.get("symbols") or settings.paper_symbols)
    if not symbols_eff:
        print("[paper:run] no symbols configured")
        try:
            con.close()
        except Exception:
            pass
        try:
            lock_ctx.__exit__(None, None, None)
        except Exception:
            pass
        raise typer.Exit(code=2)
    max_trades_eff = int(max_trades if max_trades is not None else (config.get("max_trades_per_run") or 1))
    replay_interval_eff = str(
        ((config.get("risk_limits") or {}).get("replay_interval"))
        or getattr(settings, "paper_replay_interval", "15m")
        or "15m"
    )
    run_id = ""

    try:
        run_id = create_run(
            con,
            command="paper:run",
            flags_dict={
                "symbols_requested": symbols_eff,
                "refresh_mode": "refresh" if refresh else "smart_refresh",
                "dry_run": bool(dry_run),
            },
            config_id=str(config.get("config_id") or ""),
        )
        current_run_ts = _paper_run_ts(con, run_id) or datetime.now(timezone.utc)
        print(
            f"[paper:run] run_id={run_id} status=running mode={'refresh' if refresh else 'smart_refresh'} "
            f"dry_run={str(dry_run).lower()}"
        )
        gpt_announced = False

        def _announce_gpt_once() -> None:
            nonlocal gpt_announced
            if gpt_enabled and not gpt_announced:
                print(f">>> CALLING OPENAI {gpt_model}")
                gpt_announced = True

        def _compact_candidate(candidate: Optional[dict]) -> Optional[dict]:
            if candidate is None:
                return None
            return {
                "symbol": candidate.get("symbol"),
                "side": candidate.get("side"),
                "confidence": candidate.get("confidence"),
                "effective_confidence": candidate.get("effective_confidence"),
                "agreement_score": candidate.get("agreement_score"),
                "freshness_score": candidate.get("freshness_score"),
                "candidate_score": round(float(candidate.get("candidate_score") or 0.0), 6),
                "gate_active": bool(candidate.get("gate_active")),
                "gates_blocking": list(candidate.get("gates_blocking") or []),
                "quality_flags": list(candidate.get("quality_flags") or []),
                "override_weighted_gate": bool(candidate.get("override_weighted_gate")),
                "strong_move_structure_pass": bool(candidate.get("strong_move_structure_pass")),
                "entry_eligible": bool(candidate.get("entry_eligible")),
                "entry_blockers": list(candidate.get("entry_blockers") or []),
                "entry_mode": str(candidate.get("entry_mode") or "standard"),
                "entry_mode_score": round(float(candidate.get("entry_mode_score") or candidate.get("selection_score") or 0.0), 6),
                "rescue_penalty_codes": list(candidate.get("rescue_penalty_codes") or []),
                "rescue_selected": bool(candidate.get("rescue_selected")),
                "prediction": candidate.get("prediction") or {},
                "patterns": candidate.get("patterns") or {},
                "intel_score_delta": float(candidate.get("intel_score_delta") or 0.0),
                "intel_blockers": list(candidate.get("intel_blockers") or []),
                "intel_used_for_entry": bool(candidate.get("intel_used_for_entry")),
                "regime": candidate.get("regime"),
                "event_risk": candidate.get("event_risk"),
            }

        def _tier_rank(tier: str) -> int:
            order = {
                "very_defensive": 0,
                "defensive": 1,
                "balanced": 2,
                "assertive": 3,
                "very_aggressive": 4,
            }
            return order.get(str(tier or "balanced"), 2)

        def _rescue_criteria(candidate: dict, decision: dict, cfg: dict) -> tuple[bool, List[str], float]:
            score = float(candidate.get("selection_score") or candidate.get("candidate_score") or 0.0)
            effective_conf = float(candidate.get("effective_confidence") or 0.0)
            agreement = float(candidate.get("agreement_score") or 0.0)
            decision_inputs = decision.get("strategy_inputs") or {}
            margin = float(((decision_inputs.get("scenario_snapshot") or {}).get("margin_vs_second") or 0.0))
            side = str(candidate.get("side") or "NO_TRADE")
            hard_blockers = list((decision.get("no_trade_gate") or {}).get("hard_blockers") or [])
            gate_active = bool((decision.get("no_trade_gate") or {}).get("active"))
            override_gate = bool(candidate.get("override_weighted_gate"))

            penalties: List[str] = []
            if side not in {"LONG", "SHORT"}:
                return False, ["side_not_directional"], score
            if gate_active and not override_gate:
                return False, ["gate_active"], score
            if "CRITICAL_LOW_CONFIDENCE" in hard_blockers:
                return False, ["critical_low_confidence"], score
            if "CRITICAL_RISK_CLUSTER" in hard_blockers:
                return False, ["critical_risk_cluster"], score

            min_score = float(cfg.get("entry_rescue_min_score") or 0.0)
            min_effective_confidence = float(cfg.get("entry_rescue_min_effective_confidence") or 0.0)
            min_agreement = float(cfg.get("entry_rescue_min_agreement") or 0.0)
            min_margin = float(cfg.get("entry_rescue_min_margin") or 0.0)
            if score < min_score:
                return False, ["entry_rescue_min_score"], score
            if effective_conf < min_effective_confidence:
                return False, ["entry_rescue_min_effective_confidence"], score
            if agreement < min_agreement:
                return False, ["entry_rescue_min_agreement"], score
            if margin < min_margin:
                return False, ["entry_rescue_min_margin"], score

            quality = set(str(flag or "") for flag in (candidate.get("quality_flags") or []))
            if {"MODEL_EDGE_WEAK", "LOW_BREADTH"}.issubset(quality):
                score = max(0.0, score - 0.05)
                penalties.append("QUALITY_PENALTY_MODEL_EDGE_WEAK_LOW_BREADTH")

            if "LOW_CONFIDENCE" in quality:
                penalties.append("LOW_CONFIDENCE")
            if "LOW_HORIZON_ALIGNMENT" in quality:
                penalties.append("LOW_HORIZON_ALIGNMENT")
            if "LOW_AGREEMENT" in quality:
                penalties.append("LOW_AGREEMENT")

            return True, penalties, score

        closed_positions_this_run: List[dict] = []
        if mark_first:
            mark_first_out = mark_open_positions(
                conn=con,
                positions=list_open_positions(con),
                config=config,
                exit_on_flip=False,
                allow_quick_exit=False,
            )
            closed_positions_this_run = [
                {
                    **e,
                    "exit_origin": "mark_live_check",
                }
                for e in (mark_first_out.get("exits_triggered") or [])
            ]

        refresh_notes: List[str] = []
        refresh_status_by_step: dict = {}
        refresh_retry_counts: dict = {}
        news_min_interval_minutes = max(
            0,
            int(getattr(settings, "paper_news_min_interval_minutes", 60) or 60),
        )
        news_max_pulls_per_day = max(
            1,
            int(getattr(settings, "paper_news_max_pulls_per_day", 10) or 10),
        )

        def _refresh_crypto(sym: str, interval: str) -> None:
            crypto_backfill(symbol=sym, interval=interval, lookback=2000, plain=True)

        stale_refresh_needed = False
        for sym in symbols_eff:
            notes = smart_refresh_symbol(
                conn=con,
                symbol=sym,
                interval="1h",
                force_refresh=refresh,
                refresh_callback=_refresh_crypto,
            )
            if notes:
                stale_refresh_needed = True
            refresh_notes.extend([f"{sym}:{n}" for n in notes])
        if replay_interval_eff != "1h":
            for sym in symbols_eff:
                replay_notes = smart_refresh_symbol(
                    conn=con,
                    symbol=sym,
                    interval=replay_interval_eff,
                    force_refresh=refresh,
                    refresh_callback=_refresh_crypto,
                    max_age_hours=2.0,
                )
                if replay_notes:
                    stale_refresh_needed = True
                refresh_notes.extend([f"{sym}:{replay_interval_eff}:{n}" for n in replay_notes])

        def _to_utc(ts):
            if ts is None:
                return None
            if getattr(ts, "tzinfo", None) is None:
                local_tz = datetime.now().astimezone().tzinfo or timezone.utc
                return ts.replace(tzinfo=local_tz).astimezone(timezone.utc)
            return ts.astimezone(timezone.utc)

        def _news_refresh_gate() -> tuple[bool, str]:
            now_utc = datetime.now(timezone.utc)
            pull_rows = con.execute(
                "SELECT pulled_at_utc FROM news_pull_log"
            ).fetchall()
            pulls_utc = [_to_utc(row[0]) for row in pull_rows if row and row[0] is not None]
            pulls_today = sum(1 for ts in pulls_utc if ts is not None and ts.date() == now_utc.date())
            if pulls_today >= news_max_pulls_per_day:
                return False, f"daily_cap_reached:{pulls_today}/{news_max_pulls_per_day}"
            last_pull_utc = max(pulls_utc) if pulls_utc else None
            if last_pull_utc is None:
                return True, "ok:first_pull"
            age_minutes = (now_utc - last_pull_utc).total_seconds() / 60.0
            if age_minutes < float(news_min_interval_minutes):
                return False, (
                    f"min_interval_guard:age_min={round(age_minutes, 2)}"
                    f"<{news_min_interval_minutes}"
                )
            return True, "ok:cooldown_met"

        if refresh or stale_refresh_needed:
            news_allowed, news_gate_reason = _news_refresh_gate()

            # Release coordinator connection before nested pull commands open their own
            # writer connections; this reduces DuckDB write-write contention.
            detach_for_refresh = True
            probe_con = None
            try:
                probe_con = get_connection(settings.database_path)
                if probe_con is con:
                    detach_for_refresh = False
            except Exception:
                detach_for_refresh = False
            finally:
                if probe_con is not None and probe_con is not con:
                    try:
                        probe_con.close()
                    except Exception:
                        pass

            if detach_for_refresh:
                try:
                    con.close()
                except Exception:
                    pass
                con = None
            else:
                refresh_notes.append("refresh:single_connection_mode")

            def _safe_refresh_step(name: str, fn, quick_fail_on_conflict: bool = False) -> None:
                retry_delays = [0.5, 1.0, 2.0]
                retryable_signals = ("write-write conflict", "failed to commit")
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        fn()
                        refresh_status_by_step[name] = "ok"
                        refresh_retry_counts[name] = attempts - 1
                        refresh_notes.append(f"{name}:ok")
                        return
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:  # pragma: no cover - defensive refresh isolation
                        msg = str(exc)
                        retryable = any(sig in msg.lower() for sig in retryable_signals)
                        if retryable and quick_fail_on_conflict:
                            refresh_status_by_step[name] = "error"
                            refresh_retry_counts[name] = attempts - 1
                            refresh_notes.append(f"{name}:error:conflict_quick_fail:{msg}")
                            print(
                                f"[paper:run][warn] refresh step {name} conflict quick-fail; "
                                f"switching to resume mode: {msg}"
                            )
                            return
                        if retryable and attempts <= len(retry_delays):
                            delay = retry_delays[attempts - 1]
                            print(
                                f"[paper:run][warn] refresh step {name} conflict attempt={attempts} "
                                f"retrying_in_s={delay}: {msg}"
                            )
                            time.sleep(delay)
                            continue
                        refresh_status_by_step[name] = "error"
                        refresh_retry_counts[name] = attempts - 1
                        refresh_notes.append(f"{name}:error:{msg}")
                        print(f"[paper:run][warn] refresh step {name} failed: {msg}")
                        return

            _safe_refresh_step("fundamentals:pull", lambda: fundamentals_pull(symbols=",".join(symbols_eff), plain=True))
            if news_allowed:
                _safe_refresh_step("news:pull", lambda: news_pull(symbols=",".join(symbols_eff), plain=True, debug=False))
                if refresh_status_by_step.get("news:pull") == "error":
                    refresh_notes.append("news:pull:resume_mode=per_symbol")
                    for sym in symbols_eff:
                        _safe_refresh_step(
                            f"news:pull:{sym}",
                            lambda sym=sym: news_pull(symbols=sym, plain=True, debug=False),
                        )
            else:
                refresh_status_by_step["news:pull"] = "skipped"
                refresh_retry_counts["news:pull"] = 0
                refresh_notes.append(f"news:pull:skipped:{news_gate_reason}")
                print(f"[paper:run] news:pull skipped ({news_gate_reason})")
            _safe_refresh_step("fed:pull", lambda: fed_pull(lookback=336, plain=True))
            _safe_refresh_step("sentiment:pull", lambda: sentiment_pull(plain=True))
            cal_from = datetime.now(timezone.utc).date().isoformat()
            cal_to = _plus_one_month_iso(cal_from)
            _safe_refresh_step(
                "calendar:pull",
                lambda: calendar_pull(from_date=cal_from, to_date=cal_to, debug=False, plain=True),
            )
            # Reopen after refresh commands so downstream reads (decision + GPT context)
            # see the latest committed state from nested pull connections.
            if detach_for_refresh:
                con = get_connection(settings.database_path)
                apply_all_migrations(con, skip_data_backfills=True)
            else:
                apply_all_migrations(con, skip_data_backfills=True)
            config_latest = get_active_paper_config(con)
            if config_latest:
                config = config_latest

        replay_interval = str(((config.get("risk_limits") or {}).get("replay_interval")) or replay_interval_eff or "15m")
        open_positions_pre_replay = list_open_positions(con)
        replay = replay_open_positions(
            conn=con,
            positions=open_positions_pre_replay,
            config=config,
            command="paper:run",
            run_id=run_id,
            interval=replay_interval,
        )
        replay_exits = list(replay.get("exits_triggered") or [])
        if replay_exits:
            closed_positions_this_run.extend(replay_exits)

        eq_before = compute_equity_snapshot(con, float(config.get("starting_equity") or 10000.0))
        risk_cfg = dict(config.get("risk_limits") or {})
        learning_policy = config.get("learning_policy") or {}
        gate_overrides = learning_policy.get("gate_overrides") or {}
        penalty_chop = float(learning_policy.get("penalty_high_vol_chop", 0.9))
        penalty_event = float(learning_policy.get("penalty_elevated_event_risk", 0.9))
        aggression_baseline = dict(learning_policy.get("aggression_baseline") or {})
        prediction_enabled = bool(
            risk_cfg.get("prediction_enabled", getattr(settings, "paper_prediction_enabled", True))
        )
        pattern_enabled = bool(
            risk_cfg.get("pattern_enabled", getattr(settings, "paper_pattern_enabled", True))
        )
        intel_stage = resolve_intel_stage(
            closed_trades=int(eq_before.get("closed_positions") or 0),
            mode=str(risk_cfg.get("intelligence_mode") or getattr(settings, "paper_intel_mode", "auto")),
            bootstrap=int(risk_cfg.get("intelligence_bootstrap_trades") or getattr(settings, "paper_intel_bootstrap_trades", 25)),
            promotion=int(risk_cfg.get("intelligence_promotion_trades") or getattr(settings, "paper_intel_promotion_trades", 50)),
        )
        stage_label = str(intel_stage.get("stage") or "shadow")
        pred_weight = 0.0
        pattern_weight = 0.0
        if stage_label == "soft":
            pred_weight = float(risk_cfg.get("prediction_weight_soft") or getattr(settings, "paper_prediction_weight_soft", 0.15))
            pattern_weight = float(risk_cfg.get("pattern_weight_soft") or getattr(settings, "paper_pattern_weight_soft", 0.10))
        elif stage_label == "hard":
            pred_weight = float(risk_cfg.get("prediction_weight_hard") or getattr(settings, "paper_prediction_weight_hard", 0.20))
            pattern_weight = float(risk_cfg.get("pattern_weight_hard") or getattr(settings, "paper_pattern_weight_hard", 0.10))
        allow_weighted_gate_override_cfg = bool(
            risk_cfg.get("allow_weighted_gate_override", getattr(settings, "paper_allow_weighted_gate_override", False))
        )
        bootstrap_guard_active = bool(intel_stage.get("bootstrap_guard_active"))
        allow_weighted_gate_override_eff = bool(
            allow_weighted_gate_override_cfg and (not bootstrap_guard_active or stage_label == "hard")
        )
        quality_veto_mode = "bootstrap" if bootstrap_guard_active else "full"
        adaptive_controls_disabled: List[str] = []
        is_weekend_utc = _is_weekend_utc()
        rescue_cfg = {
            "enabled": bool(risk_cfg.get("entry_rescue_enabled", getattr(settings, "paper_entry_rescue_enabled", True))),
            "max_per_run": int(risk_cfg.get("entry_rescue_max_per_run", getattr(settings, "paper_entry_rescue_max_per_run", 1))),
            "min_score": float(risk_cfg.get("entry_rescue_min_score", getattr(settings, "paper_entry_rescue_min_score", 0.05))),
            "min_effective_confidence": float(
                risk_cfg.get("entry_rescue_min_effective_confidence", getattr(settings, "paper_entry_rescue_min_effective_confidence", 0.22))
            ),
            "min_agreement": float(risk_cfg.get("entry_rescue_min_agreement", getattr(settings, "paper_entry_rescue_min_agreement", 0.55))),
            "min_margin": float(risk_cfg.get("entry_rescue_min_margin", getattr(settings, "paper_entry_rescue_min_margin", 0.015))),
            "notional_cap": float(risk_cfg.get("entry_rescue_notional_cap", getattr(settings, "paper_entry_rescue_notional_cap", 0.06))),
            "risk_mult": float(risk_cfg.get("entry_rescue_risk_mult", getattr(settings, "paper_entry_rescue_risk_mult", 0.70))),
            "stop_mult": float(risk_cfg.get("entry_rescue_stop_mult", getattr(settings, "paper_entry_rescue_stop_mult", 1.30))),
            "hold_mult": float(risk_cfg.get("entry_rescue_hold_mult", getattr(settings, "paper_entry_rescue_hold_mult", 0.90))),
            "weekend_guard": bool(risk_cfg.get("weekend_rescue_guard", getattr(settings, "paper_weekend_rescue_guard", False))),
            "weekend_notional_cap": float(risk_cfg.get("weekend_rescue_notional_cap", getattr(settings, "paper_weekend_rescue_notional_cap", 0.03))),
        }
        rescue_allowed = bool(rescue_cfg.get("enabled")) and (
            stage_label == "shadow" or str(stage_label) == "hard"
        )
        weekend_rescue_guard_active = bool(
            rescue_cfg.get("weekend_guard")
            and is_weekend_utc
            and stage_label == "shadow"
        )
        if weekend_rescue_guard_active:
            rescue_allowed = False
        if not rescue_allowed:
            rescue_cfg["max_per_run"] = 0
        if bootstrap_guard_active:
            adaptive_controls_disabled = [
                "weighted_gate_override",
                "gpt_numeric_learning_influence",
                "aggression_baseline_learning_drift",
                "quality_veto_extended_combos",
            ]

        decisions_by_symbol: dict = {}
        aggression_by_symbol: dict = {}
        candidates: List[dict] = []
        eligible_pre_intelligence = 0
        for sym in symbols_eff:
            open_pos_sym = list_open_positions(con, symbol=sym)
            pos_state = str((open_pos_sym[0].get("side") if open_pos_sym else "FLAT") or "FLAT")
            decision = run_decision_trident(
                symbol=sym,
                con=con,
                interval="1h",
                use_gpt=False,
                current_position_state_override=pos_state,
                gate_overrides=gate_overrides,
            )
            aggression_profile = fallback_aggression_profile(
                baseline=aggression_baseline,
                rationale="deterministic fallback due missing GPT aggression call",
            )
            if gpt_enabled:
                _announce_gpt_once()
                gpt_aggr = render_aggression_profile_gpt52(
                    {
                        "symbol": sym,
                        "decision": decision.get("decision"),
                        "signal_quality": ((decision.get("strategy_inputs") or {}).get("signal_quality") or {}),
                        "scenario_snapshot": ((decision.get("strategy_inputs") or {}).get("scenario_snapshot") or {}),
                        "no_trade_gate": decision.get("no_trade_gate"),
                        "paper_trade_preview": decision.get("paper_trade_preview"),
                    },
                    model=gpt_model,
                )
                aggression_profile = resolve_aggression_profile(
                    decision_json=decision,
                    gpt_profile=gpt_aggr,
                    baseline=aggression_baseline,
                )
            record_decision(con, run_id=run_id, symbol=sym, decision_json=decision)
            candidate = extract_candidate_from_decision(
                decision,
                penalty_high_vol_chop=penalty_chop,
                penalty_elevated_event_risk=penalty_event,
                aggression_profile=aggression_profile,
                allow_weighted_gate_override=allow_weighted_gate_override_eff,
            )
            if candidate.get("side") in {"LONG", "SHORT"} and float(candidate.get("candidate_score") or 0.0) > 0.0:
                eligible_pre_intelligence += 1
            prediction = (
                build_prediction_snapshot(
                    conn=con,
                    symbol=sym,
                    decision_json=decision,
                    candidate=candidate,
                    interval=replay_interval,
                )
                if prediction_enabled
                else {}
            )
            patterns = (
                build_pattern_snapshot(
                    conn=con,
                    symbol=sym,
                    decision_json=decision,
                    interval=replay_interval,
                )
                if pattern_enabled
                else {}
            )
            intel_result = apply_intelligence(
                candidate=candidate,
                prediction=prediction,
                patterns=patterns,
                stage_cfg={
                    "stage": stage_label,
                    "prediction_weight": pred_weight if prediction_enabled else 0.0,
                    "pattern_weight": pattern_weight if pattern_enabled else 0.0,
                },
            )
            candidate["prediction"] = intel_result.get("prediction") or {}
            candidate["patterns"] = intel_result.get("patterns") or {}
            candidate["intel_score_delta"] = float(intel_result.get("intel_score_delta") or 0.0)
            candidate["intel_blockers"] = list(intel_result.get("intel_blockers") or [])
            candidate["intel_used_for_entry"] = bool(intel_result.get("intel_used_for_entry"))
            candidate["selection_score"] = round(
                float(candidate.get("candidate_score") or 0.0) + float(candidate.get("intel_score_delta") or 0.0),
                6,
            )
            candidate["entry_mode"] = "standard"
            candidate["entry_mode_score"] = float(candidate.get("selection_score") or 0.0)
            candidate["rescue_penalty_codes"] = []
            candidate["rescue_selected"] = False
            entry_ok, entry_blockers = evaluate_entry_eligibility(
                candidate=candidate,
                decision_json=decision,
                cfg=risk_cfg,
                intel_result=intel_result,
                quality_veto_mode=quality_veto_mode,
            )
            candidate["entry_eligible"] = bool(entry_ok)
            candidate["entry_blockers"] = list(entry_blockers)
            record_candidate(con, run_id=run_id, candidate_dict=candidate)
            record_signal_audit(
                con,
                {
                    "ts": datetime.now(timezone.utc),
                    "run_id": run_id,
                    "symbol": sym,
                    "mode": stage_label,
                    "used_for_entry": bool(candidate.get("intel_used_for_entry")),
                    "prediction": candidate.get("prediction") or {},
                    "patterns": candidate.get("patterns") or {},
                    "intel_score_delta": float(candidate.get("intel_score_delta") or 0.0),
                    "intel_blockers": list(candidate.get("intel_blockers") or []),
                },
            )
            decisions_by_symbol[sym] = decision
            aggression_by_symbol[sym] = {
                "tier": str(candidate.get("aggression_tier") or aggression_profile.get("tier") or "balanced"),
                "source": str(candidate.get("aggression_source") or aggression_profile.get("source") or "deterministic_fallback"),
                "override_weighted_gate": bool(candidate.get("override_weighted_gate")),
                "strong_move_structure_pass": bool(candidate.get("strong_move_structure_pass")),
                "knobs_applied": dict(candidate.get("aggression_knobs") or aggression_profile.get("knobs_applied") or {}),
                "quick_exit_bias": str(candidate.get("aggression_quick_exit_bias") or aggression_profile.get("quick_exit_bias") or "none"),
                "rationale": str(aggression_profile.get("rationale") or ""),
            }
            candidates.append(candidate)
            print(
                f"[paper:run] symbol={sym} action={(decision.get('decision') or {}).get('action')} "
                f"cand_score={candidate.get('candidate_score'):.3f} "
                f"gate_active={bool((decision.get('no_trade_gate') or {}).get('active'))} "
                f"aggr={candidate.get('aggression_tier')}"
            )

        eligible_prefilter = [
            c for c in candidates if c.get("side") in {"LONG", "SHORT"} and float(c.get("candidate_score") or 0.0) > 0
        ]
        eligible = [c for c in eligible_prefilter if bool(c.get("entry_eligible"))]
        eligible.sort(
            key=lambda c: (
                -float(c.get("selection_score") or c.get("candidate_score") or 0.0),
                -float(c.get("confidence") or 0.0),
                str(c.get("symbol")),
            )
        )
        selected = eligible[0] if eligible else None
        selection_status = "selected" if selected is not None else "none"
        rescue_lane_applicable = False
        rescue_candidates: List[dict] = []
        rescue_selected = None
        if selected is None and rescue_cfg.get("max_per_run", 0) > 0:
            rescue_lane_applicable = eligible_prefilter and rescue_allowed and bool(rescue_cfg.get("enabled"))
            if rescue_lane_applicable:
                for candidate in eligible_prefilter:
                    decision = decisions_by_symbol.get(str(candidate.get("symbol") or "")) or {}
                    rescue_ok, rescue_penalties, rescue_score = _rescue_criteria(
                        candidate=candidate,
                        decision=decision,
                        cfg=rescue_cfg,
                    )
                    candidate["entry_mode_score"] = float(rescue_score)
                    candidate["rescue_penalty_codes"] = list(rescue_penalties)
                    if rescue_ok:
                        rescue_candidates.append(candidate)
                if rescue_candidates:
                    rescue_candidates.sort(
                        key=lambda c: (
                            -float(c.get("entry_mode_score") or c.get("selection_score") or 0.0),
                            -float(c.get("confidence") or 0.0),
                            str(c.get("symbol")),
                        )
                    )
                    rescue_pick = rescue_candidates[0]
                    if rescue_cfg.get("max_per_run", 0) > 0:
                        rescue_pick["entry_mode"] = "rescue"
                        rescue_pick["entry_mode_score"] = float(
                            rescue_pick.get("entry_mode_score")
                            if rescue_pick.get("entry_mode_score") is not None
                            else rescue_pick.get("selection_score", 0.0)
                        )
                        rescue_pick["rescue_selected"] = True
                        selected = rescue_pick
                        selection_status = "rescue_selected"
                        rescue_selected = str(rescue_pick.get("symbol") or "")
        placed_trade = False
        trade_block_reasons: List[str] = []
        trade_payload: Optional[dict] = None

        open_positions_count = len(list_open_positions(con))
        if selected is not None:
            if max_trades_eff <= 0:
                trade_block_reasons.append("max_trades_per_run=0")
            if open_positions_count >= int(config.get("max_open_positions") or 1):
                trade_block_reasons.append("max_open_positions_reached")
            per_symbol_cap = int(
                ((config.get("risk_limits") or {}).get("max_open_positions_per_symbol"))
                or getattr(settings, "paper_max_open_positions_per_symbol", 1)
                or 1
            )
            open_positions_symbol = len(list_open_positions(con, symbol=str(selected.get("symbol") or "")))
            if open_positions_symbol >= per_symbol_cap:
                trade_block_reasons.append("symbol_position_cap_reached")
            if dry_run:
                trade_block_reasons.append("dry_run=true")
            if not trade_block_reasons:
                sym = str(selected.get("symbol") or "")
                side = str(selected.get("side") or "NO_TRADE")
                mid_price, _ = get_latest_mid_price(con, symbol=sym, interval="1h")
                if mid_price is None:
                    trade_block_reasons.append("missing_mid_price")
                else:
                    decision = decisions_by_symbol.get(sym) or {}
                    validity = int(((decision.get("decision") or {}).get("validity_hours")) or 8)
                    aggression_knobs = dict(selected.get("aggression_knobs") or {})
                    is_rescue = str(selected.get("entry_mode") or "standard").lower() == "rescue"
                    if is_rescue:
                        rescue_knobs = {
                            "risk_mult": min(
                                _safe_float(aggression_knobs.get("risk_mult"), 1.0),
                                _safe_float(rescue_cfg.get("risk_mult"), 0.70),
                            ),
                            "stop_mult": max(
                                _safe_float(aggression_knobs.get("stop_mult"), 1.0),
                                _safe_float(rescue_cfg.get("stop_mult"), 1.30),
                            ),
                            "hold_mult": min(
                                _safe_float(aggression_knobs.get("hold_mult"), 1.0),
                                _safe_float(rescue_cfg.get("hold_mult"), 0.90),
                            ),
                        }
                        rescue_notional_cap = _safe_float(rescue_cfg.get("notional_cap"), 0.06)
                        weekend_notional_cap = _safe_float(rescue_cfg.get("weekend_notional_cap"), 0.0)
                        if weekend_notional_cap > 0 and is_weekend_utc:
                            rescue_notional_cap = (
                                weekend_notional_cap
                                if rescue_notional_cap <= 0
                                else min(rescue_notional_cap, weekend_notional_cap)
                            )
                        base_notional_cap = _safe_float(aggression_knobs.get("notional_cap"), 0.0)
                        if rescue_notional_cap > 0:
                            rescue_knobs["notional_cap"] = (
                                rescue_notional_cap if base_notional_cap <= 0 else min(base_notional_cap, rescue_notional_cap)
                            )
                        else:
                            rescue_knobs["notional_cap"] = base_notional_cap
                        aggression_knobs = rescue_knobs
                        selected["aggression_knobs"] = aggression_knobs
                        if sym in aggression_by_symbol:
                            aggression_by_symbol[str(sym)]["knobs_applied"] = dict(aggression_knobs)
                    risk_cfg = dict(config.get("risk_limits") or {})
                    tech = compute_tech_features(sym, con, interval="1h") or {}
                    atr_pct = None
                    for key in ("atr_pct",):
                        if key in tech and tech.get(key) is not None:
                            atr_pct = float(tech.get(key))
                            break
                        for block in ("volatility",):
                            b = tech.get(block)
                            if isinstance(b, dict) and b.get(key) is not None:
                                atr_pct = float(b.get(key))
                                break
                    risk_cfg = dict(config.get("risk_limits") or {})
                    plan = compute_position_plan(
                        entry_price=float(mid_price),
                        side=side,
                        equity=float(eq_before.get("equity") or config.get("starting_equity") or 10000.0),
                        risk_cfg=risk_cfg,
                        atr_pct=atr_pct,
                        validity_hours=validity,
                        aggression_knobs=aggression_knobs,
                        hard_max_hold_hours=168,
                    )
                    qty = float(plan.get("qty") or 0.0)
                    if qty <= 0:
                        trade_block_reasons.append("qty_not_positive")
                    else:
                        fill = simulate_fill(
                            mid_price=float(mid_price),
                            side=side,
                            qty=qty,
                            fee_bps=float(config.get("fee_bps") or 5.0),
                            slippage_bps=float(config.get("slippage_bps") or 8.0),
                            fill_type="ENTRY",
                        )
                        id_suffix = uuid.uuid4().hex[:10]
                        position_id = (
                            f"pos_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
                            f"{sym.replace('-', '')}_{id_suffix}"
                        )
                        position = {
                            "position_id": position_id,
                            "symbol": sym,
                            "side": side,
                            "qty": qty,
                            "entry_ts": datetime.now(timezone.utc),
                            "entry_price": fill["fill_price"],
                            "stop_price": plan["stop_price"],
                            "take_profit_price": plan["take_profit_price"],
                            "time_stop_ts": plan["time_stop_ts"],
                            "status": "OPEN",
                            "linked_run_id": run_id,
                        }
                        fill_row = {
                            "fill_id": (
                                f"fill_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
                                f"{sym.replace('-', '')}_{uuid.uuid4().hex[:10]}"
                            ),
                            "ts": datetime.now(timezone.utc),
                            "fill_price": fill["fill_price"],
                            "fees_usd": fill["fees_usd"],
                            "slippage_usd": fill["slippage_usd"],
                            "qty": qty,
                            "type": "ENTRY",
                        }
                        insert_position_and_entry_fill(con, position, fill_row)
                        placed_trade = True
                        trade_payload = {
                            "position_id": position_id,
                            "symbol": sym,
                            "side": side,
                            "entry_mode": str(selected.get("entry_mode") or "standard"),
                            "entry_mode_score": round(
                                float(selected.get("entry_mode_score") or selected.get("selection_score") or 0.0), 6
                            ),
                            "rescue_selected": bool(selected.get("rescue_selected")),
                            "qty": round(qty, 6),
                            "fill_price": fill["fill_price"],
                            "stop_price": plan["stop_price"],
                            "take_profit_price": plan["take_profit_price"],
                            "time_stop_ts": plan["time_stop_ts"],
                            "validity_hours_applied": plan.get("validity_hours_applied"),
                            "fees_usd": fill["fees_usd"],
                            "slippage_usd": fill["slippage_usd"],
                            "aggression": aggression_by_symbol.get(sym, {}),
                        }
                        print(
                            f"[paper:run] placed ENTRY qty={trade_payload['qty']} fill={trade_payload['fill_price']} "
                            f"stop={trade_payload['stop_price']} tp={trade_payload['take_profit_price']} "
                            f"time_stop={trade_payload['time_stop_ts']} fees={trade_payload['fees_usd']} "
                            f"slip={trade_payload['slippage_usd']}"
                        )
        elif not trade_block_reasons:
            if eligible_prefilter:
                trade_block_reasons.append("no_eligible_candidates_after_entry_filters")
            else:
                trade_block_reasons.append("no_eligible_candidates")

        eq_after = compute_equity_snapshot(con, float(config.get("starting_equity") or 10000.0))
        open_positions_after = list_open_positions(con)
        selected_aggression = fallback_aggression_profile(
            baseline=aggression_baseline,
            rationale="no eligible candidate selected",
        )
        open_position_timeline: List[dict] = []
        if open_positions_after:
            run_ts_cache: dict[str, Optional[datetime]] = {}
            for pos in open_positions_after:
                linked_run_id = str(pos.get("linked_run_id") or "")
                if linked_run_id and linked_run_id not in run_ts_cache:
                    run_ts_cache[linked_run_id] = _paper_run_ts(con, linked_run_id)
                linked_run_ts = run_ts_cache.get(linked_run_id)
                if linked_run_ts is None:
                    linked_run_ts = _to_utc_datetime(pos.get("entry_ts"))
                open_position_timeline.append(
                    {
                        "position_id": pos.get("position_id"),
                        "symbol": str(pos.get("symbol") or ""),
                        "linked_run_id": linked_run_id or None,
                        "entry_ts": pos.get("entry_ts"),
                        "run_age_runs": (
                            _count_paper_run_steps_between(con, linked_run_ts, current_run_ts)
                            if linked_run_ts is not None and current_run_ts is not None
                            else None
                        ),
                    }
                )
        latest_position_timeline = open_position_timeline[-1] if open_position_timeline else None

        if selected is not None:
            selected_aggression = dict(aggression_by_symbol.get(str(selected.get("symbol") or ""), {})) or selected_aggression
        elif aggression_by_symbol:
            conservative_pick = sorted(
                aggression_by_symbol.items(),
                key=lambda kv: (_tier_rank(str((kv[1] or {}).get("tier") or "balanced")), str(kv[0])),
            )[0]
            picked_symbol = str(conservative_pick[0])
            selected_aggression = dict(conservative_pick[1] or {})
            selected_aggression["rationale"] = (
                f"no eligible candidate selected; aggregate conservative posture from {picked_symbol}"
            )
        selected_aggression["override_weighted_gate"] = bool(selected_aggression.get("override_weighted_gate"))
        selected_aggression["strong_move_structure_pass"] = bool(selected_aggression.get("strong_move_structure_pass"))
        print(
            f"[paper:run] aggression={selected_aggression.get('tier')} "
            f"source={selected_aggression.get('source')} "
            f"override={str(selected_aggression.get('override_weighted_gate')).lower()}"
        )
        run_explanation = {
            "source": "deterministic_fallback",
            "model": None,
            "summary": (
                f"paper:run evaluated {len(candidates)} candidates and "
                f"{'placed a simulated trade' if placed_trade else 'did not place a trade'}."
            ),
            "current_trade_status": (
                f"open_positions={len(open_positions_after)}; "
                + (
                    f"latest_position={open_positions_after[-1].get('symbol')} {open_positions_after[-1].get('side')} "
                    f"entry={open_positions_after[-1].get('entry_price')} stop={open_positions_after[-1].get('stop_price')} "
                    f"tp={open_positions_after[-1].get('take_profit_price')} "
                    f"runs_since_entry={latest_position_timeline.get('run_age_runs')} "
                    if open_positions_after
                    else "no open paper position is currently active."
                )
            ),
        }
        if gpt_enabled:
            _announce_gpt_once()
            g = render_paper_run_explanation_gpt52(
                {
                    "run_id": run_id,
                    "equity_before": eq_before,
                    "equity_after": eq_after,
                    "candidates": candidates[:10],
                    "selected": selected,
                    "placed_trade": bool(placed_trade),
                    "trade": trade_payload,
                    "reasons": trade_block_reasons,
                    "refresh": {
                        "status_by_step": refresh_status_by_step,
                        "retry_counts": refresh_retry_counts,
                        "notes": refresh_notes,
                    },
                    "replay": {
                        "interval": replay.get("interval"),
                        "window_from": replay.get("window_from"),
                        "window_to": replay.get("window_to"),
                        "positions_checked": replay.get("positions_checked"),
                        "exits_replayed": replay.get("exits_replayed"),
                        "events": replay.get("events") or [],
                    },
                    "aggression": {
                        "selected": selected_aggression,
                        "by_symbol": aggression_by_symbol,
                    },
                    "intelligence": {
                        "mode": str(intel_stage.get("mode") or "auto"),
                        "stage": stage_label,
                        "prediction_enabled": bool(prediction_enabled),
                        "pattern_enabled": bool(pattern_enabled),
                        "influence_weights": {
                            "prediction": round(float(pred_weight if prediction_enabled else 0.0), 6),
                            "pattern": round(float(pattern_weight if pattern_enabled else 0.0), 6),
                        },
                        "adaptive_controls_disabled": adaptive_controls_disabled,
                    },
                    "closed_positions_this_run": closed_positions_this_run,
                    "open_positions_after": open_positions_after[:5],
                },
                model=gpt_model,
            )
            if g:
                run_explanation = {"source": "gpt", "model": gpt_model, **g}
        blocker_counts: Counter = Counter()
        for candidate in candidates:
            for code in list(candidate.get("gates_blocking") or []):
                blocker_counts[str(code)] += 1
        top_blocker_codes = [{"code": code, "count": count} for code, count in blocker_counts.most_common(5)]
        blocked_candidates = sum(
            1
            for candidate in candidates
            if bool(candidate.get("gate_active"))
            or bool(candidate.get("gates_blocking"))
            or (candidate.get("side") in {"LONG", "SHORT"} and not bool(candidate.get("entry_eligible")))
            or str(candidate.get("side") or "NO_TRADE") == "NO_TRADE"
        )
        run_summary = {
            "selection_status": selection_status,
            "eligible_candidates": len(eligible_prefilter),
            "eligible_after_entry_filters": len(eligible),
            "vetoed_candidates": max(0, len(eligible_prefilter) - len(eligible)),
            "blocked_candidates": int(blocked_candidates),
            "replay_exits_count": int(replay.get("exits_replayed") or 0),
            "is_weekend_utc": bool(is_weekend_utc),
            "weekend_rescue_guard_active": bool(weekend_rescue_guard_active),
            "rescue_lane_applicable": bool(rescue_lane_applicable),
            "rescue_candidates": [str(c.get("symbol") or "") for c in rescue_candidates],
            "rescue_selected": rescue_selected,
            "top_blocker_codes": top_blocker_codes,
        }

        candidates_out = candidates if verbose else [_compact_candidate(c) for c in candidates]
        selected_out = selected if verbose else _compact_candidate(selected)

        def _attach_candidate_aggression(candidate_obj: Optional[dict]) -> Optional[dict]:
            if candidate_obj is None:
                return None
            item = dict(candidate_obj)
            sym = str(item.get("symbol") or "")
            src = dict(aggression_by_symbol.get(sym) or {})
            item["aggression"] = {
                "tier": src.get("tier") or item.get("aggression_tier"),
                "source": src.get("source") or item.get("aggression_source"),
                "override_weighted_gate": bool(
                    src.get("override_weighted_gate", item.get("override_weighted_gate"))
                ),
                "strong_move_structure_pass": bool(
                    src.get("strong_move_structure_pass", item.get("strong_move_structure_pass"))
                ),
                "knobs_applied": src.get("knobs_applied") or item.get("aggression_knobs") or {},
                "quick_exit_bias": src.get("quick_exit_bias") or item.get("aggression_quick_exit_bias"),
                "rationale": str(src.get("rationale") or ""),
            }
            if verbose:
                item["aggression"]["rationale_full"] = str(src.get("rationale") or "")
            item.pop("aggression_tier", None)
            item.pop("aggression_source", None)
            item.pop("aggression_knobs", None)
            item.pop("aggression_quick_exit_bias", None)
            return item

        candidates_out = [_attach_candidate_aggression(c) for c in candidates_out]
        selected_out = _attach_candidate_aggression(selected_out)

        set_run_status(
            con,
            run_id=run_id,
            status="ok",
            notes=";".join(refresh_notes + trade_block_reasons),
        )
        run_output = {
            "version": SYSTEM_PHASE_VERSION,
            "command": "paper:run",
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "equity_before": eq_before.get("equity"),
            "equity_after": eq_after.get("equity"),
            "replay": {
                "interval": replay.get("interval"),
                "window_from": replay.get("window_from"),
                "window_to": replay.get("window_to"),
                "positions_checked": replay.get("positions_checked"),
                "exits_replayed": replay.get("exits_replayed"),
                "events": replay.get("events") or [],
            },
            "run_summary": run_summary,
            "intelligence": {
                "mode": str(intel_stage.get("mode") or "auto"),
                "stage": stage_label,
                "prediction_enabled": bool(prediction_enabled),
                "pattern_enabled": bool(pattern_enabled),
                "influence_weights": {
                    "prediction": round(float(pred_weight if prediction_enabled else 0.0), 6),
                    "pattern": round(float(pattern_weight if pattern_enabled else 0.0), 6),
                },
                "adaptive_controls_disabled": adaptive_controls_disabled,
                "eligible_pre_intelligence": int(eligible_pre_intelligence),
                "eligible_post_intelligence": int(len(eligible)),
            },
            "candidates": candidates_out,
            "aggression": {
                "tier": selected_aggression.get("tier"),
                "source": selected_aggression.get("source"),
                "override_weighted_gate": bool(selected_aggression.get("override_weighted_gate")),
                "strong_move_structure_pass": bool(selected_aggression.get("strong_move_structure_pass")),
                "knobs_applied": selected_aggression.get("knobs_applied") or {},
                "rationale": selected_aggression.get("rationale"),
            },
            "selected": selected_out,
            "placed_trade": bool(placed_trade),
            "trade": trade_payload,
            "reasons": trade_block_reasons,
        }
        if verbose:
            run_output["candidates_verbose"] = candidates
            run_output["selected_verbose"] = selected
        if closed_positions_this_run:
            run_output["closed_positions_this_run"] = closed_positions_this_run
        run_output["refresh"] = {
            "status_by_step": refresh_status_by_step,
            "retry_counts": refresh_retry_counts,
            "notes": refresh_notes,
        }
        run_output["position_timeline"] = {
            "open_positions_count": len(open_positions_after),
            "open_positions": [
                {
                    **entry,
                    "run_age_runs": next(
                        (item.get("run_age_runs") for item in open_position_timeline if item.get("position_id") == entry.get("position_id")),
                        None,
                    ),
                }
                for entry in open_positions_after
            ],
            "latest_position_run_age_runs": latest_position_timeline.get("run_age_runs") if latest_position_timeline else None,
        }
        if isinstance(run_explanation, dict):
            run_explanation.pop("position_timeline", None)
            run_explanation.pop("position_timeline_v2", None)
        run_output["explanation"] = run_explanation
        print(json.dumps(run_output, indent=2, sort_keys=False, default=str))
    except Exception as exc:
        err_con = con
        if run_id and err_con is None:
            err_con = get_connection(settings.database_path)
            apply_all_migrations(err_con, skip_data_backfills=True)
        elif run_id:
            try:
                err_con.execute("SELECT 1")
            except Exception:
                err_con = get_connection(settings.database_path)
                apply_all_migrations(err_con, skip_data_backfills=True)
        if run_id and err_con is not None:
            record_error_run(err_con, run_id=run_id, error_text=str(exc))
        if err_con is not None and err_con is not con:
            try:
                err_con.close()
            except Exception:
                pass
        print(f"[paper:run] error: {exc}")
        raise
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        try:
            lock_ctx.__exit__(None, None, None)
        except Exception:
            pass


@app.command("paper:report")
def paper_report_cmd(
    daily: bool = typer.Option(False, "--daily", help="Daily report window"),
    weekly: bool = typer.Option(False, "--weekly", help="Weekly report window"),
    last: int = typer.Option(20, "--last", help="Last N closed trades"),
    by_symbol: Optional[str] = typer.Option(None, "--by-symbol", help="Filter by one symbol"),
    use_gpt: Optional[bool] = typer.Option(
        None,
        "--use-gpt/--no-use-gpt",
        help="Use GPT narrative. If unset, uses TRIDENT_USE_GPT env.",
    ),
) -> None:
    from backend.db.paper_repo import get_active_paper_config, list_signal_audit
    from backend.services.paper_reporting import build_paper_report
    from backend.services.policy_ai import render_paper_report_summary_gpt52

    settings = get_settings()
    con = get_connection(settings.database_path)
    apply_all_migrations(con)
    config = get_active_paper_config(con)
    if not config:
        print("[paper:report] no active paper config; run paper:init first")
        raise typer.Exit(code=2)

    report = build_paper_report(con, config, daily=daily, weekly=weekly, last=last, by_symbol=by_symbol)
    signal_audit_rows = list_signal_audit(con, last_n=200, symbol=by_symbol)
    pattern_counts: Dict[str, int] = {}
    pred_conf_sum = 0.0
    for row in signal_audit_rows:
        pred_conf_sum += float((row.get("prediction") or {}).get("confidence") or 0.0)
        for pid in (row.get("patterns") or {}).get("hits") or []:
            key = str(pid)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
    prediction_summary = {
        "rows": len(signal_audit_rows),
        "avg_confidence": round(pred_conf_sum / max(1, len(signal_audit_rows)), 6),
    }
    top_patterns = [
        {"pattern_id": pid, "count": cnt}
        for pid, cnt in sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    ]
    gpt_enabled = bool(settings.openai_api_key) if use_gpt is None else bool(use_gpt)
    if gpt_enabled and not settings.openai_api_key:
        gpt_enabled = False
    gpt_model = settings.trident_gpt_model or "gpt-5.2"
    narrative = {
        "source": "deterministic_fallback",
        "model": None,
        "gpt_error": None,
        "summary": (
            f"Paper equity={report['equity']['equity']} with {len(report.get('open_positions') or [])} open positions "
            f"and {len(report.get('closed_trades') or [])} recent closed trades."
        ),
    }
    if gpt_enabled:
        print(f">>> CALLING OPENAI {gpt_model}")
        g = render_paper_report_summary_gpt52(
            {
                "window": report.get("window"),
                "equity": report.get("equity"),
                "open_positions": report.get("open_positions"),
                "closed_trades": report.get("closed_trades")[:10],
                "gate_frequency": report.get("gate_frequency")[:10],
                "prediction_summary": prediction_summary,
                "top_patterns": top_patterns,
            },
            model=gpt_model,
        )
        if isinstance(g, dict) and isinstance(g.get("summary"), str) and g.get("summary"):
            narrative = {"source": "gpt", "model": gpt_model, "gpt_error": None, "summary": g["summary"]}
        else:
            narrative["gpt_error"] = str((g or {}).get("error") or "unknown_gpt_error")
    print(
        f"[paper:report] window={report.get('window')} equity={report['equity']['equity']} "
        f"open_positions={len(report.get('open_positions') or [])} closed_trades={len(report.get('closed_trades') or [])}"
    )
    print(
        json.dumps(
            {
                "command": "paper:report",
                **report,
                "narrative": narrative,
            },
            indent=2,
            sort_keys=False,
            default=str,
        )
    )


@app.command("paper:learn")
def paper_learn_cmd(
    since: Optional[int] = typer.Option(None, "--since", help="Lookback in hours"),
    last: Optional[int] = typer.Option(None, "--last", help="Use last N closed trades"),
    apply: bool = typer.Option(False, "--apply", help="Apply learned updates"),
    explain: bool = typer.Option(False, "--explain", help="Include per-trade labels"),
    use_gpt: Optional[bool] = typer.Option(
        None,
        "--use-gpt/--no-use-gpt",
        help="Use GPT narrative. If unset, uses TRIDENT_USE_GPT env.",
    ),
) -> None:
    from backend.db.paper_repo import (
        get_active_paper_config,
        get_last_applied_learning_event_ts,
        get_last_stable_paper_config,
        list_signal_audit,
        record_adjustment_candidates,
        record_adjustment_rollback,
        record_adjustment_run,
        upsert_paper_config,
    )
    from backend.services.paper_learning import (
        build_smart_adjustment_candidates,
        classify_failures,
        evaluate_killswitch_state,
        load_closed_trades,
        propose_parameter_updates,
    )
    from backend.services.paper_engine import resolve_intel_stage
    from backend.services.policy_ai import (
        render_paper_learning_policy_proposal_gpt52,
        render_paper_learning_summary_gpt52,
    )

    settings = get_settings()
    lock_ctx = writer_lock(settings.database_path)
    lock_ctx.__enter__()
    con = get_connection(settings.database_path)
    apply_all_migrations(con)
    config = get_active_paper_config(con)
    if not config:
        print("[paper:learn] no active paper config; run paper:init first")
        try:
            con.close()
        except Exception:
            pass
        try:
            lock_ctx.__exit__(None, None, None)
        except Exception:
            pass
        raise typer.Exit(code=2)

    if since is None and last is None:
        last = 10
    trades = load_closed_trades(con, since_hours=since, last_n=last)
    counts, labeled = classify_failures(trades)
    risk_limits = dict(config.get("risk_limits") or {})
    learning_policy = dict(config.get("learning_policy") or {})
    intel_stage = resolve_intel_stage(
        closed_trades=len(trades),
        mode=str(risk_limits.get("intelligence_mode") or getattr(settings, "paper_intel_mode", "auto")),
        bootstrap=int(risk_limits.get("intelligence_bootstrap_trades") or getattr(settings, "paper_intel_bootstrap_trades", 25)),
        promotion=int(risk_limits.get("intelligence_promotion_trades") or getattr(settings, "paper_intel_promotion_trades", 50)),
    )
    stage_label = str(intel_stage.get("stage") or "shadow")
    bootstrap_guard_active = bool(intel_stage.get("bootstrap_guard_active"))
    gpt_min_trades = int(
        learning_policy.get("gpt_learn_min_trades")
        or getattr(settings, "paper_gpt_learn_min_trades", 25)
        or 25
    )
    learn_bootstrap_stop_only = bool(
        learning_policy.get("learn_bootstrap_stop_only")
        if learning_policy.get("learn_bootstrap_stop_only") is not None
        else getattr(settings, "paper_learn_bootstrap_stop_only", True)
    )
    if bootstrap_guard_active and learn_bootstrap_stop_only:
        learn_scope = "stop_only"
    elif stage_label == "soft":
        learn_scope = "stop_plus_entry"
    else:
        learn_scope = "full"
    freeze_aggression_baseline = bool(bootstrap_guard_active)
    cooldown_hours = int(
        learning_policy.get("learn_apply_cooldown_hours")
        or getattr(settings, "paper_learn_apply_cooldown_hours", 24)
        or 24
    )
    last_applied_ts = get_last_applied_learning_event_ts(con)
    apply_cooldown_passed = True
    apply_block_reason = None
    if apply and cooldown_hours > 0 and last_applied_ts is not None:
        age_h = (datetime.now(timezone.utc) - last_applied_ts).total_seconds() / 3600.0
        if age_h < float(cooldown_hours):
            apply_cooldown_passed = False
            apply_block_reason = "cooldown_active"
    try:
        max_gpt_influence = float(
            learning_policy.get("gpt_learn_max_influence")
            if learning_policy.get("gpt_learn_max_influence") is not None
            else getattr(settings, "paper_gpt_learn_max_influence", 0.30)
        )
    except Exception:
        max_gpt_influence = float(getattr(settings, "paper_gpt_learn_max_influence", 0.30))
    gpt_enabled = bool(settings.openai_api_key) if use_gpt is None else bool(use_gpt)
    if gpt_enabled and not settings.openai_api_key:
        gpt_enabled = False
    gpt_model = settings.trident_gpt_model or "gpt-5.2"
    gpt_strategy_proposal: Dict[str, Any] = {}
    if gpt_enabled:
        print(f">>> CALLING OPENAI {gpt_model}")
        gpt_strategy_proposal = render_paper_learning_policy_proposal_gpt52(
            {
                "closed_trades": len(trades),
                "taxonomy_counts": counts,
                "risk_limits": risk_limits,
                "learning_policy": learning_policy,
                "recent_trades": trades[:25],
            },
            model=gpt_model,
        )
    changes, risk_new, policy_new, diff_text, gpt_strategy, arbiter = propose_parameter_updates(
        risk_limits=risk_limits,
        learning_policy=learning_policy,
        counts=counts,
        trades=trades,
        gpt_policy_proposal=gpt_strategy_proposal,
        max_gpt_influence=max_gpt_influence,
        min_trades_for_gpt_influence=gpt_min_trades,
        learn_scope=learn_scope,
        freeze_aggression_baseline=freeze_aggression_baseline,
    )
    applied_config_id = None
    apply_effective = False
    rollback_reference: Optional[str] = None
    selected_risk_new = dict(risk_new)
    selected_policy_new = dict(policy_new)
    selected_diff_text = str(diff_text)
    selected_candidate: Dict[str, Any] = {}

    narrative = {
        "source": "deterministic_fallback",
        "model": None,
        "summary": "Deterministic learning summary generated from closed paper trades.",
        "what_changed": [f"{k}: {v['old']} -> {v['ema_new']}" for k, v in changes.items()],
        "key_risks": [k for k, v in counts.items() if int(v) > 0][:5] or ["Insufficient closed-trade failures for strong inference."],
    }
    audit_rows = list_signal_audit(con, last_n=1000)
    bucket_bounds = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bucket_map: Dict[str, Dict[str, Any]] = {}
    for lo, hi in bucket_bounds:
        key = f"{lo:.2f}-{min(1.0, hi):.2f}"
        bucket_map[key] = {"count": 0, "avg_ev_r": 0.0}
    pattern_counts: Dict[str, int] = {}
    pattern_pnl_sum: Dict[str, float] = {}
    pattern_pnl_count: Dict[str, int] = {}
    closed_trade_map: Dict[tuple, float] = {}
    for t in trades:
        run_key = str(t.get("linked_run_id") or "")
        sym_key = str(t.get("symbol") or "")
        if run_key and sym_key:
            closed_trade_map[(run_key, sym_key)] = float(t.get("gross_pnl") or 0.0)
    for row in audit_rows:
        pred = row.get("prediction") or {}
        conf = max(0.0, min(1.0, float(pred.get("confidence") or 0.0)))
        ev_r = float(pred.get("ev_r") or 0.0)
        for lo, hi in bucket_bounds:
            if conf >= lo and conf < hi:
                key = f"{lo:.2f}-{min(1.0, hi):.2f}"
                bucket_map[key]["count"] += 1
                bucket_map[key]["avg_ev_r"] += ev_r
                break
        pats = (row.get("patterns") or {}).get("hits") or []
        map_key = (str(row.get("run_id") or ""), str(row.get("symbol") or ""))
        realized = closed_trade_map.get(map_key)
        for pid in pats:
            p = str(pid)
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
            if realized is not None:
                pattern_pnl_sum[p] = pattern_pnl_sum.get(p, 0.0) + float(realized)
                pattern_pnl_count[p] = pattern_pnl_count.get(p, 0) + 1
    prediction_calibration = []
    for k, v in bucket_map.items():
        cnt = int(v.get("count") or 0)
        avg_ev = float(v.get("avg_ev_r") or 0.0) / cnt if cnt > 0 else 0.0
        prediction_calibration.append({"bucket": k, "count": cnt, "avg_ev_r": round(avg_ev, 6)})
    total_audits = max(1, len(audit_rows))
    pattern_hit_rate = [
        {"pattern_id": pid, "count": cnt, "hit_rate": round(float(cnt) / float(total_audits), 6)}
        for pid, cnt in sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    ]
    pattern_expectancy = [
        {
            "pattern_id": pid,
            "closed_trades": int(pattern_pnl_count.get(pid, 0)),
            "avg_gross_pnl": round(float(pattern_pnl_sum.get(pid, 0.0)) / float(pattern_pnl_count.get(pid, 1)), 6),
        }
        for pid in sorted(pattern_pnl_count.keys(), key=lambda p: (-pattern_pnl_count.get(p, 0), p))[:12]
    ]

    smart_eval = build_smart_adjustment_candidates(
        trades=trades,
        counts=counts,
        audit_rows=audit_rows,
        risk_limits=risk_limits,
        learning_policy=learning_policy,
        numeric_changes=changes,
        numeric_risk_new=risk_new,
        numeric_policy_new=policy_new,
        numeric_diff_text=diff_text,
        gpt_strategy=gpt_strategy,
        arbiter=arbiter,
    )
    policy_candidates: List[Dict[str, Any]] = list(smart_eval.get("candidates") or [])
    selected_candidate = dict(smart_eval.get("selected") or {})
    selected_candidate_id = str(selected_candidate.get("candidate_id") or "")
    selected_candidate_score = float(selected_candidate.get("score") or 0.0)
    selected_candidate_confidence = float(selected_candidate.get("confidence") or 0.0)
    projected_success_fail_ratio = dict(smart_eval.get("projected_success_fail_ratio") or {})
    if selected_candidate:
        selected_risk_new = dict(selected_candidate.get("risk_limits") or risk_new)
        selected_policy_new = dict(selected_candidate.get("learning_policy") or policy_new)
        selected_diff_text = str(selected_candidate.get("diff_text") or diff_text)

    killswitch_state = evaluate_killswitch_state(
        trades=trades,
        learning_policy=selected_policy_new or learning_policy,
    )
    if bool(killswitch_state.get("active")):
        stable_cfg = get_last_stable_paper_config(con) or {}
        rollback_reference = str(stable_cfg.get("config_id") or "") or None

    if apply and not apply_cooldown_passed:
        apply_block_reason = "cooldown_active"
    elif apply and bool(killswitch_state.get("active")):
        apply_block_reason = "kill_switch_active"

    if apply and not apply_block_reason:
        apply_effective = True
        new_config = dict(config)
        new_config["risk_limits"] = selected_risk_new
        new_config["learning_policy"] = selected_policy_new
        new_config.pop("config_id", None)
        new_config.pop("created_at", None)
        active_new = upsert_paper_config(con, new_config, set_active=True)
        applied_config_id = active_new.get("config_id")

    def _compact_gate_policy_changes(raw: Dict[str, Any]) -> Dict[str, Any]:
        frm = dict((raw or {}).get("from") or {})
        to = dict((raw or {}).get("to") or {})
        changed = json.dumps(frm, sort_keys=True, default=str) != json.dumps(to, sort_keys=True, default=str)
        return {
            "changed": bool(changed),
            "from_version": str(frm.get("policy_version") or ""),
            "to_version": str(to.get("policy_version") or ""),
            "from_weighted_checks": int(len(list(frm.get("enabled_weighted_checks") or []))),
            "to_weighted_checks": int(len(list(to.get("enabled_weighted_checks") or []))),
            "from_diagnostic_checks": int(len(list(frm.get("diagnostic_only_checks") or []))),
            "to_diagnostic_checks": int(len(list(to.get("diagnostic_only_checks") or []))),
        }

    def _compact_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rank": int(cand.get("rank") or 0),
            "candidate_id": str(cand.get("candidate_id") or ""),
            "candidate_type": str(cand.get("candidate_type") or ""),
            "score": round(float(cand.get("score") or 0.0), 6),
            "confidence": round(float(cand.get("confidence") or 0.0), 6),
            "reasoning": str(cand.get("reasoning") or ""),
            "projected_success_fail_ratio": dict(cand.get("projected_success_fail_ratio") or {}),
            "gate_policy_changes": _compact_gate_policy_changes(dict(cand.get("gate_policy_changes") or {})),
            "selected": bool(cand.get("selected")),
            "applied": bool(cand.get("applied")),
        }

    recommended_changes_ranked: List[Dict[str, Any]] = []
    for cand in policy_candidates:
        cid = str(cand.get("candidate_id") or "")
        if cid and cid == selected_candidate_id and apply_effective:
            cand["applied"] = True
        recommended_changes_ranked.append(_compact_candidate(cand))

    compact_policy_candidates = [dict(x) for x in recommended_changes_ranked[:3]]
    compact_selected_candidate = _compact_candidate(selected_candidate) if selected_candidate else {}

    worked = ((smart_eval.get("cohorts") or {}).get("worked") or [])[:3]
    failed = ((smart_eval.get("cohorts") or {}).get("failed") or [])[:3]
    compact_worked = [
        {
            "cohort": str(x.get("cohort") or ""),
            "count": int(x.get("count") or 0),
            "expectancy": float(x.get("expectancy") or 0.0),
            "win_rate": float(x.get("win_rate") or 0.0),
        }
        for x in worked
    ]
    compact_failed = [
        {
            "cohort": str(x.get("cohort") or ""),
            "count": int(x.get("count") or 0),
            "expectancy": float(x.get("expectancy") or 0.0),
            "win_rate": float(x.get("win_rate") or 0.0),
        }
        for x in failed
    ]

    compact_killswitch = {
        "active": bool(killswitch_state.get("active")),
        "reason_codes": [str(r.get("code") or "") for r in list(killswitch_state.get("reasons") or []) if r.get("code")],
        "limits": dict(killswitch_state.get("limits") or {}),
        "metrics": {
            "count": float(((killswitch_state.get("metrics") or {}).get("count") or 0.0)),
            "expectancy": float(((killswitch_state.get("metrics") or {}).get("expectancy") or 0.0)),
            "win_rate": float(((killswitch_state.get("metrics") or {}).get("win_rate") or 0.0)),
            "drawdown_proxy": float(((killswitch_state.get("metrics") or {}).get("drawdown_proxy") or 0.0)),
            "instability": float(((killswitch_state.get("metrics") or {}).get("instability") or 0.0)),
        },
    }

    compact_proposed_changes: Dict[str, Any] = {}
    for key, val in (changes or {}).items():
        if key == "aggression_baseline":
            old_base = dict((val or {}).get("old") or {})
            new_base = dict((val or {}).get("ema_new") or {})
            delta = {}
            for k in sorted(set(list(old_base.keys()) + list(new_base.keys()))):
                if old_base.get(k) != new_base.get(k):
                    delta[k] = {"old": old_base.get(k), "new": new_base.get(k)}
            if delta:
                compact_proposed_changes[key] = delta
            continue
        old_val = (val or {}).get("old")
        new_val = (val or {}).get("ema_new")
        if old_val != new_val:
            compact_proposed_changes[key] = {"old": old_val, "new": new_val}

    compact_prediction_pattern_diagnostics = {
        "prediction_calibration": [
            {
                "bucket": str(b.get("bucket") or ""),
                "count": int(b.get("count") or 0),
                "avg_ev_r": float(b.get("avg_ev_r") or 0.0),
            }
            for b in prediction_calibration
            if int(b.get("count") or 0) > 0
        ],
        "top_patterns": list(pattern_hit_rate[:5]),
        "pattern_expectancy_count": int(len(pattern_expectancy)),
    }

    taxonomy_counts_compact = {k: int(v) for k, v in (counts or {}).items() if int(v or 0) > 0}
    compact_policy_candidates_for_output = [
        {
            "rank": int(c.get("rank") or 0),
            "candidate_id": str(c.get("candidate_id") or ""),
            "candidate_type": str(c.get("candidate_type") or ""),
            "score": float(c.get("score") or 0.0),
            "confidence": float(c.get("confidence") or 0.0),
        }
        for c in compact_policy_candidates
    ]
    compact_stability_guardrails = {
        "stage": stage_label,
        "closed_trades": len(trades),
        "gpt_numeric_enabled": bool((len(trades) >= gpt_min_trades) and (max_gpt_influence > 0)),
        "learn_scope": learn_scope,
        "apply_cooldown_passed": bool(apply_cooldown_passed),
        "kill_switch_active": bool(killswitch_state.get("active")),
        "frozen_params_count": int(len(list((arbiter or {}).get("frozen_params") or []))),
        "apply_block_reason": apply_block_reason,
    }

    compact_diff_items: List[str] = []
    for key, value in compact_proposed_changes.items():
        if key == "aggression_baseline":
            for subk, sv in dict(value or {}).items():
                compact_diff_items.append(f"{key}.{subk}:{sv.get('old')}->{sv.get('new')}")
        else:
            compact_diff_items.append(f"{key}:{value.get('old')}->{value.get('new')}")
    compact_diff_text = "; ".join(compact_diff_items) if compact_diff_items else "no material parameter changes"

    compact_arbiter = {
        "learn_scope": str((arbiter or {}).get("learn_scope") or ""),
        "frozen_params": list((arbiter or {}).get("frozen_params") or []),
        "rejected_or_clamped_reasons": list((arbiter or {}).get("rejected_or_clamped_reasons") or []),
    }

    smart_adjustment_report: Dict[str, Any] = {
        "what_worked": compact_worked,
        "what_failed": compact_failed,
        "recommended_changes_ranked": compact_policy_candidates,
        "reasoning": {
            "selected_candidate_id": selected_candidate_id or None,
            "selected_candidate_type": selected_candidate.get("candidate_type"),
            "selected_reasoning": selected_candidate.get("reasoning"),
            "base_alpha_score": smart_eval.get("base_alpha_score"),
        },
        "projected_success_fail_ratio": projected_success_fail_ratio,
        "gate_policy_changes": _compact_gate_policy_changes(dict(selected_candidate.get("gate_policy_changes") or {})),
        "risk_controls_and_killswitch": compact_killswitch,
    }
    if apply:
        smart_adjustment_report["apply_result"] = {
            "requested": True,
            "applied": bool(apply_effective),
            "apply_block_reason": apply_block_reason,
            "selected_candidate_id": selected_candidate_id or None,
            "applied_config_id": applied_config_id,
            "rollback_reference": rollback_reference,
        }
    else:
        smart_adjustment_report["apply_preview"] = {
            "requested": False,
            "would_apply_candidate_id": selected_candidate_id or None,
            "cooldown_passed": bool(apply_cooldown_passed),
            "kill_switch_active": bool(killswitch_state.get("active")),
        }

    if gpt_enabled:
        g = render_paper_learning_summary_gpt52(
            {
                "closed_trades": len(trades),
                "taxonomy_counts": counts,
                "changes": changes,
                "apply": bool(apply_effective),
                "prediction_pattern_diagnostics": {
                    "prediction_calibration": prediction_calibration,
                    "pattern_hit_rate": pattern_hit_rate,
                    "pattern_expectancy": pattern_expectancy,
                },
            },
            model=gpt_model,
        )
        if g:
            narrative = {"source": "gpt", "model": gpt_model, **g}

    now_utc = datetime.now(timezone.utc)
    learn_id = f"learn_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    con.execute(
        """
        INSERT INTO paper_learning_events (
            learn_id, ts, scope, summary, changes_json, applied, diff_text, source_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            learn_id,
            now_utc,
            f"since_{since}h" if since is not None else f"last_{last}",
            narrative.get("summary"),
            json.dumps(
                {
                    "counts": counts,
                    "changes": changes,
                    "gpt_strategy": gpt_strategy,
                    "arbiter": arbiter,
                    "smart_adjustment_report": smart_adjustment_report,
                    "stability_guardrails": {
                        "stage": stage_label,
                        "closed_trades": len(trades),
                        "gpt_numeric_enabled": bool((len(trades) >= gpt_min_trades) and (max_gpt_influence > 0)),
                        "learn_scope": learn_scope,
                        "apply_cooldown_passed": bool(apply_cooldown_passed),
                        "kill_switch_active": bool(killswitch_state.get("active")),
                        "frozen_params": list((arbiter or {}).get("frozen_params") or []),
                        "apply_block_reason": apply_block_reason,
                    },
                    "prediction_pattern_diagnostics": {
                        "prediction_calibration": prediction_calibration,
                        "pattern_hit_rate": pattern_hit_rate,
                        "pattern_expectancy": pattern_expectancy,
                    },
                },
                sort_keys=True,
            ),
            bool(apply_effective),
            selected_diff_text,
            narrative.get("model"),
        ],
    )
    adjustment_summary = {
        "taxonomy_counts": counts,
        "metrics": dict(smart_eval.get("metrics") or {}),
        "projected_success_fail_ratio": projected_success_fail_ratio,
        "stability_guardrails": {
            "stage": stage_label,
            "learn_scope": learn_scope,
            "apply_cooldown_passed": bool(apply_cooldown_passed),
            "apply_block_reason": apply_block_reason,
        },
    }
    adjustment_run_id = record_adjustment_run(
        con,
        {
            "ts": now_utc,
            "scope": f"since_{since}h" if since is not None else f"last_{last}",
            "selected_candidate_id": selected_candidate_id,
            "selected_score": selected_candidate_score,
            "selected_confidence": selected_candidate_confidence,
            "apply_requested": bool(apply),
            "applied": bool(apply_effective),
            "apply_block_reason": apply_block_reason,
            "kill_switch_active": bool(killswitch_state.get("active")),
            "rollback_reference": rollback_reference,
            "prior_config_id": config.get("config_id"),
            "applied_config_id": applied_config_id,
            "summary": adjustment_summary,
        },
    )
    record_adjustment_candidates(
        con,
        adjustment_run_id=adjustment_run_id,
        candidates=policy_candidates,
    )
    rollback_event_id: Optional[str] = None
    if apply and str(apply_block_reason or "") == "kill_switch_active":
        rollback_event_id = record_adjustment_rollback(
            con,
            adjustment_run_id=adjustment_run_id,
            from_config_id=str(config.get("config_id") or "") or None,
            to_config_id=rollback_reference,
            reason="apply_blocked_killswitch_active",
            details={"killswitch_state": killswitch_state},
        )

    print(
        f"[paper:learn] scope={'since_'+str(since)+'h' if since is not None else 'last_'+str(last)} "
        f"closed_trades={len(trades)} apply={str(apply_effective).lower()}"
    )
    output = {
        "version": SYSTEM_PHASE_VERSION,
        "command": "paper:learn",
        "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": {"since": since} if since is not None else {"last": last},
        "closed_trades": len(trades),
        "apply": bool(apply),
        "taxonomy_counts": taxonomy_counts_compact,
        "smart_adjustment_report": smart_adjustment_report,
        "policy_candidates": compact_policy_candidates_for_output,
        "selected_policy_candidate": compact_selected_candidate,
        "projected_success_fail_ratio": projected_success_fail_ratio,
        "killswitch_state": compact_killswitch,
        "rollback_reference": rollback_reference,
        "proposed_changes": compact_proposed_changes,
        "gpt_strategy": gpt_strategy,
        "arbiter": compact_arbiter,
        "stability_guardrails": compact_stability_guardrails,
        "prediction_pattern_diagnostics": compact_prediction_pattern_diagnostics,
        "applied": bool(apply_effective),
        "applied_config_id": applied_config_id,
        "learning_event_id": learn_id,
        "adjustment_run_id": adjustment_run_id,
        "rollback_event_id": rollback_event_id,
        "diff_text": compact_diff_text,
        "narrative": {
            "source": narrative.get("source"),
            "model": narrative.get("model"),
            "summary": narrative.get("summary"),
        },
    }
    if explain:
        output["policy_candidates_full"] = policy_candidates
        output["selected_policy_candidate_full"] = selected_candidate
        output["arbiter_full"] = arbiter
        output["prediction_pattern_diagnostics_full"] = {
            "prediction_calibration": prediction_calibration,
            "pattern_hit_rate": pattern_hit_rate,
            "pattern_expectancy": pattern_expectancy,
        }
        output["killswitch_state_full"] = killswitch_state
        output["proposed_changes_full"] = changes
        output["taxonomy_counts_full"] = counts
        output["smart_adjustment_report_full"] = {
            "what_worked": ((smart_eval.get("cohorts") or {}).get("worked") or []),
            "what_failed": ((smart_eval.get("cohorts") or {}).get("failed") or []),
            "recommended_changes_ranked": recommended_changes_ranked,
            "reasoning": {
                "selected_candidate_id": selected_candidate_id or None,
                "selected_candidate_type": selected_candidate.get("candidate_type"),
                "selected_reasoning": selected_candidate.get("reasoning"),
                "base_alpha_score": smart_eval.get("base_alpha_score"),
            },
            "projected_success_fail_ratio": projected_success_fail_ratio,
            "gate_policy_changes": dict(selected_candidate.get("gate_policy_changes") or {}),
            "risk_controls_and_killswitch": killswitch_state,
            "apply_result": smart_adjustment_report.get("apply_result"),
            "apply_preview": smart_adjustment_report.get("apply_preview"),
        }
        output["diff_text_full"] = selected_diff_text
        output["narrative"]["what_changed"] = narrative.get("what_changed")
        output["narrative"]["key_risks"] = narrative.get("key_risks")
        output["per_trade_labels"] = labeled
    print(json.dumps(output, indent=2, sort_keys=False, default=str))
    try:
        con.close()
    except Exception:
        pass
    try:
        lock_ctx.__exit__(None, None, None)
    except Exception:
        pass


@app.command("news:features")
def news_features_cmd(symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")) -> None:
    """
    Compute and print news-based sentiment & article summaries.
    """
    settings = get_settings()
    con = get_connection(settings.database_path)
    from backend.features.news_source_quality import source_family

    features = compute_news_features(symbol, con)

    print(f"[news:features] {symbol}")
    print(f"  direction: {features['direction']}")
    print(f"  intensity: {features['intensity']}")
    print(f"  article_count: {features['article_count']}")
    print(f"  window_hours: {features['window_hours']}")

    # New: weighted stats (with safe defaults)
    weighted = features.get("weighted_stats", {}) or {}
    wb = weighted.get("weighted_bullish", 0.0)
    wr = weighted.get("weighted_raw", 0.0)
    wbr = weighted.get("weighted_bearish", 0.0)
    print(f"  weighted_bullish: {wb}")
    print(f"  weighted_bearish: {wbr}")
    print(f"  weighted_raw: {wr}")

    # New: category breakdown
    categories = features.get("category_breakdown", {}) or {}
    if categories:
        print("  category_breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: (-x[1], x[0])):
            print(f"    - {cat}: {count}")

    # Macro/geopolitical context (merged with macro_highlights to avoid duplicate sections)
    market_context = features.get("market_context") or []
    macro_highlights = features.get("macro_highlights") or []
    merged_macro = []
    seen_macro = set()
    seen_macro_families = set()
    for item in market_context:
        family = source_family(item.get("source") or "")
        if family and family in seen_macro_families:
            continue
        key = ((item.get("source") or "").strip().lower(), (item.get("title") or "").strip().lower())
        if key in seen_macro:
            continue
        seen_macro.add(key)
        if family:
            seen_macro_families.add(family)
        merged_macro.append(item)
    for item in macro_highlights:
        family = source_family(item.get("source") or "")
        if family and family in seen_macro_families:
            continue
        key = ((item.get("source") or "").strip().lower(), (item.get("title") or "").strip().lower())
        if key in seen_macro:
            continue
        seen_macro.add(key)
        if family:
            seen_macro_families.add(family)
        merged_macro.append(
            {
                "source": item.get("source"),
                "title": item.get("title"),
                "published_at": item.get("published_at"),
                "summary": "",
            }
        )
    if merged_macro:
        print("")
        print("Macro/geopolitical news:")
        for item in merged_macro[:10]:
            ts = item.get("published_at") or "unknown"
            print(f"  - [{item.get('source')}] {item.get('title')}  ({ts})")

    policy_events = features.get("policy_events") or []
    if policy_events:
        print("")
        print("Policy events (US federal crypto):")
        for item in policy_events[:10]:
            ts = item["published_at"]
            stage = item.get("stage") or "unknown"
            branch = item.get("branch") or "unknown"
            conf = item.get("confidence")
            if conf is None:
                print(f"  - [{item['source']}] {item['title']}  ({ts}) [{branch}/{stage}]")
            else:
                print(f"  - [{item['source']}] {item['title']}  ({ts}) [{branch}/{stage}] conf={conf}")

    sector_highlights = features.get("sector_highlights") or []
    if sector_highlights:
        print("")
        print("Sector highlights:")
        for item in sector_highlights[:3]:
            ts = item["published_at"]
            print(f"  - [{item['source']}] {item['title']}  ({ts})")

    print("")
    print("Recent relevant news (filtered):")
    for item in features["articles"]:
        # item['published_at'] is already formatted like "YYYY-MM-DD HH:MM UTC"
        ts = item["published_at"]
        print(f"  - [{item['source']}] {item['title']}  ({ts})")


@app.command("signal:fusion")
def signal_fusion_cmd(symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")) -> None:
    """
    Compute Phase 3A signal fusion (structural, non-predictive).
    """
    settings = get_settings()
    con = get_connection(settings.database_path)

    fusion = compute_signal_fusion(symbol, con)
    print(json.dumps(fusion, indent=2))


@app.command("scenario:analysis")
def scenario_analysis_cmd(symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")) -> None:
    """
    Phase 4A scenario analysis: interpret scenarios using a canonical feature snapshot only.
    """
    settings = get_settings()
    con = get_connection(settings.database_path)
    try:
        snapshot = build_phase4_snapshot(symbol, con)
        gate = _required_modality_gate(symbol, snapshot)
        if gate.get("failed"):
            print(json.dumps(_required_modality_failure_payload("scenario:analysis", symbol, gate, snapshot), indent=2))
            raise typer.Exit(code=1)
        analysis = interpret_snapshot(snapshot, con)
        print(json.dumps(analysis, indent=2))
    finally:
        try:
            con.close()
        except Exception:
            pass


@app.command("all:analysis")
def all_analysis_cmd(symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")) -> None:
    """
    Phase 4 scaffold: run scenario analysis and prepare for additional Phase 4 components.
    """
    settings = get_settings()
    con = get_connection(settings.database_path)
    try:
        snapshot = build_phase4_snapshot(symbol, con)
        gate = _required_modality_gate(symbol, snapshot)
        if gate.get("failed"):
            print(json.dumps(_required_modality_failure_payload("all:analysis", symbol, gate, snapshot), indent=2))
            raise typer.Exit(code=1)
        scenario_analysis = interpret_snapshot(snapshot, con)

        combined = {
            "symbol": symbol,
            "scenario_analysis": scenario_analysis,
            "regime_analysis": None,
            "conflict_analysis": None,
            "risk_analysis": None,
        }
        print(json.dumps(combined, indent=2))
    finally:
        try:
            con.close()
        except Exception:
            pass


@app.command("tech:features")
def tech_features_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")
) -> None:
    """
    Compute and print technical indicators for a symbol (similar style to news:features).
    """
    settings = get_settings()
    con = get_connection(settings.database_path)

    features = compute_tech_features(symbol, con)

    print(f"[tech:features] {symbol}")
    print(json.dumps(features, indent=2))


@app.command("fundamentals:features")
def fundamentals_features_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")
) -> None:
    """
    Compute and print fundamentals for a symbol (similar style to news:features).
    """
    settings = get_settings()
    con = get_connection(settings.database_path)

    features = compute_fundamentals_features(symbol, con)

    print(f"[fundamentals:features] {symbol}")
    for key, val in features.items():
        print(f"  {key}: {val}")


@app.command("all:features")
def all_features_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Symbol like BTC-USD")
) -> None:
    """
    Print combined technical, fundamentals, and news features for a symbol.
    This command must NOT call any other CLI commands; it should only use
    feature helper functions that return data (no printing inside helpers).
    """
    from backend.config.env import get_settings
    from backend.db.schema import get_connection
    from backend.features.tech_features import compute_tech_features
    from backend.features.fundamentals_features import compute_fundamentals_features
    from backend.features.news_features import compute_news_features

    settings = get_settings()
    con = get_connection(settings.database_path)

    tech = compute_tech_features(symbol, con)
    fund = compute_fundamentals_features(symbol, con)
    news = compute_news_features(symbol, con)

    print(f"[all:features] {symbol}")

    # --- Technicals ---
    print("  === Technicals ===")
    for section, values in tech.items():
        print(f"    {section}: {values}")

    # --- Fundamentals ---
    print("  === Fundamentals ===")
    ordered_keys = [
        "price",
        "mkt_cap",
        "fdv",
        "circ_supply",
        "total_supply",
        "circ_ratio",
        "mkt_cap_rank",
        "pct_change_1h",
        "pct_change_24h",
        "pct_change_7d",
        "pct_change_30d",
        "pct_change_1y",
        "ath_price",
        "ath_change_pct",
        "atl_price",
        "atl_change_pct",
        "vol_24h",
        "vol_mcap_ratio",
        "mcap_change_1d",
        "dominance_pct",
        "liquidity_score",
        "cg_score",
        "dev_score",
        "community_score",
        "mktcap_fdv_ratio",
        "mkt_cap_change_7d",
        "mkt_cap_change_30d",
        "vol_24h_change_7d",
        "vol_mcap_ratio_z_30d",
        "fundamental_trend_score",
        "vol_change_1d",
        "tier",
        "liquidity",
        "volatility_flag",
        "summary",
    ]

    for key in ordered_keys:
        if key in fund:
            print(f"    {key}: {fund[key]}")
    for key, value in fund.items():
        if key not in ordered_keys:
            print(f"    {key}: {value}")

    # --- News ---
    print("  === News ===")
    direction = news.get("direction", "neutral")
    intensity = news.get("intensity", 0.0)
    article_count = news.get("article_count", 0)
    window_hours = news.get("window_hours", 24)

    if direction == "bullish":
        dir_label = "▲ bullish"
    elif direction == "bearish":
        dir_label = "▼ bearish"
    else:
        dir_label = "→ neutral"

    print(f"    direction: {dir_label}")
    print(f"    intensity: {intensity}")
    print(f"    article_count: {article_count}")
    print(f"    window_hours: {window_hours}")
    print("")
    print("  Recent relevant news (filtered):")
    for item in news.get("articles", []):
        print(f"    - [{item['source']}] {item['title']}  ({item['published_at'].split(' ')[0]})")


if __name__ == "__main__":
    app()
