"""Environment and configuration loader for Project Trident."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# env.py is located at backend/config/env.py -> project root is two levels up.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
_ENV_LOADED = False


@dataclass(frozen=True)
class Settings:
    """Application configuration derived from environment variables."""

    env: str
    feature_equities: bool
    crypto_symbols: List[str]
    database_path: Path
    data_vendor: str
    disable_yf: bool
    news_vendor: str
    onchain_vendor: str
    ai_vendor: str
    ai_model: str
    trident_use_gpt: bool
    trident_gpt_model: str
    newsapi_ai_key: Optional[str]
    finnhub_api_key: Optional[str]
    fred_api_key: Optional[str]
    calendar_countries: List[str]
    coingecko_base: str
    kraken_base: str
    openai_api_key: Optional[str]
    paper_starting_equity: float
    paper_symbols: List[str]
    paper_fee_bps: float
    paper_slippage_bps: float
    paper_max_trades_per_run: int
    paper_max_open_positions: int
    paper_max_open_positions_per_symbol: int
    paper_max_risk_per_trade_pct: float
    paper_max_total_exposure_pct: float
    paper_replay_interval: str
    paper_replay_lookback_bars: int
    paper_entry_min_score: float
    paper_entry_min_effective_confidence: float
    paper_entry_min_agreement: float
    paper_entry_min_margin: float
    paper_quality_veto_enabled: bool
    paper_gpt_learn_max_influence: float
    paper_intel_mode: str
    paper_intel_bootstrap_trades: int
    paper_intel_promotion_trades: int
    paper_prediction_enabled: bool
    paper_pattern_enabled: bool
    paper_prediction_weight_soft: float
    paper_pattern_weight_soft: float
    paper_prediction_weight_hard: float
    paper_pattern_weight_hard: float
    paper_allow_weighted_gate_override: bool
    paper_learn_bootstrap_stop_only: bool
    paper_learn_apply_cooldown_hours: int
    paper_gpt_learn_min_trades: int
    paper_adjust_min_cohort_trades: int
    paper_adjust_structural_change_budget: int
    paper_adjust_weight_expectancy: float
    paper_adjust_weight_win_rate: float
    paper_adjust_weight_drawdown_penalty: float
    paper_adjust_weight_instability_penalty: float
    paper_adjust_killswitch_drawdown_spike: float
    paper_adjust_killswitch_flip_rate: float
    paper_adjust_killswitch_conf_collapse: float
    paper_news_min_interval_minutes: int
    paper_news_max_pulls_per_day: int
    paper_weekend_rescue_guard: bool
    paper_weekend_rescue_notional_cap: float
    paper_entry_rescue_enabled: bool
    paper_entry_rescue_max_per_run: int
    paper_entry_rescue_min_score: float
    paper_entry_rescue_min_effective_confidence: float
    paper_entry_rescue_min_agreement: float
    paper_entry_rescue_min_margin: float
    paper_entry_rescue_notional_cap: float
    paper_entry_rescue_risk_mult: float
    paper_entry_rescue_stop_mult: float
    paper_entry_rescue_hold_mult: float
    paper_db_schema_version: str


def _load_env() -> None:
    """Load the project .env file once and push values into os.environ."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    _ENV_LOADED = True


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    """Parse common string representations of booleans with a safe default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_symbols(raw: str | None) -> List[str]:
    """Split a comma-delimited symbol string into a clean list."""
    if not raw:
        return []
    return [symbol.strip() for symbol in raw.split(",") if symbol.strip()]


def _parse_calendar_countries(raw: str | None) -> List[str]:
    """Split TRIDENT_CALENDAR_COUNTRIES into a list of uppercase country codes."""
    if raw is None or not raw.strip():
        raw = "US,JP"
    codes = [code.strip().upper() for code in raw.split(",") if code.strip()]
    # Deduplicate while preserving order
    seen = set()
    cleaned: List[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            cleaned.append(code)
    return cleaned


def _optional_str(key: str) -> Optional[str]:
    """Return a non-empty environment value or None when unset/blank."""
    value = os.getenv(key)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value.strip())
    except Exception:
        return float(default)


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value.strip())
    except Exception:
        return int(default)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the loaded application settings."""
    _load_env()
    default_db = _PROJECT_ROOT / "backend" / "var" / "trident.duckdb"
    return Settings(
        env=os.getenv("ENV", "dev"),
        feature_equities=_parse_bool(os.getenv("FEATURE_EQUITIES"), default=False),
        crypto_symbols=_parse_symbols(os.getenv("CRYPTO_SYMBOLS")),
        database_path=Path(os.getenv("DATABASE_PATH", str(default_db))),
        data_vendor=os.getenv("DATA_VENDOR", "kraken"),
        disable_yf=_parse_bool(os.getenv("DISABLE_YF"), default=False),
        news_vendor=os.getenv("NEWS_VENDOR", "newsapi_ai"),
        onchain_vendor=os.getenv("ONCHAIN_VENDOR", "coingecko"),
        ai_vendor=os.getenv("AI_VENDOR", "openai"),
        ai_model=os.getenv("AI_MODEL", "gpt-5-mini"),
        trident_use_gpt=_parse_bool(os.getenv("TRIDENT_USE_GPT"), default=False),
        trident_gpt_model=os.getenv("TRIDENT_GPT_MODEL", "gpt-5.2"),
        newsapi_ai_key=_optional_str("NEWSAPI_AI_KEY"),
        finnhub_api_key=_optional_str("FINNHUB_API_KEY"),
        fred_api_key=_optional_str("FRED_API_KEY"),
        calendar_countries=_parse_calendar_countries(os.getenv("TRIDENT_CALENDAR_COUNTRIES")),
        coingecko_base=os.getenv("COINGECKO_BASE", "https://api.coingecko.com/api/v3"),
        kraken_base=os.getenv("KRAKEN_BASE", "https://api.kraken.com/0/public"),
        openai_api_key=_optional_str("OPENAI_API_KEY"),
        paper_starting_equity=_parse_float(os.getenv("PAPER_STARTING_EQUITY"), 10000.0),
        paper_symbols=_parse_symbols(
            os.getenv("PAPER_SYMBOLS", "BTC-USD,ETH-USD,XRP-USD,SOL-USD,SUI-USD")
        ),
        paper_fee_bps=_parse_float(os.getenv("PAPER_FEE_BPS"), 5.0),
        paper_slippage_bps=_parse_float(os.getenv("PAPER_SLIPPAGE_BPS"), 8.0),
        paper_max_trades_per_run=_parse_int(os.getenv("PAPER_MAX_TRADES_PER_RUN"), 1),
        paper_max_open_positions=_parse_int(os.getenv("PAPER_MAX_OPEN_POSITIONS"), 5),
        paper_max_open_positions_per_symbol=_parse_int(os.getenv("PAPER_MAX_OPEN_POSITIONS_PER_SYMBOL"), 1),
        paper_max_risk_per_trade_pct=_parse_float(os.getenv("PAPER_MAX_RISK_PER_TRADE_PCT"), 0.01),
        paper_max_total_exposure_pct=_parse_float(os.getenv("PAPER_MAX_TOTAL_EXPOSURE_PCT"), 0.60),
        paper_replay_interval=os.getenv("PAPER_REPLAY_INTERVAL", "15m"),
        paper_replay_lookback_bars=_parse_int(os.getenv("PAPER_REPLAY_LOOKBACK_BARS"), 672),
        paper_entry_min_score=_parse_float(os.getenv("PAPER_ENTRY_MIN_SCORE"), 0.18),
        paper_entry_min_effective_confidence=_parse_float(
            os.getenv("PAPER_ENTRY_MIN_EFFECTIVE_CONFIDENCE"), 0.40
        ),
        paper_entry_min_agreement=_parse_float(os.getenv("PAPER_ENTRY_MIN_AGREEMENT"), 0.60),
        paper_entry_min_margin=_parse_float(os.getenv("PAPER_ENTRY_MIN_MARGIN"), 0.06),
        paper_quality_veto_enabled=_parse_bool(os.getenv("PAPER_QUALITY_VETO_ENABLED"), default=True),
        paper_gpt_learn_max_influence=_parse_float(os.getenv("PAPER_GPT_LEARN_MAX_INFLUENCE"), 0.30),
        paper_intel_mode=os.getenv("PAPER_INTEL_MODE", "auto"),
        paper_intel_bootstrap_trades=_parse_int(os.getenv("PAPER_INTEL_BOOTSTRAP_TRADES"), 25),
        paper_intel_promotion_trades=_parse_int(os.getenv("PAPER_INTEL_PROMOTION_TRADES"), 50),
        paper_prediction_enabled=_parse_bool(os.getenv("PAPER_PREDICTION_ENABLED"), default=True),
        paper_pattern_enabled=_parse_bool(os.getenv("PAPER_PATTERN_ENABLED"), default=True),
        paper_prediction_weight_soft=_parse_float(os.getenv("PAPER_PREDICTION_WEIGHT_SOFT"), 0.15),
        paper_pattern_weight_soft=_parse_float(os.getenv("PAPER_PATTERN_WEIGHT_SOFT"), 0.10),
        paper_prediction_weight_hard=_parse_float(os.getenv("PAPER_PREDICTION_WEIGHT_HARD"), 0.20),
        paper_pattern_weight_hard=_parse_float(os.getenv("PAPER_PATTERN_WEIGHT_HARD"), 0.10),
        paper_allow_weighted_gate_override=_parse_bool(
            os.getenv("PAPER_ALLOW_WEIGHTED_GATE_OVERRIDE"), default=False
        ),
        paper_learn_bootstrap_stop_only=_parse_bool(os.getenv("PAPER_LEARN_BOOTSTRAP_STOP_ONLY"), default=True),
        paper_learn_apply_cooldown_hours=_parse_int(os.getenv("PAPER_LEARN_APPLY_COOLDOWN_HOURS"), 24),
        paper_gpt_learn_min_trades=_parse_int(os.getenv("PAPER_GPT_LEARN_MIN_TRADES"), 25),
        paper_adjust_min_cohort_trades=_parse_int(os.getenv("PAPER_ADJUST_MIN_COHORT_TRADES"), 5),
        paper_adjust_structural_change_budget=_parse_int(os.getenv("PAPER_ADJUST_STRUCTURAL_CHANGE_BUDGET"), 2),
        paper_adjust_weight_expectancy=_parse_float(os.getenv("PAPER_ADJUST_WEIGHT_EXPECTANCY"), 0.45),
        paper_adjust_weight_win_rate=_parse_float(os.getenv("PAPER_ADJUST_WEIGHT_WIN_RATE"), 0.30),
        paper_adjust_weight_drawdown_penalty=_parse_float(os.getenv("PAPER_ADJUST_WEIGHT_DRAWDOWN_PENALTY"), 0.40),
        paper_adjust_weight_instability_penalty=_parse_float(
            os.getenv("PAPER_ADJUST_WEIGHT_INSTABILITY_PENALTY"), 0.25
        ),
        paper_adjust_killswitch_drawdown_spike=_parse_float(os.getenv("PAPER_ADJUST_KILLSWITCH_DRAWDOWN_SPIKE"), 0.22),
        paper_adjust_killswitch_flip_rate=_parse_float(os.getenv("PAPER_ADJUST_KILLSWITCH_FLIP_RATE"), 0.35),
        paper_adjust_killswitch_conf_collapse=_parse_float(
            os.getenv("PAPER_ADJUST_KILLSWITCH_CONF_COLLAPSE"), 0.55
        ),
        paper_news_min_interval_minutes=_parse_int(os.getenv("PAPER_NEWS_MIN_INTERVAL_MINUTES"), 60),
        paper_news_max_pulls_per_day=_parse_int(os.getenv("PAPER_NEWS_MAX_PULLS_PER_DAY"), 10),
        paper_weekend_rescue_guard=_parse_bool(os.getenv("PAPER_WEEKEND_RESCUE_GUARD"), default=False),
        paper_weekend_rescue_notional_cap=_parse_float(
            os.getenv("PAPER_WEEKEND_RESCUE_NOTIONAL_CAP"), 0.03
        ),
        paper_entry_rescue_enabled=_parse_bool(os.getenv("PAPER_ENTRY_RESCUE_ENABLED"), default=True),
        paper_entry_rescue_max_per_run=_parse_int(os.getenv("PAPER_ENTRY_RESCUE_MAX_PER_RUN"), 1),
        paper_entry_rescue_min_score=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_MIN_SCORE"), 0.05),
        paper_entry_rescue_min_effective_confidence=_parse_float(
            os.getenv("PAPER_ENTRY_RESCUE_MIN_EFFECTIVE_CONFIDENCE"), 0.22
        ),
        paper_entry_rescue_min_agreement=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_MIN_AGREEMENT"), 0.55),
        paper_entry_rescue_min_margin=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_MIN_MARGIN"), 0.015),
        paper_entry_rescue_notional_cap=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_NOTIONAL_CAP"), 0.06),
        paper_entry_rescue_risk_mult=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_RISK_MULT"), 0.70),
        paper_entry_rescue_stop_mult=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_STOP_MULT"), 1.30),
        paper_entry_rescue_hold_mult=_parse_float(os.getenv("PAPER_ENTRY_RESCUE_HOLD_MULT"), 0.90),
        paper_db_schema_version=os.getenv("PAPER_DB_SCHEMA_VERSION", "5C.0"),
    )
