from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from backend.config.env import get_settings
from backend.features.technicals import build_technicals, rsi

EPS = 1e-12


def parse_interval_minutes(interval: Optional[str]) -> int:
    if interval is None:
        return 60
    s = str(interval).strip().lower()
    if s.endswith("m"):
        try:
            return int(s[:-1])
        except Exception:
            return 60
    if s.endswith("h"):
        try:
            return int(s[:-1]) * 60
        except Exception:
            return 60
    return 60


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def ema_last(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    seed = float(np.mean(values[:period]))
    alpha = 2.0 / (period + 1.0)
    ema_v = seed
    for v in values[period:]:
        ema_v = (alpha * float(v)) + ((1.0 - alpha) * ema_v)
    return float(ema_v)


def rsi_wilder_last(values: List[float], period: int) -> Tuple[Optional[float], Optional[List[Optional[float]]]]:
    n = len(values)
    if period <= 0 or n < period + 1:
        return None, None

    deltas = [float(values[i] - values[i - 1]) for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    if len(gains) < period:
        return None, None

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    rsi_series: List[Optional[float]] = [None] * n

    rs = avg_gain / max(avg_loss, EPS)
    rsi_now = 100.0 - (100.0 / (1.0 + rs))
    rsi_series[period] = float(rsi_now)

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        rs = avg_gain / max(avg_loss, EPS)
        rsi_now = 100.0 - (100.0 / (1.0 + rs))
        rsi_series[i + 1] = float(rsi_now)

    rsi_last = rsi_series[-1]
    return (float(rsi_last) if rsi_last is not None else None), rsi_series


def rv_logret(values: List[float], window: int) -> Optional[float]:
    if window < 2 or len(values) < 2:
        return None
    rets: List[float] = []
    for i in range(1, len(values)):
        prev_v = float(values[i - 1])
        curr_v = float(values[i])
        if prev_v <= 0.0 or curr_v <= 0.0:
            continue
        rets.append(float(np.log(curr_v / prev_v)))
    if len(rets) < window or window < 2:
        return None
    tail = np.array(rets[-window:], dtype=float)
    if tail.size < 2:
        return None
    rv = float(np.std(tail, ddof=1))
    if np.isnan(rv):
        return None
    return rv


def horizon_signal(close: List[float], interval_minutes: int, horizon_bars: int) -> Dict[str, object]:
    bars_h = int(horizon_bars)
    out: Dict[str, object] = {
        "bars": bars_h,
        "trend_bias": None,
        "momentum_state": None,
        "vol_state": None,
        "composite_strength": None,
    }
    n = len(close)
    if n < bars_h + 2:
        return out

    fast_period = max(3, int(round(0.25 * bars_h)))
    slow_period = max(fast_period + 2, int(round(0.60 * bars_h)))
    ema_fast_last = ema_last(close, fast_period)
    ema_slow_last = ema_last(close, slow_period)
    eps_band = 0.001
    trend_bias: Optional[str] = None
    if ema_fast_last is not None and ema_slow_last is not None:
        if ema_fast_last > (ema_slow_last * (1.0 + eps_band)):
            trend_bias = "bullish"
        elif ema_fast_last < (ema_slow_last * (1.0 - eps_band)):
            trend_bias = "bearish"
        else:
            trend_bias = "neutral"
    out["trend_bias"] = trend_bias

    rsi_period = min(30, max(7, int(round(0.35 * bars_h))))
    rsi_last, rsi_series = rsi_wilder_last(close, rsi_period)
    momentum_state: Optional[str] = None
    if rsi_last is not None:
        if rsi_last >= 60.0:
            momentum_state = "bullish"
        elif rsi_last <= 40.0:
            momentum_state = "bearish"
        else:
            momentum_state = "neutral"

    bars_6h = max(2, int(round((6 * 60) / max(interval_minutes, 1))))
    if bars_h == bars_6h and rsi_last is not None and rsi_series is not None:
        lookback_bars = min(5, bars_h - 1)
        if lookback_bars >= 1:
            prev_idx = len(rsi_series) - (lookback_bars + 1)
            if 0 <= prev_idx < len(rsi_series):
                rsi_prev = rsi_series[prev_idx]
                if rsi_prev is not None:
                    rsi_delta_recent = float(rsi_last - rsi_prev)
                    if 40.0 <= rsi_last <= 55.0 and rsi_delta_recent > 3.0:
                        momentum_state = "repairing"
    out["momentum_state"] = momentum_state

    window = min(bars_h, n - 1)
    rv = rv_logret(close, window)
    vol_state: Optional[str] = None
    if rv is not None:
        if rv >= 0.012:
            vol_state = "high"
        elif rv <= 0.006:
            vol_state = "low"
        else:
            vol_state = "normal"
    out["vol_state"] = vol_state

    trend_component = {
        "bullish": 1.0,
        "neutral": 0.5,
        "bearish": 0.0,
    }.get(trend_bias)
    momentum_component = {
        "bullish": 1.0,
        "neutral": 0.5,
        "bearish": 0.0,
        "repairing": 0.6,
    }.get(momentum_state)
    vol_component = {
        "low": 0.8,
        "normal": 0.6,
        "high": 0.4,
    }.get(vol_state)
    if trend_component is not None and momentum_component is not None and vol_component is not None:
        composite = (0.45 * trend_component) + (0.45 * momentum_component) + (0.10 * vol_component)
        out["composite_strength"] = round(clamp01(float(composite)), 3)

    return out


def _build_multi_horizon(close: List[float], interval: Optional[str]) -> Dict[str, object]:
    interval_minutes = parse_interval_minutes(interval)
    bars_6h = max(2, int(round((6 * 60) / interval_minutes)))
    bars_2d = max(2, int(round((2 * 24 * 60) / interval_minutes)))
    bars_14d = max(2, int(round((14 * 24 * 60) / interval_minutes)))
    return {
        "windows": ["6h", "2d", "14d"],
        "horizon_signals": {
            "6h": horizon_signal(close, interval_minutes, bars_6h),
            "2d": horizon_signal(close, interval_minutes, bars_2d),
            "14d": horizon_signal(close, interval_minutes, bars_14d),
        },
    }


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA (EMA with alpha=1/period)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def _bars_per_day(ts: pd.Series) -> int:
    if ts is None or len(ts) < 2:
        return 24
    try:
        deltas = ts.diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 24
        seconds = float(deltas.median())
        if seconds <= 0:
            return 24
        bars = int(round(86400 / seconds))
        return max(1, min(1440, bars))
    except Exception:
        return 24


def _pivot_labels(highs: pd.Series, lows: pd.Series) -> Optional[str]:
    """Determine last market structure pivot label."""
    pivots: List[tuple] = []
    for i in range(1, len(highs) - 1):
        if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i + 1]:
            pivots.append(("H", highs.iloc[i]))
        if lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i + 1]:
            pivots.append(("L", lows.iloc[i]))

    if len(pivots) < 2:
        return None

    last_two = pivots[-2:]
    (type1, val1), (type2, val2) = last_two

    if type1 == "H" and type2 == "H":
        return "HH" if val2 > val1 else "LH"
    if type1 == "L" and type2 == "L":
        return "HL" if val2 > val1 else "LL"
    return None


def _paper_universe_symbols(current_symbol: str) -> List[str]:
    default_universe = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "SUI-USD"]
    symbols: List[str] = []
    try:
        settings = get_settings()
        for raw in list(settings.paper_symbols or []):
            sym = str(raw or "").strip()
            if sym and sym not in symbols:
                symbols.append(sym)
    except Exception:
        symbols = []

    if not symbols:
        symbols = list(default_universe)
    if current_symbol not in symbols:
        symbols.append(current_symbol)
    return symbols


def _compute_relative_strength_and_breadth(
    symbol: str,
    con,
    interval: str,
    bars_per_day: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    rs_block: Dict[str, object] = {
        "rs_vs_btc_7d": None,
        "rs_rank": None,
        "rs_percentile": None,
    }
    breadth_block: Dict[str, object] = {
        "pct_symbols_above_ema20": None,
        "pct_symbols_bullish_ema_cross": None,
        "breadth_score": None,
    }

    symbols = _paper_universe_symbols(symbol)
    if not symbols:
        return rs_block, breadth_block

    bars_7d = max(1, int(bars_per_day) * 7)
    lookback = max(50, bars_7d + 1)
    placeholders = ", ".join(["?"] * len(symbols))
    query = f"""
        SELECT symbol, ts, close
        FROM candles
        WHERE interval = ? AND symbol IN ({placeholders})
        ORDER BY symbol ASC, ts ASC
    """
    rows_df = con.execute(query, [interval, *symbols]).fetch_df()
    if rows_df is None or len(rows_df) == 0:
        return rs_block, breadth_block

    rows_df = rows_df.sort_values(["symbol", "ts"]).groupby("symbol", group_keys=False).tail(lookback).copy()
    if rows_df is None or len(rows_df) == 0:
        return rs_block, breadth_block

    ret_7d: Dict[str, float] = {}
    above_ema20: Dict[str, bool] = {}
    bullish_cross: Dict[str, bool] = {}

    for sym, group in rows_df.groupby("symbol"):
        close = group["close"].astype(float).reset_index(drop=True)
        if len(close) >= 20:
            ema20_last = close.ewm(span=20, adjust=False).mean().iloc[-1]
            last_close = close.iloc[-1]
            if pd.notna(ema20_last) and pd.notna(last_close):
                above_ema20[sym] = bool(float(last_close) > float(ema20_last))

        if len(close) >= 50:
            ema20_last = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50_last = close.ewm(span=50, adjust=False).mean().iloc[-1]
            if pd.notna(ema20_last) and pd.notna(ema50_last):
                bullish_cross[sym] = bool(float(ema20_last) > float(ema50_last))

        if len(close) > bars_7d:
            prev_close = float(close.iloc[-(bars_7d + 1)])
            if prev_close != 0.0:
                ret_7d[sym] = float((float(close.iloc[-1]) / prev_close) - 1.0)

    if above_ema20:
        breadth_block["pct_symbols_above_ema20"] = float(
            sum(1 for v in above_ema20.values() if v) / len(above_ema20)
        )
    if bullish_cross:
        breadth_block["pct_symbols_bullish_ema_cross"] = float(
            sum(1 for v in bullish_cross.values() if v) / len(bullish_cross)
        )

    breadth_parts = [
        v
        for v in [
            breadth_block["pct_symbols_above_ema20"],
            breadth_block["pct_symbols_bullish_ema_cross"],
        ]
        if isinstance(v, (int, float))
    ]
    if breadth_parts:
        breadth_block["breadth_score"] = float(sum(breadth_parts) / len(breadth_parts))

    if ret_7d:
        ranked = sorted(ret_7d.items(), key=lambda kv: kv[1], reverse=True)
        rank_map = {sym: i + 1 for i, (sym, _ret) in enumerate(ranked)}
        n = len(ranked)
        if symbol in rank_map:
            rank = int(rank_map[symbol])
            rs_block["rs_rank"] = rank
            rs_block["rs_percentile"] = 1.0 if n <= 1 else float((n - rank) / max(n - 1, 1))

        btc_ret = ret_7d.get("BTC-USD")
        sym_ret = ret_7d.get(symbol)
        if symbol == "BTC-USD" and btc_ret is not None:
            rs_block["rs_vs_btc_7d"] = 0.0
        elif sym_ret is not None and btc_ret is not None:
            rs_block["rs_vs_btc_7d"] = float(sym_ret - btc_ret)

    return rs_block, breadth_block


def compute_technical_features(symbol: str, con, interval: Optional[str] = None) -> Dict[str, object]:
    """
    Compute expanded technical indicators for a symbol.
    Returns a dict with nested blocks for EMA, trend, momentum,
    volatility, volume, and market structure. Also retains legacy fields
    from build_technicals for compatibility.
    """
    # Legacy/base metrics
    base = build_technicals(con, symbol)

    query_interval = interval or "1h"
    df = con.execute(
        """
        SELECT ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND interval = ?
        ORDER BY ts ASC
        """,
        [symbol, query_interval],
    ).fetch_df()

    close_all: List[float] = []
    if df is not None and len(df) > 0 and "close" in df:
        close_all = [float(v) for v in df["close"].astype(float).tolist()]
    multi_horizon = _build_multi_horizon(close_all, query_interval)
    interval_minutes = parse_interval_minutes(query_interval)
    bars_per_day_guess = max(1, int(round(1440 / max(interval_minutes, 1))))
    rs_block_guess, breadth_block_guess = _compute_relative_strength_and_breadth(
        symbol=symbol,
        con=con,
        interval=query_interval,
        bars_per_day=bars_per_day_guess,
    )

    if df is None or len(df) < 50:
        base["ema"] = {}
        base["trend"] = {}
        base["momentum"] = {}
        base["volatility"] = {}
        base["volume"] = {}
        base["relative_strength"] = rs_block_guess
        base["breadth"] = breadth_block_guess
        base["market_structure"] = {"last_pivot": None}
        base["multi_horizon"] = multi_horizon
        return base

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    bars_per_day = _bars_per_day(df["ts"])
    rs_block, breadth_block = _compute_relative_strength_and_breadth(
        symbol=symbol,
        con=con,
        interval=query_interval,
        bars_per_day=bars_per_day,
    )

    # EMA block
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    sma20 = close.rolling(window=20).mean().iloc[-1]
    last_close = close.iloc[-1]
    ema_block = {
        "ema20": float(ema20) if pd.notna(ema20) else None,
        "ema50": float(ema50) if pd.notna(ema50) else None,
        "ema200": float(ema200) if pd.notna(ema200) else None,
        "ema20_above_50": bool(ema20 > ema50) if pd.notna(ema20) and pd.notna(ema50) else None,
        "ema50_above_200": bool(ema50 > ema200) if pd.notna(ema50) and pd.notna(ema200) else None,
        "sma20": float(sma20) if pd.notna(sma20) else None,
        "price_above_20sma": bool(last_close > sma20) if pd.notna(last_close) and pd.notna(sma20) else None,
    }

    # Momentum block
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    momentum_block = {
        "roc_1d": None,
        "roc_7d": None,
        "roc_30d": None,
        "stoch_rsi_k": None,
        "stoch_rsi_d": None,
        "rsi14": base.get("rsi14"),
        "rsi14_delta_6": None,
        "rsi14_delta_24": None,
        "macd_line": float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else None,
        "macd_signal": float(signal.iloc[-1]) if pd.notna(signal.iloc[-1]) else None,
        "macd_hist": float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else None,
    }
    if len(close) > bars_per_day and close.iloc[-(bars_per_day + 1)] != 0:
        momentum_block["roc_1d"] = float((last_close / close.iloc[-(bars_per_day + 1)]) - 1)
    if len(close) > bars_per_day * 7 and close.iloc[-(bars_per_day * 7 + 1)] != 0:
        momentum_block["roc_7d"] = float((last_close / close.iloc[-(bars_per_day * 7 + 1)]) - 1)
    if len(close) > bars_per_day * 30 and close.iloc[-(bars_per_day * 30 + 1)] != 0:
        momentum_block["roc_30d"] = float((last_close / close.iloc[-(bars_per_day * 30 + 1)]) - 1)

    rsi_series = rsi(close, 14)
    if rsi_series.notna().sum() >= 14:
        rsi_window = rsi_series.tail(14)
        rsi_min = rsi_window.min()
        rsi_max = rsi_window.max()
        if rsi_max != rsi_min:
            k_series = (rsi_window - rsi_min) / (rsi_max - rsi_min) * 100.0
            momentum_block["stoch_rsi_k"] = float(k_series.iloc[-1])
            if len(k_series) >= 3:
                momentum_block["stoch_rsi_d"] = float(k_series.tail(3).mean())
            else:
                momentum_block["stoch_rsi_d"] = float(k_series.mean())
        rsi_last = rsi_series.iloc[-1]
        if len(rsi_series) >= 7 + 1 and pd.notna(rsi_last) and pd.notna(rsi_series.iloc[-7 - 1]):
            momentum_block["rsi14_delta_6"] = float(rsi_last - rsi_series.iloc[-7 - 1])
        if len(rsi_series) >= 25 and pd.notna(rsi_last) and pd.notna(rsi_series.iloc[-25]):
            momentum_block["rsi14_delta_24"] = float(rsi_last - rsi_series.iloc[-25])

    # Volatility block
    range_pct = None
    if last_close:
        range_pct = float((high.iloc[-1] - low.iloc[-1]) / last_close) if last_close != 0 else None
    if len(df) >= 168:
        recent_high = high.tail(168).max()
        recent_low = low.tail(168).min()
        range_7d_pct = None
        if last_close != 0 and pd.notna(recent_high) and pd.notna(recent_low):
            range_7d_pct = float((recent_high - recent_low) / last_close)
    else:
        range_7d_pct = None

    volatility_block = {
        "atr_pct": base.get("atr_pct"),
        "vol_regime": base.get("vol_regime"),
        "range_pct": range_pct,
        "range_7d_pct": range_7d_pct,
        "rv_24h": None,
        "rv_7d": None,
        "bb_mid_20": None,
        "bb_upper_20_2": None,
        "bb_lower_20_2": None,
        "bb_pctb_20_2": None,
        "bb_bandwidth_20_2": None,
        "kc_mid_20": None,
        "kc_upper_20_15": None,
        "kc_lower_20_15": None,
        "bb_kc_compression_ratio": None,
        "squeeze_on": None,
        "squeeze_level": "off",
        "squeeze_fired": None,
    }
    if len(close) > 1:
        log_ret = np.log(close / close.shift(1))
        if len(log_ret) >= bars_per_day:
            volatility_block["rv_24h"] = float(log_ret.tail(bars_per_day).std())
        if len(log_ret) >= bars_per_day * 7:
            volatility_block["rv_7d"] = float(log_ret.tail(bars_per_day * 7).std())

    if len(close) >= 20:
        mid_s = close.rolling(window=20).mean()
        std_s = close.rolling(window=20).std()
        mid = mid_s.iloc[-1]
        std = std_s.iloc[-1]
        upper = None
        lower = None
        if pd.notna(mid) and pd.notna(std):
            upper = mid + 2 * std
            lower = mid - 2 * std
            volatility_block["bb_mid_20"] = float(mid)
            volatility_block["bb_upper_20_2"] = float(upper)
            volatility_block["bb_lower_20_2"] = float(lower)
            denom = max(float(upper - lower), EPS)
            pctb = (last_close - lower) / denom if pd.notna(last_close) else None
            if pctb is not None:
                volatility_block["bb_pctb_20_2"] = float(max(0.0, min(1.0, pctb)))
            bw = (upper - lower) / max(abs(float(mid)), EPS)
            volatility_block["bb_bandwidth_20_2"] = float(bw)

        # Keltner Channels and squeeze state.
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr20_s = tr.rolling(window=20).mean()
        kc_mid_s = close.ewm(span=20, adjust=False).mean()
        kc_mid = kc_mid_s.iloc[-1]
        atr20 = atr20_s.iloc[-1]
        if pd.notna(kc_mid) and pd.notna(atr20):
            kc_upper = kc_mid + (1.5 * atr20)
            kc_lower = kc_mid - (1.5 * atr20)
            volatility_block["kc_mid_20"] = float(kc_mid)
            volatility_block["kc_upper_20_15"] = float(kc_upper)
            volatility_block["kc_lower_20_15"] = float(kc_lower)

            if upper is not None and lower is not None:
                bb_width = max(float(upper - lower), EPS)
                kc_width = max(float(kc_upper - kc_lower), EPS)
                compression = float(bb_width / kc_width)
                volatility_block["bb_kc_compression_ratio"] = compression
                squeeze_now = bool((float(upper) < float(kc_upper)) and (float(lower) > float(kc_lower)))
                volatility_block["squeeze_on"] = squeeze_now

                if squeeze_now and compression <= 0.80:
                    volatility_block["squeeze_level"] = "high"
                elif squeeze_now and compression <= 1.00:
                    volatility_block["squeeze_level"] = "medium"
                else:
                    volatility_block["squeeze_level"] = "off"

                if len(close) >= 21:
                    upper_prev = (mid_s + 2 * std_s).iloc[-2]
                    lower_prev = (mid_s - 2 * std_s).iloc[-2]
                    kc_upper_prev = (kc_mid_s + (1.5 * atr20_s)).iloc[-2]
                    kc_lower_prev = (kc_mid_s - (1.5 * atr20_s)).iloc[-2]
                    if (
                        pd.notna(upper_prev)
                        and pd.notna(lower_prev)
                        and pd.notna(kc_upper_prev)
                        and pd.notna(kc_lower_prev)
                    ):
                        squeeze_prev = bool(
                            (float(upper_prev) < float(kc_upper_prev))
                            and (float(lower_prev) > float(kc_lower_prev))
                        )
                        volatility_block["squeeze_fired"] = bool(squeeze_prev and not squeeze_now)

    # Volume block
    vol_5d_avg = float(volume.tail(120).mean()) if len(volume) >= 5 * 24 else None
    vol_20d_avg = float(volume.tail(480).mean()) if len(volume) >= 20 * 24 else None
    volume_spike = None
    if vol_20d_avg not in (None, 0) and pd.notna(volume.iloc[-1]):
        volume_spike = bool(volume.iloc[-1] > 2 * vol_20d_avg)

    volume_block = {
        "vol_z": base.get("vol_z"),
        "vol_5d_avg": vol_5d_avg,
        "vol_20d_avg": vol_20d_avg,
        "volume_spike": volume_spike,
        "cmf_20": None,
        "cmf_20_label": None,
        "vwap_24h": None,
        "vwap_7d": None,
        "price_above_vwap": None,
        "vwap_distance_pct": None,
    }
    if len(df) >= 20:
        high20 = high.tail(20)
        low20 = low.tail(20)
        close20 = close.tail(20)
        vol20 = volume.tail(20)
        mfm = ((close20 - low20) - (high20 - close20)) / (high20 - low20).replace(0, np.nan)
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mfv = mfm * vol20
        denom = max(float(vol20.sum()), EPS)
        cmf = float(mfv.sum() / denom)
        volume_block["cmf_20"] = cmf
        if cmf > 0.05:
            volume_block["cmf_20_label"] = "accumulation"
        elif cmf < -0.05:
            volume_block["cmf_20_label"] = "distribution"
        else:
            volume_block["cmf_20_label"] = "neutral"

    if len(volume) >= bars_per_day:
        vol_24h = volume.tail(bars_per_day)
        close_24h = close.tail(bars_per_day)
        denom_24h = float(vol_24h.sum())
        if denom_24h > EPS:
            vwap_24h = float((close_24h * vol_24h).sum() / denom_24h)
            volume_block["vwap_24h"] = vwap_24h
            if vwap_24h != 0:
                volume_block["price_above_vwap"] = bool(float(last_close) > vwap_24h)
                volume_block["vwap_distance_pct"] = float((float(last_close) - vwap_24h) / vwap_24h)

    if len(volume) >= bars_per_day * 7:
        vol_7d = volume.tail(bars_per_day * 7)
        close_7d = close.tail(bars_per_day * 7)
        denom_7d = float(vol_7d.sum())
        if denom_7d > EPS:
            volume_block["vwap_7d"] = float((close_7d * vol_7d).sum() / denom_7d)

    # ADX/DMI(14)
    adx14 = None
    plus_di14 = None
    minus_di14 = None
    if len(df) >= 15:
        up_move = high.diff()
        down_move = low.diff() * -1
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        tr_rma = _rma(tr, 14)
        plus_rma = _rma(pd.Series(plus_dm, index=tr.index), 14)
        minus_rma = _rma(pd.Series(minus_dm, index=tr.index), 14)
        plus_di = 100 * (plus_rma / tr_rma.replace(0, np.nan))
        minus_di = 100 * (minus_rma / tr_rma.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = _rma(dx, 14)
        if pd.notna(adx.iloc[-1]):
            adx14 = float(adx.iloc[-1])
        if pd.notna(plus_di.iloc[-1]):
            plus_di14 = float(plus_di.iloc[-1])
        if pd.notna(minus_di.iloc[-1]):
            minus_di14 = float(minus_di.iloc[-1])

    # EMA slopes
    ema20_slope = None
    ema50_slope = None
    if len(close) >= 2:
        ema20_series = close.ewm(span=20, adjust=False).mean()
        ema50_series = close.ewm(span=50, adjust=False).mean()
        if pd.notna(ema20_series.iloc[-2]):
            ema20_slope = float(
                (ema20_series.iloc[-1] - ema20_series.iloc[-2])
                / max(abs(float(ema20_series.iloc[-2])), EPS)
            )
        if pd.notna(ema50_series.iloc[-2]):
            ema50_slope = float(
                (ema50_series.iloc[-1] - ema50_series.iloc[-2])
                / max(abs(float(ema50_series.iloc[-2])), EPS)
            )

    # Donchian(20)
    donchian_high = None
    donchian_low = None
    donchian_pos = None
    if len(df) >= 20:
        donchian_high = float(high.tail(20).max())
        donchian_low = float(low.tail(20).min())
        denom = max(donchian_high - donchian_low, EPS)
        pos = (last_close - donchian_low) / denom
        donchian_pos = float(max(0.0, min(1.0, pos)))

    # Trend block (ordered)
    trend_block = {
        "ema_cross": base.get("ema_cross"),
        "trend_strength": base.get("trend_strength"),
        "adx14": adx14,
        "plus_di14": plus_di14,
        "minus_di14": minus_di14,
        "ema20_slope_pct_per_bar": ema20_slope,
        "ema50_slope_pct_per_bar": ema50_slope,
        "donchian_high_20": donchian_high,
        "donchian_low_20": donchian_low,
        "donchian_pos_20": donchian_pos,
    }

    # Attach nested blocks
    base["ema"] = ema_block
    base["trend"] = trend_block
    base["momentum"] = momentum_block
    base["volatility"] = volatility_block
    base["volume"] = volume_block
    base["relative_strength"] = rs_block
    base["breadth"] = breadth_block
    base["multi_horizon"] = multi_horizon

    return base
