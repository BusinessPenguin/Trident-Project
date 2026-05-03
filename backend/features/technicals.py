import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using a classic Wilder-style EMA approach.
    Returns a pandas Series of RSI values in [0, 100].
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_technicals(con, symbol: str) -> dict:
    """
    Build technical features for a symbol using 1h candles from DuckDB.

    Uses table:
        candles(symbol, interval, ts, open, high, low, close, volume)

    Returns a dict with keys:
        ema20, ema50, ema_cross, trend_strength, rsi14,
        atr_pct, vol_regime, vol_z,
        ret_1d, ret_3d, ret_7d,
        dist_from_local_high, dist_from_local_low

    Values may be None if there is insufficient data.
    """
    df = con.execute(
        """
        SELECT ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND interval = '1h'
        ORDER BY ts ASC
        """,
        [symbol],
    ).fetch_df()

    # Initialize default result with all keys present
    result = {
        "last_close": None,
        "ema20": None,
        "ema50": None,
        "ema_cross": None,
        "trend_strength": None,
        "rsi14": None,
        "atr_pct": None,
        "vol_regime": None,
        "vol_z": None,
        "ret_1d": None,
        "ret_3d": None,
        "ret_7d": None,
        "dist_from_local_high": None,
        "dist_from_local_low": None,
    }

    if df is None or len(df) < 20:
        # Not enough history to compute most metrics
        return result

    df = df.copy()

    # Latest close
    last_close = float(df["close"].iloc[-1])
    result["last_close"] = last_close

    # EMA20, EMA50
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    ema20 = float(df["ema20"].iloc[-1])
    ema50 = float(df["ema50"].iloc[-1])

    result["ema20"] = ema20
    result["ema50"] = ema50

    # EMA cross + trend strength
    if last_close != 0:
        if abs(ema20 - ema50) / last_close < 0.002:
            ema_cross = "neutral"
        elif ema20 > ema50:
            ema_cross = "bullish"
        else:
            ema_cross = "bearish"
        trend_strength = abs(ema20 - ema50) / last_close
    else:
        ema_cross = None
        trend_strength = None

    result["ema_cross"] = ema_cross
    result["trend_strength"] = trend_strength

    # RSI14
    df["rsi14"] = rsi(df["close"], 14)
    result["rsi14"] = float(df["rsi14"].iloc[-1])

    # ATR14 and ATR%
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    atr14 = float(df["atr14"].iloc[-1])
    atr_pct = atr14 / last_close if last_close else None
    result["atr_pct"] = atr_pct

    # Volume z-score over last 20 bars
    vol_z = None
    if len(df) >= 20:
        tail = df["volume"].tail(20)
        mean = float(tail.mean())
        std = float(tail.std())
        if std not in (0.0, None):
            vol_z = float((tail.iloc[-1] - mean) / std)
    result["vol_z"] = vol_z

    # Helper to compute returns over N bars
    def safe_return(df_local, bars: int):
        if len(df_local) <= bars:
            return None
        prev_price = float(df_local["close"].iloc[-(bars + 1)])
        if prev_price == 0:
            return None
        return (last_close - prev_price) / prev_price

    result["ret_1d"] = safe_return(df, 24)
    result["ret_3d"] = safe_return(df, 72)
    result["ret_7d"] = safe_return(df, 168)

    # Volatility regime based on ATR% quantiles over history
    vol_regime = None
    atr_pct_series = df["atr14"] / df["close"]
    if atr_pct_series.notna().sum() >= 60:
        current = float(atr_pct_series.iloc[-1])
        q20, q80, q95 = atr_pct_series.quantile([0.2, 0.8, 0.95])
        if current <= q20:
            vol_regime = "low"
        elif current <= q80:
            vol_regime = "normal"
        elif current <= q95:
            vol_regime = "high"
        else:
            vol_regime = "extreme"
    result["vol_regime"] = vol_regime

    # Distance to local 7-day high/low (last 168 hours)
    if len(df) >= 168:
        window = df.tail(168)
        local_high = float(window["close"].max())
        local_low = float(window["close"].min())

        if local_high != 0:
            result["dist_from_local_high"] = (last_close - local_high) / local_high
        if local_low != 0:
            result["dist_from_local_low"] = (last_close - local_low) / local_low

    return result
