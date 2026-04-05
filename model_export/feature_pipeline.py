
# =============================================================
# FEATURE ENGINEERING PIPELINE (hourly bars → model features)
# =============================================================
# Input: Polars DataFrame with columns:
#   timestamp, open, high, low, close, volume,
#   tx_count, active_senders, active_receivers,
#   total_eth_transferred, total_gas_used
#
# All features are computed from 1h OHLCV + on-chain data.
# No forward-looking information is used.

import polars as pl
import numpy as np

def engineer_hourly_features(df_h: pl.DataFrame) -> pl.DataFrame:
    c = df_h["close"]
    o = df_h["open"]
    h = df_h["high"]
    l = df_h["low"]
    v = df_h["volume"]

    # --- Candle features ---
    body = (c - o) / (o + 1e-10)
    upper_wick = (h - pl.max_horizontal(o, c)) / (h - l + 1e-10)
    lower_wick = (pl.min_horizontal(o, c) - l) / (h - l + 1e-10)
    range_pct = (h - l) / (l + 1e-10)

    # --- Multi-timeframe returns ---
    returns = {}
    for w in [1, 2, 4, 8, 12, 24, 48, 168]:
        returns[f"ret_{w}h"] = c / c.shift(w) - 1

    # --- SMAs ---
    smas = {}
    for w in [4, 12, 24, 48, 168]:
        smas[f"sma_{w}h"] = c.rolling_mean(window_size=w)

    # --- RSI at multiple horizons ---
    def compute_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
        loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=period)
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    rsis = {}
    for p in [6, 14, 24]:
        rsis[f"rsi_{p}h"] = compute_rsi(c, p)

    # --- MACD ---
    ema12 = c.ewm_mean(span=12)
    ema26 = c.ewm_mean(span=26)
    macd = ema12 - ema26
    macd_signal = macd.ewm_mean(span=9)

    # --- Bollinger ---
    bb_mid = c.rolling_mean(window_size=24)
    bb_std = c.rolling_std(window_size=24)
    bb_zscore = (c - bb_mid) / (bb_std + 1e-10)
    bb_width = (bb_std * 2) / (bb_mid + 1e-10)

    # --- Volume features ---
    vol_sma = v.rolling_mean(window_size=24)
    vol_ratio = v / (vol_sma + 1e-10)
    obv = (v * c.diff().sign()).cum_sum()
    obv_sma = obv.rolling_mean(window_size=24)

    # --- Volatility ---
    vol_avgs = {}
    hr_ret = c / c.shift(1) - 1
    for w in [4, 12, 24]:
        vol_avgs[f"vol_avg_{w}h"] = hr_ret.rolling_std(window_size=w)

    # --- Range features ---
    range_avgs = {}
    for w in [12, 24]:
        range_avgs[f"range_avg_{w}h"] = range_pct.rolling_mean(window_size=w)

    # --- Lag features ---
    lags = {}
    for lag in [1, 2, 3]:
        lags[f"ret_lag_{lag}"] = returns["ret_1h"].shift(lag)
        lags[f"range_lag_{lag}"] = range_pct.shift(lag)

    # --- On-chain momentum ---
    onchain = {}
    for col_name in ["tx_count", "active_senders", "active_receivers",
                      "total_eth_transferred", "total_gas_used"]:
        if col_name in df_h.columns:
            s = df_h[col_name]
            onchain[f"{col_name}_change_24h"] = s / s.shift(24) - 1

    # --- Cyclical time ---
    hour = df_h["timestamp"].dt.hour()
    dow = df_h["timestamp"].dt.weekday()

    # Assemble all features into the DataFrame
    df_feat = df_h.with_columns([
        body.alias("body"), upper_wick.alias("upper_wick"),
        lower_wick.alias("lower_wick"), range_pct.alias("range_pct"),
        *[v.alias(k) for k, v in returns.items()],
        *[v.alias(k) for k, v in smas.items()],
        *[v.alias(k) for k, v in rsis.items()],
        macd.alias("macd"), macd_signal.alias("macd_signal"),
        (macd - macd_signal).alias("macd_hist"),
        bb_zscore.alias("bb_zscore"), bb_width.alias("bb_width_24h"),
        vol_ratio.alias("vol_ratio_24h"), obv.alias("obv"),
        (obv - obv_sma).alias("obv_diff"),
        *[v.alias(k) for k, v in vol_avgs.items()],
        *[v.alias(k) for k, v in range_avgs.items()],
        *[v.alias(k) for k, v in lags.items()],
        *[v.alias(k) for k, v in onchain.items()],
        (hour * 2 * np.pi / 24).sin().alias("hour_sin"),
        (hour * 2 * np.pi / 24).cos().alias("hour_cos"),
        (dow * 2 * np.pi / 7).sin().alias("dow_sin"),
        (dow * 2 * np.pi / 7).cos().alias("dow_cos"),
    ])

    return df_feat
