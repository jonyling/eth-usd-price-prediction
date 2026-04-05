#!/usr/bin/env python3
"""
ETH-USD 4h Selective Trading — Signal Generator
================================================
Loads the trained LGBM model and generates trade signals
from new hourly OHLCV + on-chain data.

Usage:
    python predict.py --input latest_hourly.csv --output signals.csv
"""
import argparse, json, os
import numpy as np
import polars as pl
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    model_path = os.path.join(SCRIPT_DIR, "lgbm_eth_4h_regressor.txt")
    booster = lgb.Booster(model_file=model_path)
    return booster

def load_config():
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(config_path) as f:
        return json.load(f)

def compute_volatility_filter(close_series, percentile=50):
    """24h rolling std of hourly returns."""
    hr_ret = close_series / close_series.shift(1) - 1
    vol_24h = hr_ret.rolling_std(window_size=24)
    vol_thresh = vol_24h.drop_nulls().quantile(percentile / 100)
    return vol_24h, vol_thresh

def generate_signals(df_h: pl.DataFrame, config: dict):
    """Generate trading signals from hourly data."""
    from feature_pipeline import engineer_hourly_features

    # Engineer features
    df_feat = engineer_hourly_features(df_h)

    features = config["features"]
    K = config["trading"]["entry_K"]
    cost = config["trading"]["cost_per_trade"]
    threshold = K * cost

    # Volatility filter
    vol_24h, vol_thresh = compute_volatility_filter(
        df_feat["close"], config["trading"]["vol_percentile"])

    # Get latest non-overlapping 4h bars
    df_clean = df_feat.drop_nulls().gather_every(4)

    X = df_clean.select(features).to_numpy()
    timestamps = df_clean["timestamp"].to_list()
    closes = df_clean["close"].to_list()

    # Load model and predict
    booster = load_model()
    predictions = booster.predict(X)

    # Generate signals
    vol_vals = vol_24h.to_numpy()[-len(predictions):]

    signals = []
    for i in range(len(predictions)):
        pred = predictions[i]
        high_vol = vol_vals[i] > vol_thresh if not np.isnan(vol_vals[i]) else False

        if high_vol and pred > threshold:
            action = "LONG"
        elif high_vol and pred < -threshold:
            action = "SHORT"
        else:
            action = "FLAT"

        signals.append({
            "timestamp": timestamps[i],
            "close": closes[i],
            "predicted_4h_return": round(float(pred), 6),
            "threshold": round(threshold, 6),
            "high_vol": high_vol,
            "signal": action,
            "confidence": round(abs(float(pred)) / threshold, 2) if threshold > 0 else 0,
        })

    return pl.DataFrame(signals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to hourly OHLCV CSV")
    parser.add_argument("--output", default="signals.csv", help="Output signals CSV")
    args = parser.parse_args()

    config = load_config()
    df_h = pl.read_csv(args.input, try_parse_dates=True)
    signals = generate_signals(df_h, config)

    signals.write_csv(args.output)
    print(f"Signals written to {args.output}")

    # Show latest signals
    latest = signals.tail(10)
    print("\nLatest signals:")
    print(latest)

    active = signals.filter(pl.col("signal") != "FLAT")
    print(f"\nActive signals: {len(active)} / {len(signals)} "
          f"({len(active)/len(signals)*100:.1f}%)")
