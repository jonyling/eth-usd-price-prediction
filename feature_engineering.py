import pandas as pd
import numpy as np

df = pd.read_csv("xrp_eth_1min_data.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp").sort_index()

price = df["close_xrp_eth"]

# 1. Basic returns & lags (most important for time-series models)
df["return"] = np.log(price / price.shift(1))
for lag in [1, 5, 15, 30, 60]:
    df[f"return_lag_{lag}"] = df["return"].shift(lag)
    df[f"price_lag_{lag}"]   = price.shift(lag)

# 2. Rolling statistics
for window in [5, 10, 20, 60, 180]:  # 5min, 10min, 1h, 3h
    df[f"rolling_mean_{window}"] = price.rolling(window).mean()
    df[f"rolling_std_{window}"]  = price.rolling(window).std()
    df[f"rolling_vol_{window}"]  = df["return"].rolling(window).std()   # realized volatility

# 3. Technical indicators (manual implementations)
# RSI (14-period)
delta = price.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# EMA (fast/slow)
df["ema_12"] = price.ewm(span=12, adjust=False).mean()
df["ema_26"] = price.ewm(span=26, adjust=False).mean()
df["macd"]   = df["ema_12"] - df["ema_26"]
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

# Bollinger Bands
df["bb_middle"] = price.rolling(20).mean()
df["bb_std"]    = price.rolling(20).std()
df["bb_upper"]  = df["bb_middle"] + 2 * df["bb_std"]
df["bb_lower"]  = df["bb_middle"] - 2 * df["bb_std"]

# 4. External / cross-market features (very powerful for XRP-ETH)
df["xrp_usd_volume"] = df["volume_xrp_usd"]   # liquidity in XRP market
df["eth_usd_volume"] = df["volume_eth_usd"]
df["xrp_usd_return"] = np.log(df["close_xrp_usd"] / df["close_xrp_usd"].shift(1))  # add the raw USD series too if you kept them
df["eth_usd_return"] = np.log(df["close_eth_usd"] / df["close_eth_usd"].shift(1))

# 5. Time features (crypto has intraday patterns)
df.index = pd.to_datetime(df.index)
df["hour"] = df.index.hour # type: ignore
df["dow"]  = df.index.dayofweek # type: ignore

# 6. Target variable examples
df["target_next_close"] = price.shift(-1)                    # regression
df["target_direction"]  = (df["target_next_close"] > price).astype(int)  # binary classification (up/down)

df = df.dropna()  # important before training
df.to_csv("xrp_eth_features.csv", index=True)
print("Feature engineering complete → xrp_eth_features.csv")