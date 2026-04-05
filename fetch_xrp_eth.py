import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# ========================= CONFIG =========================
PRODUCT_XRP = "XRP-USD"
PRODUCT_ETH = "ETH-USD"
GRANULARITY = 60  # 1 minute (do not change for 1-min candles)

# How much history do you want? (adjust as needed)
DAYS_TO_FETCH = 90  # e.g. 30 days → ~43,200 candles per pair
# ======================================================

def fetch_candles(product_id: str, start_unix: int, end_unix: int):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "granularity": GRANULARITY,
        "start": start_unix,
        "end": end_unix
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"Error {resp.status_code} for {product_id}: {resp.text}")
        return pd.DataFrame()
    
    data = resp.json()
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

def get_full_history(product_id: str):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    print(f"Fetching {DAYS_TO_FETCH} days of 1-min data for {product_id}...")
    all_dfs = []
    current_end = int(end_date.timestamp())
    
    while current_end > int(start_date.timestamp()):
        current_start = max(current_end - (300 * GRANULARITY), int(start_date.timestamp()))
        df_chunk = fetch_candles(product_id, current_start, current_end)
        
        if df_chunk.empty:
            print("  No more data returned.")
            break
        
        all_dfs.append(df_chunk)
        current_end = current_start  # move backward
        
        time.sleep(0.2)  # polite rate-limit (Coinbase allows ~10 req/s)
    
    if not all_dfs:
        raise ValueError(f"No data fetched for {product_id}")
    
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"   → {len(df):,} candles fetched for {product_id}")
    return df

# ============== FETCH BOTH PAIRS ==============
df_xrp = get_full_history(PRODUCT_XRP)
df_eth = get_full_history(PRODUCT_ETH)

# ============== MERGE & COMPUTE XRP-ETH ==============
df = pd.merge(df_xrp[["timestamp", "open", "high", "low", "close", "volume"]],
              df_eth[["timestamp", "open", "high", "low", "close", "volume"]],
              on="timestamp", suffixes=("_xrp_usd", "_eth_usd"), how="inner")

df["close_xrp_eth"] = df["close_xrp_usd"] / df["close_eth_usd"]
df["open_xrp_eth"]  = df["open_xrp_usd"]  / df["open_eth_usd"]
df["high_xrp_eth"]  = df["high_xrp_usd"]  / df["high_eth_usd"]
df["low_xrp_eth"]   = df["low_xrp_usd"]   / df["low_eth_usd"]

# Keep useful columns
df = df[["timestamp", 
         "open_xrp_eth", "high_xrp_eth", "low_xrp_eth", "close_xrp_eth",
         "volume_xrp_usd", "volume_eth_usd"]]

df.to_csv("xrp_eth_1min_data.csv", index=False)
print(f"\n✅ Done! Saved {len(df):,} rows to xrp_eth_1min_data.csv")
print(df.head())
