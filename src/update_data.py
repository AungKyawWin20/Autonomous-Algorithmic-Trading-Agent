import os
import requests
import pandas as pd
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

BASE_URL = "https://api.tiingo.com/tiingo/daily"
RAW_PATH = Path(__file__).parent.parent / "data" / "raw"
STOCKS = ["AAPL", "NVDA", "MSFT", "TSLA"]


def get_api_key():
    load_dotenv()
    return os.getenv("Tiingo_API")


def fetch_prices(ticker, start_date, token):
    url = f"{BASE_URL}/{ticker}/prices"
    params = {"startDate": start_date, "resampleFreq": "daily"}
    resp = requests.get(url, headers={"Authorization": f"Token {token}"}, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.sort_values("date", inplace=True)
    return df[["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]]


def update_csv(path, token):
    ticker = path.stem
    existing = pd.read_csv(path, parse_dates=["date"])
    existing.sort_values("date", inplace=True)
    if "ticker" not in existing.columns:
        existing["ticker"] = ticker
    last_date = existing["date"].max()
    if pd.isna(last_date):
        print(f"[{ticker}] no date, skip")
        return
    next_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[{ticker}] last {last_date.date()}, fetch from {next_date}")
    new_data = fetch_prices(ticker, next_date, token)
    if new_data.empty:
        print(f"[{ticker}] up to date")
        return
    new_data["ticker"] = ticker
    combined = pd.concat([existing, new_data], ignore_index=True, sort=False)
    combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
    combined.sort_values("date", inplace=True)
    combined.to_csv(path, index=False)
    added = len(combined) - len(existing)
    print(f"[{ticker}] added {added}, total {len(combined)}")


def main():
    token = get_api_key()
    if not token:
        raise SystemExit("Tiingo API key not found")
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    print(f"RAW_PATH: {RAW_PATH}")
    print("Files in RAW_PATH:", list(RAW_PATH.glob("*.csv")))
    for ticker in STOCKS:
        path = RAW_PATH / f"{ticker}.csv"
        if not path.exists():
            print(f"[{ticker}] file missing, skip")
            continue
        try:
            update_csv(path, token)
            from src.preprocess_ohlcv import main as preprocess_main
            preprocess_main()
        except Exception as e:
            print(f"Error {ticker}: {e}")
    print("Done")


if __name__ == "__main__":
    main()
