import os
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

# api key
def load_api_keys():
    load_dotenv()
    tiingo_key = os.getenv("Tiingo_API")
    if not tiingo_key:
        raise ValueError("Tiingo API key not found.")
    return tiingo_key

# get data for a specific ticker
def fetch_tiingo_data(ticker, api_key, start_date, end_date):
    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        f"?startDate={start_date}&endDate={end_date}&token={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            print(f"No data returned for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.sort_values("date", inplace=True)
        df["ticker"] = ticker
        cols = [c for c in ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "ticker"] if c in df.columns]
        df = df[cols]
        print(f"Downloaded {len(df)} rows for {ticker}.")
        return df
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error for {ticker}: {err}")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
    return None

def ensure_data_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    tiingo_key = load_api_keys()
    stock_tickers = ["AAPL", "NVDA", "MSFT", "TSLA"]
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("Getting stock data...")
    results = []
    for ticker in stock_tickers:
        df = fetch_tiingo_data(ticker, tiingo_key, start_date, end_date)
        if df is not None and not df.empty:
            results.append(df)
        else:
            print(f"No data returned for {ticker}")

    if not results:
        print("No stock data downloaded")
        return

    combined = pd.concat(results, ignore_index=True)

    savedir = '../data/raw'
    ensure_data_dir(savedir)
    # save each file
    for ticker, group in combined.groupby("ticker"):
        out_path = os.path.join(savedir, f"{ticker}.csv")
        group.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(group)} rows).")

    # also a combined file just in case
    combined_path = os.path.join(savedir, "stock_data_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined dataset to {combined_path} ({len(combined)} rows).")

if __name__ == "__main__":
    main()
