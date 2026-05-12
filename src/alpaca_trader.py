import os
import time
import requests
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_signals import (
    fetch_news,
    load_finbert,
    get_sentiment_signals,
    load_model_and_artifacts,
    get_lstm_signals,
    fuse_and_print,
    PROJECT_ROOT,   
)

# load env
load_dotenv()

# API keys 
SERPAPI_KEY = os.environ.get("SERP_API")
TIINGO_TOKEN = os.environ.get("Tiingo_API") 
ALPACA_API_KEY = os.environ.get("Alpaca_Key")
ALPACA_SECRET_KEY = os.environ.get("Alpaca_Secret")

ALLOCATION_PCT = 0.25  
SETTLE_WAIT_SECONDS = 10  

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading 

def fuse_signals(lstm_signals: dict, sentiment_signals: dict) -> list[str]:
    buy_list = []
    for ticker in ["AAPL", "TSLA", "NVDA", "MSFT"]:
        lstm_pred  = lstm_signals.get(ticker, 0)
        sentiment  = sentiment_signals.get(ticker, "NEUTRAL")
        if lstm_pred == 1 and sentiment in ("POSITIVE", "NEUTRAL"):
            buy_list.append(ticker)
    return buy_list


class AlpacaTrader:
    def __init__(self, api_key: str, secret_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID"    : api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type"       : "application/json",
        })

    def get_account(self) -> dict:
        resp = self.session.get(f"{ALPACA_BASE_URL}/v2/account", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_portfolio_value(self) -> float:
        return float(self.get_account()["portfolio_value"])

    def get_cash(self) -> float:
        return float(self.get_account()["cash"])

    def get_open_positions(self) -> list:
        resp = self.session.get(f"{ALPACA_BASE_URL}/v2/positions", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def close_all_positions(self):
        positions = self.get_open_positions()

        if not positions:
            print("  No open positions to close.")
            return

        print(f"  Closing {len(positions)} position(s):")
        for p in positions:
            print(f"    {p['symbol']}: {p['qty']} shares "
                  f"(current value: ${float(p['market_value']):,.2f})")

        resp = self.session.delete(
            f"{ALPACA_BASE_URL}/v2/positions",
            params={"cancel_orders": "true"},
            timeout=15,
        )
        resp.raise_for_status()
        print("All positions closed.")

    def place_buy_order(self, ticker: str, notional: float) -> dict | None:
        if notional < 1:
            print(f"  [{ticker}] Amount ${notional:.2f} too small — skipping.")
            return None

        payload = {
            "symbol"       : ticker,
            "notional"     : round(notional, 2),
            "side"         : "buy",
            "type"         : "market",
            "time_in_force": "day",
        }

        print(f"  [{ticker}] Placing BUY: ${notional:,.2f} ...", end=" ")
        resp = self.session.post(
            f"{ALPACA_BASE_URL}/v2/orders",
            json=payload,
            timeout=15,
        )
        if resp.status_code != 201:
            print(f"Failed: {resp.status_code} - {resp.text}")
            return None
        resp.raise_for_status()
        order = resp.json()
        print(f"Order ID: {order['id']}")
        return order

def run(serpapi_key: str, tiingo_token: str, finbert_dir: str = None, dry_run: bool = False):
    print("Updating price data ...")
    from src.update_data import main as update_main
    update_main()

    articles        = fetch_news(serpapi_key)
    tokenizer, fb   = load_finbert(finbert_dir)
    sentiment       = get_sentiment_signals(articles, tokenizer, fb)
    lstm, scaler, features = load_model_and_artifacts()
    lstm_signals    = get_lstm_signals(lstm, scaler, features)
    buy_list        = fuse_signals(lstm_signals, sentiment)

    # dry run option, doesn't place orders
    if dry_run:
        print("\n[DRY RUN] Signals generated. No orders will be placed.")
        print(f"Would have bought: {buy_list if buy_list else 'nothing'}")
        return

    api_key    = ALPACA_API_KEY
    secret_key = ALPACA_SECRET_KEY

    if not api_key or not secret_key:
        raise ValueError(
            "\nMissing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables."
        )

    trader = AlpacaTrader(api_key, secret_key)

    account = trader.get_account()
    account_status = account.get("status")
    if account_status != "ACTIVE":
        raise ValueError(f"Alpaca account status is '{account_status}'. Must be 'ACTIVE' to trade.")

    print("\nClosing existing positions ...")
    trader.close_all_positions()

    print(f"\nWaiting {SETTLE_WAIT_SECONDS}s for cash to settle ...")
    time.sleep(SETTLE_WAIT_SECONDS)

    print("\nPlacing new orders ...")

    if not buy_list:
        print("  No BUY signals this week — holding cash.")
        return

    portfolio_value  = trader.get_portfolio_value()
    cash_available   = trader.get_cash()
    amount_per_stock = portfolio_value * ALLOCATION_PCT

    print(f"\n  Portfolio value : ${portfolio_value:,.2f}")
    print(f"  Cash available  : ${cash_available:,.2f}")
    print(f"  Per position    : ${amount_per_stock:,.2f}  ({ALLOCATION_PCT:.0%} of portfolio)")
    print(f"  Buying          : {buy_list}\n")

    orders_placed = []
    for ticker in buy_list:
        notional = min(amount_per_stock, cash_available)

        order = trader.place_buy_order(ticker, notional)
        if order:
            orders_placed.append(ticker)
            cash_available -= notional 

    print("\n" + "=" * 55)
    print("  WEEKLY RUN COMPLETE")
    print("=" * 55)
    print(f"  Signals generated : {buy_list}")
    print(f"  Orders placed     : {orders_placed}")
    skipped = [t for t in buy_list if t not in orders_placed]
    if skipped:
        print(f"  Skipped           : {skipped}  (insufficient cash)")
    print("=" * 55)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Alpaca trading workflow.")
    parser.add_argument("--dry-run", action="store_true", help="Run the workflow without placing orders")
    args = parser.parse_args()

    serpapi_key = args.serpapi_key or SERPAPI_KEY
    tiingo_token = args.tiingo_token or TIINGO_TOKEN

    if not serpapi_key:
        raise SystemExit("SERP_API environment variable not set or --serpapi-key not provided.")
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables not set.")

    run(
        serpapi_key  = serpapi_key,
        tiingo_token = tiingo_token,
        finbert_dir  = args.finbert_dir,
        dry_run      = args.dry_run,
    )
