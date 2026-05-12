import sys
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_technical_features
from src.backtest_strategy import load_model_and_artifacts

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]

NAME_TO_SYMBOL = {
    "Apple"    : "AAPL",
    "Tesla"    : "TSLA",
    "Nvidia"   : "NVDA",
    "Microsoft": "MSFT",
}

KEYWORDS             = "earnings OR forecast OR 'stock price' OR acquisition OR lawsuit"
ARTICLES_PER_TICKER  = 20
CONFIDENCE_THRESHOLD = 0.40   
LSTM_THRESHOLD       = 0.5
LOOKBACK             = 60
DATA_PATH            = PROJECT_ROOT / "data" / "processed"

def fetch_news(serpapi_key: str) -> dict[str, list[dict]]:
    from serpapi import GoogleSearch

    print("Fetching news ...")
    articles_by_name = {}

    for name in NAME_TO_SYMBOL:
        params = {
            "engine" : "google_news",
            "q"      : f"{name} {KEYWORDS}",
            "hl"     : "en",
            "gl"     : "us",
            "api_key": serpapi_key,
        }
        results = GoogleSearch(params).get_dict()
        articles = results.get("news_results", [])[:ARTICLES_PER_TICKER]
        articles_by_name[name] = articles
        print(f"  {name}: {len(articles)} articles fetched")

    return articles_by_name

def load_finbert(model_dir: str = None):
    source = "ProsusAI/finbert"
    print(f"Loading FinBERT from: {source} ...")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model     = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()
    print("FinBERT loaded.")
    return tokenizer, model


def analyze_sentiment(text: str, tokenizer, model) -> str:
    inputs  = tokenizer(text, return_tensors="pt", truncation=True,
                        padding=True, max_length=512)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    # 0=positive, 1=negative, 2=neutral
    labels = ["positive", "negative", "neutral"]
    return labels[torch.argmax(probs).item()]


def get_sentiment_signals(
    articles_by_name : dict,
    tokenizer,
    model,
) -> dict[str, str]:
    sentiment_signals = {}

    print("\nSentiment vote results:")

    for name, symbol in NAME_TO_SYMBOL.items():
        articles = articles_by_name.get(name, [])
        counts   = {"positive": 0, "negative": 0, "neutral": 0}

        for article in articles:
            title   = article.get("title", "")
            snippet = article.get("snippet", "")
            text    = f"{title}. {snippet}" if snippet else title
            if not text:
                continue
            label = analyze_sentiment(text, tokenizer, model)
            counts[label] += 1

        total    = sum(counts.values())
        dominant = max(counts, key=counts.get)
        dom_pct  = counts[dominant] / total if total > 0 else 0

        final = dominant if dom_pct >= CONFIDENCE_THRESHOLD else "neutral"
        sentiment_signals[symbol] = final.upper()

        icon = {"positive": "", "negative": "", "neutral": ""}[final]
        note = "  below threshold → NEUTRAL" if dom_pct < CONFIDENCE_THRESHOLD else ""
        print(
            f"  {symbol} | {icon} {final.upper():8s} | "
            f"pos={counts['positive']:2d} neg={counts['negative']:2d} neu={counts['neutral']:2d} "
            f"| dominant={dom_pct:.0%}{note}"
        )

    print("  " + "─" * 51)
    return sentiment_signals

def predict_lstm(
    ticker      : str,
    scaler,
    feature_cols: list,
    lstm_model,
) -> float | None:
    path = DATA_PATH / f"{ticker}.csv"
    if not path.exists():
        print(f"  [{ticker}] CSV not found — skipping.")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < LOOKBACK + 20:
        print(f"  [{ticker}] Not enough rows ({len(df)}) — skipping.")
        return None

    # feature engineering 
    tech_df = add_technical_features(df.copy())

    # ticker one-hot encoding
    for t in TICKERS:
        tech_df[f"t_{t}"] = 1 if t == ticker else 0

    missing = [c for c in feature_cols if c not in tech_df.columns]
    if missing:
        print(f"  [{ticker}] Missing features: {missing} — skipping.")
        return None

    # take last LOOKBACK rows
    seq = tech_df[feature_cols].iloc[-LOOKBACK:].copy()
    if seq.isna().any().any():
        seq = seq.ffill().bfill().fillna(0)

    # scaler
    scaled  = scaler.transform(seq.values)
    X_input = scaled.reshape(1, LOOKBACK, len(feature_cols))

    prob = float(lstm_model.predict(X_input, verbose=0)[0][0])
    return prob


def get_lstm_signals(lstm_model, scaler, feature_cols) -> dict[str, int]:
    print("\nLSTM prediction results:")

    lstm_signals = {}
    for ticker in TICKERS:
        prob = predict_lstm(ticker, scaler, feature_cols, lstm_model)
        if prob is None:
            lstm_signals[ticker] = 0
            continue
        signal = 1 if prob > LSTM_THRESHOLD else 0
        lstm_signals[ticker] = signal
        icon = "" if signal == 1 else ""
        print(f"  {ticker} | {icon} prob={prob:.4f} → {'BUY signal' if signal else 'no signal'}")

    print("  " + "─" * 51)
    return lstm_signals

def fuse_and_print(lstm_signals: dict, sentiment_signals: dict):
    print("\nFusing signals ...")
    print("FINAL SIGNALS")
    print("=" * 55)

    buy_list = []

    for ticker in TICKERS:
        lstm_pred  = lstm_signals.get(ticker, 0)
        sentiment  = sentiment_signals.get(ticker, "NEUTRAL")

        if lstm_pred == 1 and sentiment in ("POSITIVE", "NEUTRAL"):
            action = "BUY"
            buy_list.append(ticker)
            icon = ""
        else:
            action = "DO NOTHING"
            icon = ""

        lstm_str = "1 (signal) " if lstm_pred == 1 else "0 (no signal)"
        print(f"  {ticker} | LSTM={lstm_str} | Sentiment={sentiment:8s} | {icon} {action}")

    print("=" * 55)

    if buy_list:
        print(f"\n  Stocks to BUY: {buy_list}")
    else:
        print("\n  No trades today.")
