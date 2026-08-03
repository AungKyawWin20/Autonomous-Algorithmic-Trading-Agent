# Autonomous Algorithmic Trading Agent: Multi-modal Fusion of LSTM and Financial NLP (Senior Capstone Project)

## Project Overview
This repository contains an end-to-end AI trading research project for U.S. large-cap equities, built around an LSTM-based predictive model and a backtesting engine.

Key capabilities:
- preprocess historical equities data
- engineer technical features
- train a short-term LSTM classifier for 5-day return predictions
- evaluate the trained model using backtests across multiple time ranges
- visualize portfolio performance and trade results
- optionally fuse sentiment signals and run an Alpaca paper trading workflow
- explore results through a Streamlit dashboard in `app.py`

## Repository Structure

- `app.py` - Streamlit dashboard and UI wrapper for backtests and trading outputs
- `requirements.txt` - Python dependencies
- `backtest_results/` - generated CSV and chart outputs from backtesting
- `data/` - raw and processed market data
  - `data/raw/` - raw downloaded CSVs for each ticker
  - `data/processed/` - cleaned and shuffled input CSVs used for training and backtesting
- `notebooks/eda.ipynb` - notebook with exploratory model evaluation and training workflow
- `src/` - core project modules
  - `alpaca_trader.py` - live/paper trading workflow with Alpaca and signal fusion
  - `backtest_strategy.py` - robust backtest engine, metrics, plotting, and model loading
  - `data_loader.py` - load and sequence the processed stock data for training
  - `feature_engineering.py` - technical feature generation logic
  - `generate_signals.py` - news sentiment + LSTM signal generation pipeline
  - `preprocess_ohlcv.py` - likely data cleaning and CSV preparation utilities
  - `run_backtest.py` - script wrapper for running a fixed backtest period and saving results
  - `train_lstm.py` - training pipeline for the LSTM model
  - `update_data.py` - data refresh/update workflow
  - `scaler_and_features.joblib` - saved scaler and feature list for model inference
  - `ShortTerm_LSTM_5d_v1.keras` - pretrained LSTM model artifact

## Installation

1. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the data files exist:
   - `data/processed/AAPL.csv`
   - `data/processed/TSLA.csv`
   - `data/processed/NVDA.csv`
   - `data/processed/MSFT.csv`

4. If you want to run the Alpaca trading workflow or Streamlit dashboard, set the required environment variables in a `.env` file or your shell:
   ```ini
   SERP_API=<your_serpapi_key>
   Tiingo_API=<your_tiingo_token>
   Alpaca_Key=<your_alpaca_api_key>
   Alpaca_Secret=<your_alpaca_secret>
   ```

## Training the LSTM Model

The training pipeline is implemented in `src/train_lstm.py`.

### Model training logic
- Loads processed stock CSVs through `src/data_loader.py`
- Uses `src.feature_engineering.py` to add technical indicators
- Builds a dataset of 60-day sequences for 4 tickers (`AAPL`, `TSLA`, `NVDA`, `MSFT`)
- Creates a binary target where positive class means the 5-day future return is greater than `1.5%`
- Splits data by time using `TRAIN_SPLIT = 0.8`
- Standard-scales the feature set before training
- Trains a Sequential LSTM model with:
  - `LSTM(64, return_sequences=True)`
  - `Dropout(0.3)`
  - `LSTM(32)`
  - `Dropout(0.3)`
  - `Dense(16, relu)`
  - `Dense(1, sigmoid)`
- Uses class weighting to mitigate label imbalance
- Monitors `val_loss` and applies early stopping + LR reduction

### Run training
```bash
python src/train_lstm.py
```

### Output artifacts
- `src/ShortTerm_LSTM_5d_v1.keras` - saved model
- `src/scaler_and_features.joblib` - StandardScaler and feature column list

## Data Preparation and Features

### Data loader
`src/data_loader.py` loads processed stock data and prepares LSTM sequences.

The loader:
- reads each ticker CSV from `data/processed/`
- computes technical features using `src.feature_engineering.add_technical_features`
- creates a 5-day forward return target
- one-hot encodes ticker identity
- builds 60-day sequences for each ticker

### Technical features
In `src/feature_engineering.py`, the following features are computed:
- 1-day return
- 20- and 60-day simple moving averages (SMA)
- 20- and 60-day exponential moving averages (EMA)
- 20-day volatility (rolling standard deviation)
- 20-day rate-of-change (ROC)
- 14-day RSI
- MACD and MACD signal line
- 30-day Bollinger Bands (mid, upper, lower)
- 1-day volume change

These features form the LSTM input together with ticker one-hot encodings.

## Backtesting Engine

Primary backtest logic lives in `src/backtest_strategy.py`.

### Backtest rules
- Uses historical price data from `data/processed/`
- Runs a daily simulation across all tickers
- Evaluates the LSTM model on the most recent 60-day feature window
- Buys a stock if the model outputs probability > 0.5
- Allocates up to `25%` of portfolio value per position
- Requires minimum trade size of `$500`
- Sells positions after exactly `5` calendar days (`HOLD_PERIOD = 5`)
- Closes any remaining positions at the backtest end date

### Trading assumptions
- Uses adjusted close prices (`adjclose`) to avoid dividend/split distortions
- Maintains cash and open positions in a simple spot-style portfolio
- Does not model transaction costs, slippage, or margin
- Executes market buys on signal generation day and fixed hold exits

### Backtest outputs
The engine returns:
- `trades_df` with BUY/SELL events, P&L, percentages, and holding days
- `portfolio_df` with daily portfolio value, cash, and position count

### Metrics calculated
The project computes:
- total return
- annualized Sharpe ratio (252 trading days)
- maximum drawdown
- win rate of completed trades
- average trade return, average win, and average loss

### Plotting
`src/backtest_strategy.py` also generates summary charts:
- portfolio value over time
- drawdown curve
- trade returns distribution
- text summary of performance metrics

## Evaluation and Showcase

If you only want to showcase the pretrained model, use the saved artifacts:
- `src/ShortTerm_LSTM_5d_v1.keras`
- `src/scaler_and_features.joblib`

Then:
1. Load the data using `src.data_loader.load_and_process_data`
2. Scale with the loaded scaler
3. Build sequences with the same feature columns
4. Predict with the loaded Keras model
5. Report classification metrics or backtest performance

## Backtest Script

Run the fixed date-range backtest script:
```bash
python src/run_backtest.py
```

This script currently backtests a specific date range and, saves:
- `backtest_results/Trades (YYYY-MM-DD to YYYY-MM-DD).csv`
- `backtest_results/Portfolio (YYYY-MM-DD to YYYY-MM-DD).csv`
- plots saved as PNG files

## Streamlit Dashboard and CLI Interface

The repository includes a dashboard and CLI wrapper:
- `app.py` - Streamlit app with login, metrics display, and backtest/trade output rendering
- `src/run_backtest.py` - CLI-friendly backtest runner

### Run the dashboard
```bash
streamlit run app.py
```

### Run the Alpaca workflow in dry-run mode
```bash
python src/alpaca_trader.py --dry-run
```

## Signal Fusion and Live Trading

The live signal pipeline uses both model and news sentiment:
- `src/generate_signals.py` fetches news for each ticker
- sentiment is analyzed with FinBERT via `transformers`
- `src/alpaca_trader.py` fuses LSTM predictions with sentiment
- buy decisions require both a bullish model signal and non-negative sentiment

This part of the system is designed for paper trading through Alpaca.

## Notes and Caveats

- The model target is a binary classification for a 5-day return threshold (`1.5%`), so the strategy is explicitly short-term.
- Backtest logic uses fixed position sizing and fixed holding periods.
- There is no risk management beyond fixed allocation and exit by hold period.
- The code assumes the `data/processed` dataset is aligned and contains all tickers.
- The notebook `notebooks/eda.ipynb` is intended for exploration and evaluation.
