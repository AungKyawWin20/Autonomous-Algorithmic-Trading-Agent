import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_technical_features


SRC_DIR = Path(__file__).parent
MODEL_PATH = SRC_DIR / "ShortTerm_LSTM_5d_v1.keras"
ARTIFACTS_PATH = SRC_DIR / "scaler_and_features.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed"
LOOKBACK = 60
HOLD_PERIOD = 5  
INITIAL_CAPITAL = 100000
ALLOCATION_PCT = 0.25  # 25% of portfolio per position
MIN_TRADE_SIZE = 500  # minimum trade amount
TICKERS = ['AAPL', 'TSLA', 'NVDA', 'MSFT']


def load_model_and_artifacts():
    from tensorflow.keras.models import load_model
    import json
    import zipfile
    import h5py
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    
    def _is_hdf5_file(path: str) -> bool:
        try:
            with open(path, 'rb') as f:
                header = f.read(8)
            return header.startswith(b"\x89HDF\r\n\x1a\n")
        except Exception:
            return False
    
    def _build_model_from_config(config: dict) -> Sequential:
        layers = []
        input_shape = None
        for layer in config['config']['layers']:
            cls = layer['class_name']
            conf = layer['config']
            if cls == 'InputLayer':
                batch = conf.get('batch_input_shape')
                if batch and len(batch) >= 3:
                    input_shape = tuple(batch[1:])
            elif cls == 'LSTM':
                units = conf['units']
                return_sequences = conf.get('return_sequences', False)
                if input_shape is not None:
                    layers.append(LSTM(units, return_sequences=return_sequences, input_shape=input_shape, name=conf.get('name')))
                    input_shape = None
                else:
                    layers.append(LSTM(units, return_sequences=return_sequences, name=conf.get('name')))
            elif cls == 'Dropout':
                rate = conf.get('rate', 0.0)
                layers.append(Dropout(rate, name=conf.get('name')))
            elif cls == 'Dense':
                units = conf['units']
                activation = conf.get('activation')
                layers.append(Dense(units, activation=activation, name=conf.get('name')))
        return Sequential(layers)
    
    def _load_weights_from_hdf5_into_model(model: Sequential, h5_path: str):
        import numpy as _np
        with h5py.File(h5_path, 'r') as f:
            mw = f['model_weights']
            for layer in model.layers:
                if layer.name not in mw:
                    continue
                g = mw[layer.name]
                ds = {}
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        key = name.split('/')[-1]
                        ds[key] = obj[()]
                g.visititems(visit)
                expected_shapes = [w.shape for w in layer.get_weights()]
                if not expected_shapes:
                    continue
                weights_to_set = []
                lname = layer.__class__.__name__.lower()
                if 'lstm' in lname:
                    order = ['kernel:0','recurrent_kernel:0','bias:0']
                elif 'dense' in lname:
                    order = ['kernel:0','bias:0']
                else:
                    order = list(ds.keys())
                for name in order:
                    if name in ds:
                        weights_to_set.append(_np.array(ds[name]))
                if len(weights_to_set) < len(expected_shapes):
                    remaining = [k for k in ds.keys() if k not in order]
                    for shp in expected_shapes[len(weights_to_set):]:
                        found = False
                        for k in list(remaining):
                            if ds[k].shape == shp:
                                weights_to_set.append(_np.array(ds[k]))
                                remaining.remove(k)
                                found = True
                                break
                        if not found:
                            raise ValueError(f"Cannot find weight with shape {shp} for layer {layer.name}")
                layer.set_weights(weights_to_set)
    
    def load_model_robust(path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if zipfile.is_zipfile(path):
            return load_model(str(path))
        if _is_hdf5_file(str(path)):
            with h5py.File(str(path), 'r') as f:
                model_conf = f.attrs.get('model_config')
                if model_conf is None:
                    raise ValueError("Legacy HDF5 model missing 'model_config' attribute")
                model_conf_str = model_conf if isinstance(model_conf, str) else model_conf.decode('utf-8')
                model_conf_json = json.loads(model_conf_str)
            model = _build_model_from_config(model_conf_json)
            _load_weights_from_hdf5_into_model(model, str(path))
            return model
        raise ValueError(f"Unknown model file format for: {path}")
    
    print("Loading model and artifacts...")
    model = load_model_robust(str(MODEL_PATH))
    artifacts = joblib.load(str(ARTIFACTS_PATH))
    scaler = artifacts['scaler']
    feature_cols = artifacts['feature_columns']
    
    print(f"Model loaded. Expected features: {len(feature_cols)}")
    return model, scaler, feature_cols


def load_historical_data(start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    print(f"Loading historical data from {start_date.date()} to {end_date.date()}...")
    data = {}
    
    for ticker in TICKERS:
        path = DATA_PATH / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        df = pd.read_csv(path, parse_dates=['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        data[ticker] = df
        print(f"  {ticker}: {len(df)} days of data")
    
    return data


def prepare_features_for_prediction(
    df: pd.DataFrame, 
    ticker: str, 
    feature_cols: List[str],
    scaler,
    lookback: int = LOOKBACK
) -> np.ndarray:
    
    if len(df) < lookback + 20:  # need buffer for technical features
        return None
    
    # feature engineering
    tech_df = add_technical_features(df.copy())
    
    # one-hot
    for t in TICKERS:
        tech_df[f't_{t}'] = 1 if t == ticker else 0
    
    missing_cols = [col for col in feature_cols if col not in tech_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns for {ticker}: {missing_cols}")
        return None
    
    # get the last lokback rows
    if len(tech_df) < lookback:
        return None
    
    last_seq_df = tech_df[feature_cols].iloc[-lookback:].copy()
    
    if last_seq_df.isna().any().any():
        last_seq_df = last_seq_df.ffill().bfill().fillna(0)
    
    # back to df for scaling
    last_seq_df_scaled = pd.DataFrame(last_seq_df, columns=feature_cols)
    last_seq_scaled = scaler.transform(last_seq_df_scaled)
    
    # LSTM reshaping 
    X_input = last_seq_scaled.reshape(1, lookback, len(feature_cols))
    
    return X_input


class BacktestEngine:
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # {ticker: {'shares': int, 'entry_date': date, 'entry_price': float}}
        self.trades: List[Dict] = []
        self.portfolio_values: List[Dict] = []
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        stock_value = sum(
            pos['shares'] * current_prices.get(ticker, 0)
            for ticker, pos in self.positions.items()
        )
        return self.cash + stock_value
    
    def execute_buy(self, ticker: str, date, price: float, portfolio_value: float):
        """Execute a buy order."""
        allocation = portfolio_value * ALLOCATION_PCT
        amount_to_invest = min(allocation, self.cash)
        
        if amount_to_invest < MIN_TRADE_SIZE:
            return False
        
        shares = int(amount_to_invest // price)
        if shares == 0:
            return False
        
        cost = shares * price
        self.cash -= cost
        
        # convert date to Timestamp
        date_ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_date': date_ts,
            'entry_price': price
        }
        
        self.trades.append({
            'date': date_ts,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'cost': cost
        })
        
        return True
    
    def execute_sell(self, ticker: str, date, price: float):
        """Execute a sell order."""
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        shares = pos['shares']
        proceeds = shares * price
        self.cash += proceeds
        
        pnl = proceeds - (shares * pos['entry_price'])
        pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
        
        # date to timestamp
        date_ts = pd.Timestamp(date)
        entry_date_ts = pd.Timestamp(pos['entry_date'])
        hold_days = (date_ts - entry_date_ts).days
        
        self.trades.append({
            'date': date_ts if isinstance(date, (pd.Timestamp, np.datetime64)) else date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_days': hold_days
        })
        
        del self.positions[ticker]
        return True
    
    def check_exits(self, current_date: datetime, current_prices: Dict[str, float]):
        """Check if any positions should be exited based on hold period."""
        to_exit = []
        for ticker, pos in self.positions.items():
            hold_days = (current_date - pos['entry_date']).days
            if hold_days >= HOLD_PERIOD:
                to_exit.append(ticker)
        
        for ticker in to_exit:
            if ticker in current_prices:
                self.execute_sell(ticker, current_date, current_prices[ticker])


def run_backtest(
    start_date: datetime,
    end_date: datetime,
    model,
    scaler,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backtest."""
    print(f"\nStarting backtest from {start_date.date()} to {end_date.date()}")
    
    data = load_historical_data(start_date, end_date)
    
    engine = BacktestEngine(INITIAL_CAPITAL)
    
    # get all unique trading days for all tickers
    all_dates = set()
    for df in data.values():
        all_dates.update(df['date'].values)
    all_dates = sorted(list(all_dates))
    
    # coverting to datetim (if they are numpy datetime64)
    all_dates = [pd.Timestamp(d).to_pydatetime() if isinstance(d, (np.datetime64, pd.Timestamp)) else d for d in all_dates]
    
    print(f">>> Processing {len(all_dates)} trading days...")
    
    # process each day
    for i, current_date in enumerate(all_dates):
        if i % 50 == 0:
            # Handle both datetime and numpy datetime64
            if isinstance(current_date, (np.datetime64, pd.Timestamp)):
                date_str = pd.Timestamp(current_date).strftime('%Y-%m-%d')
            else:
                date_str = current_date.date() if hasattr(current_date, 'date') else str(current_date)
            print(f"  Processing day {i+1}/{len(all_dates)}: {date_str}")
        
        # get current prices for all tickers
        current_prices = {}
        # convert current_date to pandas timestamp for comparison
        current_date_ts = pd.Timestamp(current_date)
        for ticker in TICKERS:
            df_ticker = data[ticker]

            day_data = df_ticker[pd.to_datetime(df_ticker['date']) == current_date_ts]
            if not day_data.empty:
                current_prices[ticker] = day_data.iloc[0]['adjclose']
        
        # check for exits first
        engine.check_exits(current_date, current_prices)
        
        # then check for new entries
        portfolio_value = engine.get_portfolio_value(current_prices)
        
        for ticker in TICKERS:
            # skip if we already own shares of this stock
            if ticker in engine.positions:
                continue
            
            # skip if no data for price
            if ticker not in current_prices:
                continue
            
            # get historical data up to current date
            df_ticker = data[ticker]
            current_date_ts = pd.Timestamp(current_date)
            historical_df = df_ticker[pd.to_datetime(df_ticker['date']) <= current_date_ts].copy()
            
            if len(historical_df) < LOOKBACK + 20:
                continue
            
            # Prepare features and predict
            X_input = prepare_features_for_prediction(
                historical_df, ticker, feature_cols, scaler, LOOKBACK
            )
            
            if X_input is None:
                continue
            
            prob = model.predict(X_input, verbose=0)[0][0]
            
            # buy if probability > 0.5
            if prob > 0.5:
                engine.execute_buy(ticker, current_date, current_prices[ticker], portfolio_value)
                portfolio_value = engine.get_portfolio_value(current_prices)  # Update after buy
        
        # get portfolio value
        pv = engine.get_portfolio_value(current_prices)
        date_for_record = pd.Timestamp(current_date) if not isinstance(current_date, pd.Timestamp) else current_date
        engine.portfolio_values.append({
            'date': date_for_record,
            'portfolio_value': pv,
            'cash': engine.cash,
            'positions_count': len(engine.positions)
        })
    
    # close all remaining positions at end date
    final_prices = {}
    end_date_ts = pd.Timestamp(end_date)
    for ticker in TICKERS:
        df_ticker = data[ticker]
        final_day = df_ticker[pd.to_datetime(df_ticker['date']) <= end_date_ts]
        if not final_day.empty:
            final_prices[ticker] = final_day.iloc[-1]['adjclose']
    
    for ticker in list(engine.positions.keys()):
        if ticker in final_prices:
            engine.execute_sell(ticker, end_date, final_prices[ticker])
    
    # convert to DataFrames
    trades_df = pd.DataFrame(engine.trades)
    portfolio_df = pd.DataFrame(engine.portfolio_values)
    
    return trades_df, portfolio_df


def calculate_metrics(trades_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> Dict:
    """Calculate backtest performance metrics."""
    if trades_df.empty or portfolio_df.empty:
        return {}
    
    # Portfolio metrics
    initial_value = portfolio_df.iloc[0]['portfolio_value']
    final_value = portfolio_df.iloc[-1]['portfolio_value']
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate daily returns
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
    
    # Sharpe ratio (annualized)
    if portfolio_df['daily_return'].std() > 0:
        sharpe_ratio = np.sqrt(252) * portfolio_df['daily_return'].mean() / portfolio_df['daily_return'].std()
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
    max_drawdown = portfolio_df['drawdown'].min() * 100
    
    # Trade metrics
    sell_trades = trades_df[trades_df['action'] == 'SELL']
    if not sell_trades.empty:
        win_rate = (sell_trades['pnl'] > 0).sum() / len(sell_trades) * 100
        avg_return = sell_trades['pnl_pct'].mean()
        total_trades = len(sell_trades)
        winning_trades = (sell_trades['pnl'] > 0).sum()
        losing_trades = (sell_trades['pnl'] < 0).sum()
        avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = sell_trades[sell_trades['pnl'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
    else:
        win_rate = 0
        avg_return = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        avg_win = 0
        avg_loss = 0
    
    return {
        'initial_capital': initial_value,
        'final_value': final_value,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_pct': win_rate,
        'avg_return_pct': avg_return,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss
    }


def plot_results(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, metrics: Dict):
    """Plot backtest results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value over time
    axes[0, 0].plot(portfolio_df['date'], portfolio_df['portfolio_value'], linewidth=2)
    axes[0, 0].axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital')
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax'] * 100
    axes[0, 1].fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trade returns distribution
    if not trades_df.empty:
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if not sell_trades.empty:
            axes[1, 0].hist(sell_trades['pnl_pct'], bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_title('Trade Returns Distribution')
            axes[1, 0].set_xlabel('Return (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics summary
    axes[1, 1].axis('off')
    metrics_text = f"""
    Backtest Results Summary
    
    Initial Capital: ${metrics.get('initial_capital', 0):,.2f}
    Final Value: ${metrics.get('final_value', 0):,.2f}
    Total Return: {metrics.get('total_return_pct', 0):.2f}%
    
    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
    Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%
    
    Total Trades: {metrics.get('total_trades', 0)}
    Win Rate: {metrics.get('win_rate_pct', 0):.2f}%
    Avg Return: {metrics.get('avg_return_pct', 0):.2f}%
    Avg Win: {metrics.get('avg_win_pct', 0):.2f}%
    Avg Loss: {metrics.get('avg_loss_pct', 0):.2f}%
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    family='monospace')
    
    plt.tight_layout()
    return fig
