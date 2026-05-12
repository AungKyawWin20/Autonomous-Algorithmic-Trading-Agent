import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_technical_features

# stocks
TICKERS = ['AAPL', 'TSLA', 'NVDA', 'MSFT']

# processed data path
DATA_PATH = PROJECT_ROOT / "data" / "processed"

def load_and_process_data(lookback=60, future_days=5, target_thresh=0.015):

    # load csvs into a single df
    dfs = []
    for t in TICKERS:
        path = DATA_PATH / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        df = pd.read_csv(path, parse_dates=['date'])
        df.sort_values('date', inplace=True)
        df['ticker'] = t
        dfs.append(df)
    
    raw = pd.concat(dfs, ignore_index=True)
    
    # use the feature engineering import
    feature_frames = []
    for t in TICKERS:
        sub = raw[raw['ticker'] == t].copy()
        feat = add_technical_features(sub)
        feature_frames.append(feat)
        
    features_df = pd.concat(feature_frames, ignore_index=True)
    
    # target variable
    features_df['future_close'] = features_df.groupby('ticker')['close'].shift(-future_days)
    features_df['future_return'] = (features_df['future_close'] - features_df['close']) / features_df['close']
    
    # target 1 if future return > 1.5%, else 0
    features_df['target'] = (features_df['future_return'] > target_thresh).astype(int)
    features_df = features_df.dropna(subset=['future_close']).reset_index(drop=True)
    
    # one-hot encode for stock
    ticker_dummies = pd.get_dummies(features_df['ticker'], prefix='t')
    features_df = pd.concat([features_df, ticker_dummies], axis=1)
    
    # features for lstm
    base_feats = [
        'return_1d', 'sma_20', 'sma_60', 'ema_20', 'ema_60',
        'vol_20', 'roc_20', 'rsi_14', 'macd', 'macd_signal',
        'bb_mid', 'bb_up', 'bb_low', 'vol_change_1d'
    ]
    
    # 14 base + 4 stock, total 18
    feature_columns = base_feats + list(ticker_dummies.columns)
    
    # keeping unscaled features
    X_df = features_df[feature_columns].copy()
    X_df['ticker'] = features_df['ticker'].values
    X_df['date'] = features_df['date'].values
    X_df['target'] = features_df['target'].values
    X_df['future_return'] = features_df['future_return'].values
    
    # sequence creations
    X_seqs, y_seqs = [], []
    dates_seq, returns_seq = [], []
    
    for t in TICKERS:
        df_t = X_df[X_df['ticker'] == t].reset_index(drop=True)
        if len(df_t) < lookback: continue
        
        # sequence for 60 days
        for i in range(lookback, len(df_t)):
            X_seqs.append(df_t[feature_columns].iloc[i-lookback:i].values)
            y_seqs.append(df_t['target'].iloc[i])
            dates_seq.append(df_t['date'].iloc[i])
            returns_seq.append(df_t['future_return'].iloc[i])
            
    return (np.array(X_seqs), np.array(y_seqs), np.array(dates_seq), 
            feature_columns, np.array(returns_seq))