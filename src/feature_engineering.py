import pandas as pd
import numpy as np

def add_technical_features(df):
    df = df.copy().sort_values('date')
    
    if 'adjclose' not in df.columns:
        raise ValueError("missing 'adjclose' column")
        
    # price features
    df['close'] = df['adjclose']
    df['return_1d'] = df['close'].pct_change()
    
    # moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_60'] = df['close'].rolling(window=60).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
    
    # volatility (Rolling Std Dev)
    df['vol_20'] = df['close'].rolling(20).std()
    
    # momentum / ROC
    df['roc_20'] = df['close'].pct_change(periods=20)
    
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # bollinger bands (30)
    df['bb_mid'] = df['close'].rolling(30).mean()
    df['bb_std'] = df['close'].rolling(30).std()
    df['bb_up'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_low'] = df['bb_mid'] - 2 * df['bb_std']
    
    # volume features
    if 'adjvolume' in df.columns:
        df['vol_change_1d'] = df['adjvolume'].pct_change()
    else:
        df['vol_change_1d'] = 0.0
    
    # getting rid of null rows 
    df = df.dropna().reset_index(drop=True)
    
    return df