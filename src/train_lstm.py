# src/train_lstm.py
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# root dir
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_process_data

# config for model training
LOOKBACK = 60
FUTURE_DAYS = 5
TARGET_THRESHOLD = 0.015
TRAIN_SPLIT = 0.8

SRC_DIR = Path(__file__).parent
MODEL_NAME = SRC_DIR / "ShortTerm_LSTM_5d_v1.keras"
ARTIFACTS_NAME = SRC_DIR / "scaler_and_features.joblib"

def train_model():
    print("Loading and Processing Data...")
    X, y, dates, feat_cols, returns = load_and_process_data(
        lookback=LOOKBACK, 
        future_days=FUTURE_DAYS, 
        target_thresh=TARGET_THRESHOLD
    )
    
    print(f"Data Loaded. Sequences: {X.shape}, Targets: {y.shape}")
    
    # split based on time
    unique_dates = np.sort(np.unique(dates))
    cutoff_idx = int(len(unique_dates) * TRAIN_SPLIT)
    cutoff_date = unique_dates[cutoff_idx]
    print(f"Train/Test Split Date: {cutoff_date}")
    
    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    # scaling
    scaler = StandardScaler()
    n_train, lookback, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    scaler.fit(X_train_2d) 
    
    # tranform both
    X_train_scaled = scaler.transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # reshape back to 3D
    X_train = X_train_scaled.reshape(n_train, lookback, n_features)
    X_test = X_test_scaled.reshape(n_test, lookback, n_features)
    returns_test = returns[test_mask]
    
    # class weights (to handle imbalance)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = {0: cw[0], 1: cw[1]}
    
    # building the model
    print("Building LSTM...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        class_weight=weights_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("Evaluation...")
    y_pred_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # return chceck for financial evaluation
    buy_mask = y_pred == 1
    avg_return_buy = returns_test[buy_mask].mean() if buy_mask.any() else 0.0
    avg_return_mkt = returns_test.mean()
    print(f"Market Avg Return (5d): {avg_return_mkt:.4f}")
    print(f"Model Buy Return (5d):  {avg_return_buy:.4f}")
    
    # saving
    print("Saving Artifacts...")
    model.save(str(MODEL_NAME))
    joblib.dump({'scaler': scaler, 'feature_columns': feat_cols}, str(ARTIFACTS_NAME))
    print(f"Model saved to: {MODEL_NAME}")
    print(f"Artifacts saved to: {ARTIFACTS_NAME}")
    print("Done.")

if __name__ == "__main__":
    train_model()