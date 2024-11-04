# src/feature_engineering.py

import pandas as pd
import numpy as np

def add_technical_indicators(data):
    """
    Adds various technical indicators and lagged features to the data.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing stock price data with at least 'Close' column.

    Returns:
    - pd.DataFrame: Data with additional technical indicators and lagged features.
    """
    # Simple Moving Averages (SMA)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Exponential Moving Averages (EMA)
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['Bollinger_Upper'] = data['SMA_20'] + 2 * data['Close'].rolling(window=20).std()
    data['Bollinger_Lower'] = data['SMA_20'] - 2 * data['Close'].rolling(window=20).std()

    # Moving Average Convergence Divergence (MACD)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean() 
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12']-data['EMA_26']
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()

    # Stochastic Oscillator
    data['Stochastic_Oscillator'] = (data['Close'] - data['Low'].rolling(14).min()) / (data['High'].rolling(14).max() - data['Low'].rolling(14).min()) * 100

    # Williams %R
    data['Williams_%R'] = (data['High'].rolling(14).max() - data['Close']) / (data['High'].rolling(14).max() - data['Low'].rolling(14).min()) * -100

    # Commodity Channel Index (CCI)
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    data['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

    # On-Balance Volume (OBV)
    obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], -data['Volume'])
    data['OBV'] = obv.cumsum()

    # Price Rate of Change (ROC)
    data['ROC'] = data['Close'].pct_change(periods=12) * 100

    # Lagged Features (up to 15 days)
    for lag in range(1, 16):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)

    # Momentum Indicator
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)

    # Drop any rows with NaN values generated by rolling or shift operations
    data.dropna(inplace=True)
    
    return data
