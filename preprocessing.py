# src/preprocessing.py

import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD

def load_raw_data(filepath='data/raw/bitcoin_data.csv'):
    """
    Load raw Bitcoin price data from CSV.
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    return df

def calculate_technical_indicators(df):
    """
    Calculate technical indicators: daily returns, moving averages, RSI, MACD
    """
    df = df.copy()
    
    # Daily return
    df['Return'] = df['Close'].pct_change()

    # 7-day and 21-day moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()

    # RSI (14 days)
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # MACD line and signal line
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    return df

def create_target_variable(df):
    """
    Create binary target: 1 if next day's close price is higher, else 0.
    """
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

def preprocess_data(raw_filepath='data/raw/bitcoin_data.csv', save_filepath='data/processed/bitcoin_features.csv'):
    """
    Full preprocessing pipeline: load, calculate indicators, create target, and save processed data
    """
    df = load_raw_data(raw_filepath)
    df = calculate_technical_indicators(df)
    df = create_target_variable(df)

    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    df.to_csv(save_filepath)

    print(f"Processed data saved to {save_filepath}")
    return df

if __name__ == "__main__":
    preprocess_data()
