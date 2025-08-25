# src/data_collection.py

import yfinance as yf
import os
import pandas as pd

def download_bitcoin_data(start_date='2016-01-01', end_date=None, save_path='data/raw/bitcoin_data.csv'):
    """
    Download historical Bitcoin price data from Yahoo Finance.

    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format. Use None for today's date.
        save_path (str): Filepath to save the downloaded CSV data.

    Returns:
        pd.DataFrame: Downloaded price DataFrame.
    """
    # Download historical price data for Bitcoin (BTC-USD)
    btc = yf.Ticker("BTC-USD")
    df = btc.history(start=start_date, end=end_date)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the raw data to CSV
    df.to_csv(save_path)
    print(f"Historical Bitcoin data saved to {save_path}")

    return df

if __name__ == "__main__":
    # Example usage
    download_bitcoin_data()
