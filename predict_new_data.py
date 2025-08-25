import pandas as pd
import numpy as np
import joblib
import os

def preprocess_new_data(df):
    """
    Preprocess the new Bitcoin price data for prediction.
    Assumes features already present, fills missing values.
    """
    df = df.copy()
    required_cols = ['Return', 'MA7', 'MA21', 'RSI', 'MACD', 'MACD_Signal']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input data.")
    df = df[required_cols]
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def predict_direction(df, model, scaler):
    feature_cols = ['Return', 'MA7', 'MA21', 'RSI', 'MACD', 'MACD_Signal']
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    df['Predicted_Target'] = predictions
    df['Predicted_Label'] = df['Predicted_Target'].map({0: 'Down', 1: 'Up'})
    return df

def main():
    csv_path = input("Enter path to the new Bitcoin CSV file for prediction: ").strip()
    if not csv_path:
        print("No file path provided. Exiting.")
        return
    if not os.path.isfile(csv_path):
        print(f"Error: File '{csv_path}' does not exist. Exiting.")
        return

    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')

    print("Preprocessing data ...")
    df_processed = preprocess_new_data(df)

    print("Loading saved model and scaler ...")
    saved = joblib.load('model.pkl')
    model = saved['model']
    scaler = saved['scaler']

    print("Predicting price movement direction ...")
    df_predicted = predict_direction(df_processed, model, scaler)

    output_file = "predicted_results.csv"
    df_predicted.to_csv(output_file)
    print(f"Prediction complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
