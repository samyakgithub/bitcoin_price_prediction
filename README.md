# Bitcoin Price Movement Prediction with Machine Learning and Retrieval-Augmented Generation (RAG)

## Project Overview
This project aims to predict the next-day direction (up/down) of Bitcoin prices by integrating traditional machine learning techniques with Retrieval-Augmented Generation (RAG). The approach combines structured historical price data augmented with real-time external textual information such as cryptocurrency news headlines and social media sentiment for improved prediction accuracy.

## Features
- Historical Bitcoin price data collection from Yahoo Finance.
- Calculation of technical indicators: moving averages (MA), relative strength index (RSI), and moving average convergence divergence (MACD).
- Binary classification target indicating next-day price direction.
- Retrieval system that embeds and fetches relevant external textual data using vector similarity.
- Combined modeling using both numeric and textual augmented features.
- Traditional ML classifiers: Logistic Regression and Random Forest.
- Model evaluation with accuracy, precision, recall, and visualization.
- Live prediction demonstration with real-time data and retrieval.

## Project Structure

bitcoin_price_prediction_rag/
├── data/
│   ├── raw/ # Raw historical and external text data
│   ├── processed/ # Processed feature datasets
│   └── external_texts/ # External textual data for retrieval
├── notebooks/
│   ├── 1_data_collection.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_retrieval_module.ipynb
│   ├── 4_data_integration.ipynb
│   ├── 5_model_building.ipynb
│   ├── 6_model_evaluation.ipynb
│   └── 7_live_prediction.ipynb
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── retrieval.py
│   ├── model_training.py
│   └── utils.py
├── requirements.txt
├── environment.yml
└── README.md

Install dependencies:
pip install -r requirements.txt


## Usage
python src/data_collection.py
python src/preprocessing.py
python src/model_training.py


This will save the trained Random Forest model and scaler to `model.pkl`.

### Predict on New Data

To predict price movements on a new CSV file:


python src/predict_new_data.py path_to_new_csv_file.csv

text

