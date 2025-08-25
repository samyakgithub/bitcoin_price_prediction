# src/model_training.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_features(filepath='data/processed/bitcoin_features.csv'):
    df = pd.read_csv(filepath, index_col=0)
    return df

def prepare_data(df):
    feature_cols = ['Return', 'MA7', 'MA21', 'RSI', 'MACD', 'MACD_Signal']
    X = df[feature_cols].ffill().bfill()
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, scaler

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Evaluation results for {model_name}:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    plot_filename = f'confusion_matrix_{model_name}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Confusion matrix saved to {plot_filename}")

def save_model_and_scaler(model, scaler, model_path='model.pkl'):
    dir_name = os.path.dirname(model_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print(f"Model and scaler saved to {model_path}")


def train_and_evaluate_all():
    df = load_features()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, model_name="Logistic_Regression")

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, model_name="Random_Forest")

    # Save the Random Forest model and scaler for later inference
    save_model_and_scaler(rf_model, scaler)

if __name__ == "__main__":
    train_and_evaluate_all()
