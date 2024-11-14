# run_pipeline.py
import pandas as pd
import yfinance as yf
from src.data_gathering import fetch_stock_data
from src.preprocessing import preprocess_data
from src.feature_engineering import add_technical_indicators
from src.prophet_model import ProphetModel
from src.hybrid_model1 import HybridModel
from src.evaluation import evaluate_classification_metrics, evaluate_regression_metrics
from src.data_split import split_data
from src.visualisation import plot_actual_vs_predicted, plot_confusion_matrix
from src.config import TICKER, START_DATE, END_DATE
from src.suggest import suggest_trading_strategy
from src.utils import generate_labels

def main():
    # Step 1: Data Gathering
    print("Fetching stock data...")
    data = yf.download("CSCO", start="2014-01-01", interval='1d')
    data.to_csv('result/gather.csv')

    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    data = preprocess_data(data)
    if data.index.name == 'Date':
        data = data.reset_index()
    data.to_csv('result/preprocessed.csv')
    
    # Step 3: Feature Engineering
    print("Adding technical indicators...")
    data = add_technical_indicators(data)
    data.to_csv('result/feature_engineered.csv')
    
    # Step 4: Data Preparation for Prophet
    print("Preparing data for Prophet model...")
    prophet_data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_data = pd.concat([prophet_data, data.drop(['Date', 'Close'], axis=1)], axis=1).dropna()
    prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)
    prophet_data.to_csv('result/prophet_data.csv')

    # Step 5: Initialize and Train Prophet Model for Full Data Period
    print("Initializing and training Prophet model...")
    additional_regressors = list(prophet_data.columns.difference(['ds', 'y']))
    prophet_model = ProphetModel(additional_regressors=additional_regressors)
    prophet_model.train(prophet_data)

    # Step 6: Make Predictions with Prophet for Full Data Period
    print("Creating predictions for Prophet model...")
    future_data = prophet_model.make_future_dataframe(periods=0, include_history=True)  # Use all data for history
    for regressor in additional_regressors:
        future_data[regressor] = prophet_data[regressor].values
    prophet_forecast = prophet_model.predict(future_data)
    
    # Include Prophet's forecast as a feature in the entire dataset
    prophet_data['prophet_forecast'] = prophet_forecast['yhat'].values
    prophet_data.to_csv('result/prophet_data_with_forecast.csv')

    # Step 7: Data Splitting (with prophet_forecast included)
    print("Splitting data into train and test sets...")
    train_data, test_data = split_data(prophet_data)
    train_data.to_csv('result/train_data.csv')
    test_data.to_csv('result/test_data.csv')

    # Step 8: Prepare Hybrid Model Features
    print("Preparing features for hybrid model...")
    train_features = train_data.drop(['ds', 'y'], axis=1)
    train_target = train_data['y']
    test_features = test_data.drop(['ds', 'y'], axis=1)
    test_target = test_data['y']

    # Step 9: Initialize and Train Hybrid Model (LSTM with Prophet forecast as feature)
    print("Initializing and training Hybrid model...")
    hybrid_model = HybridModel(input_size=train_features.shape[1], hidden_size=256, num_layers=3, dropout_rate=0.3, epochs=150, batch_size=64, learning_rate=0.0005)
    hybrid_model.train(train_features, train_target)

    # Step 10: Hybrid Model Prediction
    print("Making predictions with Hybrid model...")
    hybrid_predictions = hybrid_model.predict(test_features)
    print(hybrid_predictions)

    # Step 11: Evaluate Regression Performance
    print("Evaluating regression performance...")
    if len(hybrid_predictions) < len(test_target):
        test_target = test_target.iloc[:len(hybrid_predictions)]
        test_data = test_data.iloc[:len(hybrid_predictions)]

    regression_metrics = evaluate_regression_metrics(test_target, hybrid_predictions)
    print("Regression Metrics:", regression_metrics)

    # Step 12: Classification Evaluation
    actual_labels = generate_labels(test_target)
    predicted_labels = generate_labels(hybrid_predictions)
    classification_metrics = evaluate_classification_metrics(actual_labels, predicted_labels)
    print("Classification Metrics:", classification_metrics)
    plot_confusion_matrix(actual_labels, predicted_labels)

    # Step 13: Plot Actual vs Predicted Prices
    dates = test_data['ds']
    plot_actual_vs_predicted(dates, test_target, hybrid_predictions)

    # Step 14: Trading Strategy Recommendation
    current_price = test_target.iloc[-1]
    strategy_recommendation = suggest_trading_strategy(hybrid_predictions, current_price)
    print("\nTrading Strategy Suggestion:")
    print(strategy_recommendation)

if __name__ == "__main__":
    main()
