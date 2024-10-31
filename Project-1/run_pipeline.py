# run_pipeline.py
import yfinance as yf
import pandas as pd
#from src.data_gathering import fetch_stock_data
from src.preprocessing import preprocess_data
from src.feature_engineering import add_technical_indicators
from src.prophet_model import ProphetModel
from src.hybrid_model import HybridModel
from src.evaluation import evaluate_regression, evaluate_classification
from src.data_split import split_data
from src.visualisation import plot_actual_vs_predicted, plot_confusion_matrix
#from src.config import TICKER, START_DATE, END_DATE

def main():
    # Step 1: Data Gathering
    print("Fetching stock data...")
    data = yf.download("AAPL", start="2014-01-01", end="2024-01-01", interval='1d')
   


    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    data = preprocess_data(data)
    if data.index.name == 'Date':
        data = data.reset_index()
    
    
    # Step 3: Feature Engineering
    print("Adding technical indicators...")
    data = add_technical_indicators(data)
    
    
    
    # Step 4: Data Preparation for Prophet
    print("Preparing data for Prophet model...")
    prophet_data = data[['Date','Close']].rename(columns={'Date':'ds','Close': 'y'})

    
    prophet_data = pd.concat([prophet_data, data.drop(['Date', 'Close'], axis=1)], axis=1).dropna()
    prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)
    prophet_data.to_csv('test.csv')
    # Step 5: Data Splitting
    print("Splitting data into train and test sets...")
    train_data, test_data = split_data(prophet_data)
    #test_data.to_csv('test.csv')

    # Step 6: Initialize and Train Prophet Model
    print("Initializing and training Prophet model...")
    additional_regressors = list(train_data.columns.difference(['ds', 'y']))  # Select all columns except 'ds' and 'y'
    prophet_model = ProphetModel(additional_regressors=additional_regressors)
    prophet_model.train(train_data)

    # Step 7: Create Future DataFrame and Make Predictions with Prophet
    print("Creating future DataFrame and predicting with Prophet model...")
    future_data = prophet_model.make_future_dataframe(periods=len(test_data), include_history=False)
    for regressor in additional_regressors:
        future_data[regressor] = test_data[regressor].values  # Add test set regressors to future data
    prophet_forecast = prophet_model.predict(future_data)

    # Step 8: Prepare Hybrid Model Features
    print("Preparing features for hybrid model...")
    train_features = train_data.drop(['ds', 'y'], axis=1)
    train_target = train_data['y']
    test_features = test_data.drop(['ds', 'y'], axis=1)
    test_target = test_data['y']

    # Step 9: Initialize and Train Hybrid Model
    print("Initializing and training Hybrid model...")
    hybrid_model = HybridModel(model_type="RandomForest", n_estimators=100, random_state=42)
    hybrid_model.train(train_features, train_target)

    # Step 10: Hybrid Model Prediction
    print("Making predictions with Hybrid model...")
    hybrid_predictions = hybrid_model.predict(test_features)

    # Step 11: Evaluate the Regression Model
    print("Evaluating regression performance...")
    mae, rmse = evaluate_regression(test_target, hybrid_predictions)
    print(f"Regression Evaluation - MAE: {mae}, RMSE: {rmse}")

    # Step 12: Plot Actual vs Predicted Prices
    dates = test_data['ds']
    plot_actual_vs_predicted(dates, test_target, hybrid_predictions)

    # Step 13: Classification Evaluation
    # Create binary labels for direction (Up=1, Down=0) for test set and predictions
    test_direction = (test_target.diff().shift(-1) > 0).astype(int)  # Actual direction
    predicted_direction = (pd.Series(hybrid_predictions).diff().shift(-1) > 0).astype(int)  # Predicted direction

    # Evaluate Classification Model
    print("Evaluating classification performance...")
    accuracy, precision, recall, conf_matrix = evaluate_classification(test_direction.dropna(), predicted_direction.dropna())
    print(f"Classification Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()
