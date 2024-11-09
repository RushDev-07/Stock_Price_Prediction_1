# run_pipeline.py
import pandas as pd
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
    data = fetch_stock_data(TICKER,START_DATE)
    #data = yf.download("AAPL", start="2014-01-01", interval='1d')
    


    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    data = preprocess_data(data)
    if data.index.name == 'Date':
        data = data.reset_index()
    data1 = pd.DataFrame(data)
    #data1.to_csv('data/raw/raw_data.csv')
    
    
    # Step 3: Feature Engineering
    print("Adding technical indicators...")
    data = add_technical_indicators(data)
    
    
    
    # Step 4: Data Preparation for Prophet
    print("Preparing data for Prophet model...")
    prophet_data = data[['Date','Close']].rename(columns={'Date':'ds','Close': 'y'})

    
    prophet_data = pd.concat([prophet_data, data.drop(['Date', 'Close'], axis=1)], axis=1).dropna()
    prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)
    prophet_data.to_csv('data/raw/raw_data4.csv')
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
    #print(train_features)
    # Step 9: Initialize and Train Hybrid Model
    print("Initializing and training Hybrid model...")
    #hybrid_model = HybridModel1( model_type="RandomForest",
    #n_estimators=300,
    #max_depth=15,
    #min_samples_split=5,
    #min_samples_leaf=2)
    #hybrid_model.train(train_features, train_target)
    hybrid_model = HybridModel(input_size=train_features.shape[1], hidden_size=256, num_layers=3, dropout_rate=0.3, epochs=150, batch_size=64,learning_rate=0.0005)
    hybrid_model.train(train_features, train_target)


    # Step 10: Hybrid Model Prediction
    print("Making predictions with Hybrid model...")
    hybrid_predictions = hybrid_model.predict(test_features)
    #print(hybrid_predictions)
    # Step 11: Evaluate the Regression Model
    print("Evaluating regression performance...")
    #mae, rmse = evaluate_regression(test_target, hybrid_predictions)
    #print(f"Regression Evaluation - MAE: {mae}, RMSE: {rmse}")
    print("Length of test_target:", len(test_target))
    print("Length of hybrid_predictions:", len(hybrid_predictions))
    if len(hybrid_predictions) < len(test_target):
        test_target = test_target.iloc[:len(hybrid_predictions)]
        test_data = test_data.iloc[:len(hybrid_predictions)]

    # Evaluate the model performance for regression
        

    # Example for regression evaluation
    regression_metrics = evaluate_regression_metrics(test_target, hybrid_predictions)
    print("Regression Metrics:", regression_metrics)

    # Step 5: Generate Labels for Classification
    actual_labels = generate_labels(test_target)
    predicted_labels = generate_labels(hybrid_predictions)

    # Step 6: Evaluate Classification Metrics
    classification_metrics = evaluate_classification_metrics(actual_labels, predicted_labels)
    print("Classification Metrics:", classification_metrics)

    # Step 7: Plot Confusion Matrix
    plot_confusion_matrix(actual_labels, predicted_labels)

    #regression_metrics = evaluate_regression_metrics(test_target, hybrid_predictions)
    #print("Regression Metrics:")
    #for metric, value in regression_metrics.items():
    #    print(f"{metric}: {value:.4f}")

    # Step 12: Plot Actual vs Predicted Prices
    dates = test_data['ds']
    plot_actual_vs_predicted(dates, test_target, hybrid_predictions)

    # Step 13: Classification Evaluation
    # Create binary labels for direction (Up=1, Down=0) for test set and predictions
    test_direction = (test_target.diff().shift(-1) > 0).astype(int)  # Actual direction
    predicted_direction = (pd.Series(hybrid_predictions).diff().shift(-1) > 0).astype(int)  # Predicted direction

    # Suggest trading strategy
    current_price = test_target.iloc[-1]  # Assume the last actual value is the current price
    strategy_recommendation = suggest_trading_strategy(hybrid_predictions, current_price)
    print("\nTrading Strategy Suggestion:")
    print(strategy_recommendation)

if __name__ == "__main__":
    main()
