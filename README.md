
# Stock Price Prediction Pipeline

This project is a stock prediction pipeline that forecasts future stock prices using a combination of the Prophet model 
and a machine learning model (e.g., Random Forest). The pipeline includes data gathering, preprocessing, feature 
engineering, training, and evaluation. The goal is to provide accurate predictions for stock price trends based on 
historical data and technical indicators.

## Project Structure

The project is organized into the following main files and directories:

```
project/
├── src/                           # Source code directory
│   ├── __init__.py
│   ├── config.py                  # Configurations like ticker and date range
│   ├── data_gathering.py          # Fetches stock data from Yahoo Finance
│   ├── preprocessing.py           # Preprocesses data (e.g., filling missing values)
│   ├── feature_engineering.py     # Adds technical indicators (SMA, EMA, etc.)
│   ├── prophet_model.py           # Manages Prophet model training and prediction
│   ├── hybrid_model.py            # Handles machine learning model training and prediction
│   ├── data_split.py              # Splits data into training and testing sets
│   ├── evaluation.py              # Evaluates model with MAE, RMSE, accuracy, etc.
│   └── visualization.py           # Visualizes actual vs. predicted prices and confusion matrix
├── scripts/
│   └── run_pipeline.py            # Main pipeline script
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites

1. Python 3.7 or later.
2. Install required Python packages using the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

### Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory and install dependencies.

### Running the Pipeline

To run the entire pipeline and make predictions, execute the following command:

```bash
python scripts/run_pipeline.py
```

This script will:
- Fetch historical stock data based on the ticker symbol and date range defined in `config.py`.
- Preprocess and add technical indicators to the data.
- Train the Prophet model to capture seasonality and trends.
- Train a hybrid machine learning model on technical indicators and Prophet predictions.
- Evaluate the model's performance using MAE, RMSE, accuracy, and visualize results.

### Configuration

All configurable parameters are located in `src/config.py`:
- `TICKER`: The stock ticker symbol (e.g., `AAPL` for Apple).
- `START_DATE`: Start date for the data in `YYYY-MM-DD` format.
- `END_DATE`: End date for the data in `YYYY-MM-DD` format.
- `TEST_SIZE`: Fraction of data to use for testing (default is `0.2`).

### Explanation of Each Module in `src/`

1. **`config.py`**: Contains configuration parameters like the stock ticker symbol, start and end dates, and test set size.

2. **`data_gathering.py`**: Defines `fetch_stock_data`, which fetches stock data from Yahoo Finance. The data typically includes Date, Open, High, Low, Close, and Volume.

3. **`preprocessing.py`**: Defines `preprocess_data`, which fills missing values, scales features, and prepares the data for feature engineering and modeling.

4. **`feature_engineering.py`**: Defines `add_technical_indicators`, which calculates technical indicators (e.g., SMA, EMA) to capture trends.

5. **`prophet_model.py`**: Contains the `ProphetModel` class, which handles Prophet model training, creating future dataframes, and making predictions.

6. **`hybrid_model.py`**: Contains the `HybridModel` class, which uses machine learning models (e.g., Random Forest) for advanced predictions.

7. **`data_split.py`**: Defines `split_data`, which divides the dataset into training and testing sets.

8. **`evaluation.py`**: Contains functions for regression (MAE, RMSE) and classification (accuracy, precision, recall) metrics, as well as confusion matrix generation for model evaluation.

9. **`visualization.py`**: Provides functions to plot actual vs. predicted prices and visualize the confusion matrix, helping interpret model performance.

### Recommended UIs

To visualize and interact with the prediction pipeline, here are some recommended UI frameworks:

1. **Streamlit**: Simple and effective for data science applications. Great for quickly creating interactive UIs.
2. **Dash**: Flexible and robust, ideal for more customizable, multi-page applications.
3. **Gradio**: Simple and minimal, useful for rapid prototyping.
4. **Flask + React**: Best for production-level applications, providing full customization and scalability.

### Example Streamlit Setup

You can quickly create an interactive UI using Streamlit by installing it and adding the following code:

```python
# Run with: streamlit run app.py

import streamlit as st
from src.data_gathering import fetch_stock_data
from src.preprocessing import preprocess_data
# Define more imports and pipeline functions as needed

st.title("Stock Prediction Pipeline")
ticker = st.text_input("Enter stock ticker:", "AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Run Prediction"):
    # Run the pipeline functions and display results
    st.write("Displaying results for:", ticker)
```

### Results Interpretation

- **MAE (Mean Absolute Error)**: Measures average magnitude of errors between predictions and actual values.
- **RMSE (Root Mean Squared Error)**: Measures the square root of the average of squared differences between predictions and actual values, giving more weight to large errors.
- **Confusion Matrix**: Visualizes classification performance by showing the counts of true positives, true negatives, false positives, and false negatives.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
