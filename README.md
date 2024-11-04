# Stock Price Prediction and Trading Strategy Tool

## Overview
This project is an advanced tool for predicting stock prices and generating trading strategies using a hybrid bidirectional LSTM model. It processes historical stock data, forecasts future price trends, and provides actionable trading insights such as whether to buy, sell, or hold stocks. The system features an intuitive frontend built with Streamlit for ease of use, making it ideal for traders and analysts seeking data-driven decision support.

## Features
- **Stock Price Prediction**: Uses a powerful bidirectional LSTM model for accurate time-series forecasting.
- **Trading Strategy Recommendation**: Suggests whether to buy, sell, or hold stocks based on predicted trends.
- **Interactive User Interface**: Built with Streamlit for seamless user interaction and visualization.
- **Technical Indicator Integration**: Includes various technical indicators for enhanced prediction.

## Technology Stack
- **Frontend**: Streamlit for interactive web applications.
- **Backend**: Flask to manage the pipeline execution and API integration.
- **Machine Learning**: PyTorch for building and training the bidirectional LSTM model.
- **Data Processing**: Pandas and NumPy for data manipulation and feature engineering.
- **Visualization**: Matplotlib for generating prediction plots.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install additional dependencies** (e.g., `plotly` for better visualization):
   ```bash
   pip install plotly
   ```

## Usage
### Running the Backend
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Run the Flask server:
   ```bash
   python app.py
   ```
### Running the Frontend
1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How to Use
1. **Enter a Stock Ticker**: Input the stock ticker symbol (e.g., AAPL) in the provided field.
2. **Run the Prediction**: Click the "Run Prediction" button.
3. **View Results**: The prediction plot and trading strategy will be displayed on the screen.

## Key Components
### 1. LSTM Model
A bidirectional LSTM model captures both past and future trends in the data for more accurate predictions. The model includes multiple layers, dropout for regularization, and an adjustable hidden size.

### 2. Trading Strategy
The project provides a trading recommendation based on the predicted trend. The logic suggests:
- **Buy**: When a significant price increase is expected.
- **Sell**: When a significant price decrease is anticipated.
- **Hold**: When minimal price change is expected.

### 3. Data Pipeline
The data pipeline involves:
- **Data Gathering**: Fetching stock price data using APIs (e.g., `yfinance`).
- **Preprocessing**: Scaling and structuring data into sequences for LSTM input.
- **Feature Engineering**: Adding technical indicators like SMA, EMA, and Bollinger Bands.

## Future Enhancements
- **Integration with live market data** for real-time predictions.
- **Improved trading logic** with risk management strategies.
- **Enhanced visualizations** using interactive libraries like `plotly`.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## License
This project is licensed under the MIT License.

---

For any questions or support, please reach out to [your.email@example.com].

