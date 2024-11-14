---

# Stock Price Prediction and Trading Strategy Pipeline

This project provides a stock price prediction and trading recommendation tool. It leverages a hybrid model combining Prophet (for trend and seasonality) and LSTM (for sequential learning) to predict future stock prices. The tool includes an interactive user interface, allowing users to enter stock tickers, view predictions, and receive buy/sell/hold recommendations.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
  - [Unit Testing](#unit-testing)
  - [Integration Testing](#integration-testing)
  - [System Testing](#system-testing)
  - [Performance and Load Testing](#performance-and-load-testing)
  - [Security Testing](#security-testing)
  - [Code Quality Checks](#code-quality-checks)
- [Continuous Integration](#continuous-integration)
- [License](#license)

## Project Overview

The goal of this project is to forecast stock prices based on historical data and provide actionable trading strategies (buy, sell, hold) using an intuitive Streamlit interface. This software includes modules for data collection, preprocessing, feature engineering, modeling, and visualization.

## Features

- **Stock Price Forecasting**: Uses a hybrid Prophet and bidirectional LSTM model for accurate stock price predictions.
- **Technical Indicators**: Incorporates SMA, EMA, Bollinger Bands, and other indicators for enriched features.
- **Trading Strategy Recommendations**: Suggests buy, sell, or hold actions based on predictions.
- **Interactive Interface**: Streamlit-powered UI for entering stock tickers, displaying predictions, and viewing trading recommendations.

## Technology Stack

- **Frontend**: Streamlit for user interface.
- **Backend**: Flask for pipeline and API management.
- **Machine Learning**: PyTorch for LSTM modeling and Facebook Prophet for time-series analysis.
- **Data Processing**: Pandas and NumPy for data manipulation.
- **Testing**: Pytest for unit and integration tests, Locust for performance testing, OWASP ZAP for security testing.
- **CI/CD**: GitHub Actions for automated testing.

## Project Structure

```plaintext
project/
├── src/                           # Source code directory
│   ├── __init__.py
│   ├── config.py                  # Configurations (ticker symbol, date range, etc.)
│   ├── data_gathering.py          # Fetches stock data from Yahoo Finance
│   ├── preprocessing.py           # Data preprocessing (missing values, scaling)
│   ├── feature_engineering.py     # Adds technical indicators (SMA, EMA, etc.)
│   ├── prophet_model.py           # Manages Prophet model training and predictions
│   ├── hybrid_model.py            # Manages LSTM model training and predictions
│   ├── data_split.py              # Splits data into training and testing sets
│   ├── evaluation.py              # Model evaluation (MAE, RMSE, accuracy, etc.)
│   └── visualization.py           # Plots actual vs. predicted prices
├── tests/                         # Tests directory
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── system/                    # System and end-to-end tests
│   ├── performance/               # Performance and load tests
│   ├── security/                  # Security tests
│   └── linting/                   # Code quality checks
├── .github/                       # CI configuration
│   └── workflows/
│       └── test.yml               # GitHub Actions for automated testing
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies** (e.g., Locust for load testing, Selenium for end-to-end tests):
   ```bash
   pip install locust pytest flake8
   ```

## Usage

### Running the Backend

1. Navigate to the project directory:
   ```bash
   cd backend
   ```
2. Start the Flask server:
   ```bash
   python app.py
   ```

### Running the Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### Running the Entire Prediction Pipeline

To run the pipeline end-to-end and obtain predictions, use the following command:
```bash
python scripts/run_pipeline.py
```

## Testing

To ensure quality, this project includes comprehensive testing for all components, using various testing strategies.

### Unit Testing

**Location**: `tests/unit/`

Run all unit tests to verify the functionality of individual functions:
```bash
pytest tests/unit/
```

### Integration Testing

**Location**: `tests/integration/`

Run integration tests to verify interactions between modules:
```bash
pytest tests/integration/
```

### System Testing

**Location**: `tests/system/`

Run end-to-end tests to verify the entire pipeline functions correctly:
```bash
pytest tests/system/
```

### Performance and Load Testing

**Location**: `tests/performance/`

Run load testing with Locust to simulate multiple users:
1. Start Locust load testing:
   ```bash
   locust -f tests/performance/load_test_backend.py
   ```
2. Access Locust’s web interface at `http://127.0.0.1:8089`.

### Security Testing

**Location**: `tests/security/`

Security tests check for vulnerabilities in the backend:
```bash
pytest tests/security/
```

Alternatively, use OWASP ZAP for more detailed security scans:
1. Install [OWASP ZAP](https://www.zaproxy.org/download/).
2. Run ZAP against the backend endpoint to identify potential security vulnerabilities.

### Code Quality Checks

**Location**: `tests/linting/`

Run `pylint` and `flake8` to enforce code standards:
```bash
python tests/linting/run_code_quality_checks.py
```

## Continuous Integration

GitHub Actions is set up to automate tests on each commit or pull request. The CI workflow file (`.github/workflows/test.yml`) includes steps for:

- Running unit, integration, and system tests with `pytest`.
- Checking code quality with `pylint` and `flake8`.
- Running end-to-end tests for the UI.

To trigger CI/CD, push changes to the repository, and GitHub Actions will automatically start the testing workflow.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For questions or contributions, feel free to open an issue or submit a pull request.