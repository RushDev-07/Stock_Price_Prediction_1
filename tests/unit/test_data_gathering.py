import pytest
from src.data_gathering import fetch_stock_data

def test_fetch_stock_data():
    ticker = "AAPL"
    data = fetch_stock_data(ticker)
    assert data is not None, "Data should not be None"
    assert 'Close' in data.columns, "Data should contain 'Close' column"
