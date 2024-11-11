import pytest
from src.data_gathering import fetch_stock_data
from src.preprocessing import preprocess_data
from src.feature_engineering import add_technical_indicators
from src.hybrid_model import HybridModel

def test_pipeline_integration():
    ticker = "AAPL"
    raw_data = fetch_stock_data(ticker)
    preprocessed_data = preprocess_data(raw_data)
    feature_data = add_technical_indicators(preprocessed_data)
    model = HybridModel(input_size=feature_data.shape[1], hidden_size=50, num_layers=2, dropout_rate=0.2, epochs=1, batch_size=32)
    assert model.train(feature_data, feature_data["Close"]) is None
