import pytest
from src.feature_engineering import add_technical_indicators
import pandas as pd

def test_add_technical_indicators():
    data = pd.DataFrame({"Close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    data_with_indicators = add_technical_indicators(data)
    assert "SMA" in data_with_indicators.columns, "SMA indicator should be added"
    assert "EMA" in data_with_indicators.columns, "EMA indicator should be added"
