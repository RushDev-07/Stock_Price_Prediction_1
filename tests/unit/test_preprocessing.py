import pytest
from src.preprocessing import preprocess_data
import pandas as pd

def test_preprocess_data():
    raw_data = pd.DataFrame({
        "Open": [1, 2, None, 4],
        "Close": [1.5, 2.5, 3.5, 4.5]
    })
    processed_data = preprocess_data(raw_data)
    assert processed_data.isnull().sum().sum() == 0, "All missing values should be filled"
