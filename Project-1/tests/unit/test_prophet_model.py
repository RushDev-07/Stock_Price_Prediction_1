import pytest
from src.prophet_model import ProphetModel
import pandas as pd

def test_prophet_model_prediction():
    model = ProphetModel()
    data = pd.DataFrame({"ds": pd.date_range(start="2023-01-01", periods=10), "y": range(10)})
    model.train(data)
    future_dates = model.predict(5)
    assert len(future_dates) == 5, "The model should predict for 5 future dates"
