import pytest
import requests

def test_system_end_to_end():
    response = requests.post("http://127.0.0.1:5000/run_pipeline", json={"ticker": "AAPL"})
    assert response.status_code == 200, "System should complete end-to-end prediction successfully"
    data = response.json()
    assert "prediction_plot" in data, "Prediction plot URL should be present in response"
