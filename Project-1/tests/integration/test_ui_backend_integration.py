import requests

def test_ui_backend_integration():
    response = requests.post("http://127.0.0.1:5000/run_pipeline", json={"ticker": "AAPL"})
    assert response.status_code == 200, "Backend should return a 200 status code"
    data = response.json()
    assert "prediction_plot" in data, "Prediction plot URL should be included in the response"
