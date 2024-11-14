import requests

def test_no_sql_injection():
    url = "http://127.0.0.1:5000/run_pipeline"
    response = requests.post(url, json={"ticker": "AAPL' OR '1'='1"})
    assert response.status_code == 400, "API should handle malformed inputs securely"
