from locust import HttpUser, task

class LoadTestUser(HttpUser):
    @task
    def run_prediction(self):
        self.client.post("/run_pipeline", json={"ticker": "AAPL"})
