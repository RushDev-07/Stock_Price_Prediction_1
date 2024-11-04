# frontend/app.py

import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000"

st.title("Stock Price Prediction")

# Input for ticker
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL)")

if st.button("Run Prediction"):
    if ticker:
        # Run the pipeline through the backend
        response = requests.post(f"{BACKEND_URL}/run_pipeline", json={"ticker": ticker})

        if response.status_code == 200:
            st.success("Prediction pipeline executed successfully.")
            
            # Display the prediction plot
            st.image(f"{BACKEND_URL}/get_prediction_plot", caption="Prediction Plot")

            # Display the confusion matrix
            st.image(f"{BACKEND_URL}/get_confusion_matrix", caption="Confusion Matrix")
        else:
            st.error("Error running pipeline: " + response.json().get("error", "Unknown error"))
    else:
        st.warning("Please enter a valid ticker symbol.")
