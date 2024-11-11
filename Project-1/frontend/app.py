import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000"

st.title("Stock Price Prediction and Trading Strategy")

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
            confusion_matrix_url = response.json().get("confusion_matrix")
            if confusion_matrix_url:
                st.image(f"{BACKEND_URL}/get_confusion_matrix", caption="Confusion Matrix")
            
            # Display the strategy recommendation
            strategy_recommendation = response.json().get("strategy_recommendation", "No recommendation available.")
            st.subheader("Trading Strategy Recommendation")
            st.write(strategy_recommendation)
        else:
            st.error("Error running pipeline: " + response.json().get("error", "Unknown error"))
    else:
        st.warning("Please enter a valid ticker symbol.")
