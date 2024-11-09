# src/prophet_model.py

from prophet import Prophet
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetModel:
    def __init__(self, additional_regressors=None, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        """
        Initialize the ProphetModel with optional additional regressors and seasonality settings.

        Parameters:
        - additional_regressors (list of str): List of columns to add as regressors in the Prophet model.
        - yearly_seasonality (bool): Whether to include yearly seasonality.
        - weekly_seasonality (bool): Whether to include weekly seasonality.
        - daily_seasonality (bool): Whether to include daily seasonality.
        """
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        self.additional_regressors = additional_regressors if additional_regressors is not None else []
        
        # Add regressors to model
        for regressor in self.additional_regressors:
            self.model.add_regressor(regressor)
        
        logger.info("Prophet model initialized with additional regressors: %s", self.additional_regressors)

    def train(self, train_data):
        """
        Train the Prophet model with the provided training data.

        Parameters:
        - train_data (pd.DataFrame): DataFrame with 'ds' as the date column, 'y' as the target variable, and any additional regressors.

        Returns:
        - self: The fitted model.
        """
        try:
            self.model.fit(train_data)
            logger.info("Prophet model training completed.")
        except Exception as e:
            logger.error("Error during training: %s", e)
            raise e
        return self

    def predict(self, future_data):
        """
        Make predictions on the provided future data.

        Parameters:
        - future_data (pd.DataFrame): DataFrame with 'ds' as the date column and any additional regressors.

        Returns:
        - pd.DataFrame: DataFrame with predictions and dates.
        """
        try:
            forecast = self.model.predict(future_data)
            logger.info("Prediction completed.")
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            raise e

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        """
        Create a future DataFrame with specified periods and frequency.

        Parameters:
        - periods (int): Number of future periods to predict.
        - freq (str): Frequency of the future periods, e.g., 'D' for daily.
        - include_history (bool): Whether to include historical data.

        Returns:
        - pd.DataFrame: Future dates DataFrame with 'ds' column.
        """
        try:
            future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
            logger.info("Future dataframe created with %d periods and frequency %s", periods, freq)
            return future
        except Exception as e:
            logger.error("Error creating future dataframe: %s", e)
            raise e
