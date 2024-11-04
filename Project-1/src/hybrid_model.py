# src/hybrid_model.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridModel1:
    def __init__(self, model_type="RandomForest", **model_params):
        """
        Initialize the HybridModel with a specified model type.

        Parameters:
        - model_type (str): Type of model to use ('RandomForest' or 'LinearRegression').
        - model_params (dict): Parameters for the selected model.
        """
        self.model_type = model_type
        self.model = self._initialize_model(model_type, model_params)
        logger.info(f"Hybrid model initialized with {model_type}")

    def _initialize_model(self, model_type, model_params):
        """
        Initialize the specified ML model.

        Parameters:
        - model_type (str): The type of model to initialize.
        - model_params (dict): Parameters for the model.

        Returns:
        - model: Initialized machine learning model.
        """
        if model_type == "RandomForest":
            return RandomForestRegressor(**model_params)
        elif model_type == "LinearRegression":
            return LinearRegression(**model_params)
        else:
            raise ValueError("Unsupported model type. Choose 'RandomForest' or 'LinearRegression'.")

    def train(self, train_features, train_target):
        """
        Train the hybrid model on the training data.

        Parameters:
        - train_features (pd.DataFrame): Features for training.
        - train_target (pd.Series): Target values for training.

        Returns:
        - self: Trained model.
        """
        try:
            self.model.fit(train_features, train_target)
            logger.info("Hybrid model training completed.")
        except Exception as e:
            logger.error("Error during training: %s", e)
            raise e
        return self

    def predict(self, test_features):
        """
        Make predictions using the trained hybrid model.

        Parameters:
        - test_features (pd.DataFrame): Features for prediction.

        Returns:
        - pd.Series: Predicted values.
        """
        try:
            predictions = self.model.predict(test_features)
            logger.info("Prediction completed.")
            return predictions
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            raise e

    def evaluate(self, test_target, predictions):
        """
        Evaluate the model performance using MAE and RMSE.

        Parameters:
        - test_target (pd.Series): Actual target values.
        - predictions (pd.Series): Predicted target values.

        Returns:
        - dict: Evaluation metrics (MAE and RMSE).
        """
        mae = mean_absolute_error(test_target, predictions)
        rmse = mean_squared_error(test_target, predictions, squared=False)
        logger.info(f"Evaluation - MAE: {mae}, RMSE: {rmse}")
        return {"MAE": mae, "RMSE": rmse}