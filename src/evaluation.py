# evaluation.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score


def evaluate_regression_metrics(y_true, y_pred):
    """
    Evaluates regression metrics: MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    return {"MAE": mae, "RMSE": rmse}


def evaluate_classification_metrics(y_true, y_pred):
    """
    Evaluates classification metrics: Accuracy, Precision, and Recall.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division = 1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall}



