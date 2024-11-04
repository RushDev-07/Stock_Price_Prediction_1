from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

def evaluate_regression_metrics(true_values, predicted_values):
    """
    Evaluates regression model performance using common metrics.

    Parameters:
    - true_values: Actual target values (ground truth).
    - predicted_values: Predicted target values from the model.

    Returns:
    - A dictionary containing RMSE, MAE, and R2 score.
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

def evaluate_classification_metrics(true_labels, predicted_labels):
    """
    Evaluates classification model performance using common metrics.

    Parameters:
    - true_labels: Actual class labels (ground truth).
    - predicted_labels: Predicted class labels from the model.

    Returns:
    - A dictionary containing accuracy, confusion matrix, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return {
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

if __name__ == "__main__":
    # Example usage for regression testing
    true_values = [3, -0.5, 2, 7]
    predicted_values = [2.5, 0.0, 2, 8]
    regression_metrics = evaluate_regression_metrics(true_values, predicted_values)
    print("Regression Metrics:")
    for metric, value in regression_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Example usage for classification testing
    true_labels = [1, 0, 1, 1, 0, 1, 0, 1]
    predicted_labels = [1, 0, 1, 0, 0, 1, 1, 1]
    classification_metrics = evaluate_classification_metrics(true_labels, predicted_labels)
    print("\nClassification Metrics:")
    for metric, value in classification_metrics.items():
        if metric == 'Confusion Matrix':
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")
