from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_regression(test_target, predictions):
    mae = mean_absolute_error(test_target, predictions)
    rmse = mean_squared_error(test_target, predictions, squared=False)
    return mae, rmse

def evaluate_classification(test_target, predicted_direction):
    accuracy = accuracy_score(test_target, predicted_direction)
    precision = precision_score(test_target, predicted_direction)
    recall = recall_score(test_target, predicted_direction)
    conf_matrix = confusion_matrix(test_target, predicted_direction)
    return accuracy, precision, recall, conf_matrix
