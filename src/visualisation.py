import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_actual_vs_predicted(dates, actual, predicted, title='Actual vs Predicted Prices'):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Prices', color='blue')
    plt.plot(dates, predicted, label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.savefig("output/prediction_plot4.png")
    plt.clf()

def plot_confusion_matrix(y_true, y_pred, labels=['Buy', 'Hold', 'Sell']):
    """
    Plots a confusion matrix for classification results.
    
    Parameters:
    - y_true: List of true labels
    - y_pred: List of predicted labels
    - labels: List of class names (default: ['Buy', 'Hold', 'Sell'])
    """
    # Generate confusion matrix with explicit labels to ensure a square matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot the confusion matrix using Seaborn's heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("output/confusion_matrix.png")
    plt.close()
