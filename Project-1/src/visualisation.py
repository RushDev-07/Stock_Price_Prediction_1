import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(dates, actual, predicted, title='Actual vs Predicted Prices'):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Prices', color='blue')
    plt.plot(dates, predicted, label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names=['Down', 'Up']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
