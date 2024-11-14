import numpy as np
import pandas as pd

def generate_labels(prices, threshold=0.01):
    """
    Generates buy, hold, or sell labels based on price change thresholds.

    Parameters:
    - prices: List, numpy array, or pandas Series of prices
    - threshold: Percentage threshold for buy/sell decision (default 1%)

    Returns:
    - List of labels ('Buy', 'Hold', 'Sell')
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    elif isinstance(prices, list):
        prices = np.array(prices)

    prices = prices.reshape(-1)
    labels = []
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i - 1]) / prices[i - 1]
        if change > threshold:
            labels.append("Buy")
        elif change < -threshold:
            labels.append("Sell")
        else:
            labels.append("Hold")
    
    return labels
