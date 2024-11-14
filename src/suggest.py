def suggest_trading_strategy(predicted_prices, current_price):
    """
    Suggests a trading strategy based on predicted prices.

    Parameters:
    - predicted_prices: Array of predicted future prices.
    - current_price: The current stock price.

    Returns:
    - A suggestion string with a recommendation to buy, sell, or hold.
    """
    future_trend = predicted_prices[-1] - current_price
    percentage_change = (future_trend / current_price) * 100

    if percentage_change > 5:
        recommendation = f"Buy more stocks. Expected price increase of {percentage_change:.2f}% over the next period."
    elif percentage_change < -5:
        recommendation = f"Consider selling stocks. Expected price decrease of {percentage_change:.2f}% over the next period."
    else:
        recommendation = "Hold your current stocks. Minimal expected change in price."
    print(percentage_change)
    # Write the recommendation to a file
    with open("output/strategy.txt", 'w') as file:
        file.write(recommendation)

    return recommendation
