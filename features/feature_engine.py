from features.indicators import returns, volatility, sma, trend


def calculate_features(history, min_history=20):
    result = {}

    for pair in history.pairs():
        data = history.get(pair)

        if len(data) < min_history:
            continue

        prices = [price for _, price in data]

        result[pair] = {
            "momentum_1": returns(data, 60),
            "momentum_5": returns(data, 300),
            "volatility": volatility(prices, 20),
            "sma_10": sma(prices, 10),
            "sma_20": sma(prices, 20),
            "trend": trend(prices, 20)
        }

    return result