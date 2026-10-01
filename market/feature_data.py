from market.db import price_history
from features.feature_engine import calculate_features
from market.price_history import PriceHistory


def build_features(pairs, limit=100):
    history = PriceHistory(max_length=limit)

    for pair in pairs:
        data = price_history(pair, limit)

        for timestamp, price in data:
            history.history[pair].append((timestamp, price))

    return calculate_features(history)