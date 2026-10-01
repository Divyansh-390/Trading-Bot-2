import numpy as np
from datetime import timedelta

def returns(history, seconds):
    if len(history) < 2:
        return None

    now = history[-1][0]
    current_price = history[-1][1]
    target = now - timedelta(seconds=seconds)

    previous = None

    for timestamp, price in reversed(history[:-1]):
        if timestamp <= target:
            previous = price
            break

    if previous is None:
        return None

    return current_price / previous - 1

def volatility(prices, periods=20):
    prices = np.asarray(prices, dtype=float)

    if len(prices) < periods + 1:
        return None

    r = np.diff(np.log(prices[-periods - 1:]))
    return np.std(r)


def sma(prices, periods):
    prices = np.asarray(prices, dtype=float)

    if len(prices) < periods:
        return None

    return np.mean(prices[-periods:])


def trend(prices, periods=20):
    prices = np.asarray(prices, dtype=float)

    if len(prices) < periods:
        return None

    x = np.arange(periods)
    y = prices[-periods:]

    slope = np.polyfit(x, y, 1)[0]

    return slope / np.mean(y)