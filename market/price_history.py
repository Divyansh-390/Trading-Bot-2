from collections import defaultdict, deque


class PriceHistory:
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.history = defaultdict(lambda: deque(maxlen=max_length))

    def update(self, market_data):
        for pair, data in market_data.items():
            price = data.get("LastPrice")
            timestamp = data.get("timestamp")

            if price is not None and timestamp is not None:
                self.history[pair].append((timestamp, price))

    def get(self, pair):
        return list(self.history[pair])

    def pairs(self):
        return list(self.history.keys())

    def length(self, pair):
        return len(self.history[pair])