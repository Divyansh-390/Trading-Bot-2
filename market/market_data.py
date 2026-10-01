import time


class MarketData:
    def __init__(self, client=None):
        self.client = client
        self.data = {}
        self.last_update = 0

    def update(self):
        if self.client is None:
            raise RuntimeError("No market data source configured")

        response = self.client.ticker()

        if not response.get("Success", True):
            raise RuntimeError(
                response.get("ErrMsg", "Ticker request failed")
            )

        self.data = response["Data"]
        self.last_update = time.time()

        return self.data

    def set_data(self, data):
        self.data = data
        self.last_update = time.time()

    def all(self):
        return self.data

    def get(self, pair):
        return self.data.get(pair)

    def price(self, pair):
        data = self.get(pair)

        if data is None:
            return None

        return data.get("LastPrice")

    def bid(self, pair):
        data = self.get(pair)

        if data is None:
            return None

        return data.get("MaxBid")

    def ask(self, pair):
        data = self.get(pair)

        if data is None:
            return None

        return data.get("MinAsk")

    def spread(self, pair):
        data = self.get(pair)

        if data is None:
            return None

        bid = data.get("MaxBid")
        ask = data.get("MinAsk")

        if bid is None or ask is None:
            return None

        return ask - bid

    def age(self):
        if self.last_update == 0:
            return None

        return time.time() - self.last_update