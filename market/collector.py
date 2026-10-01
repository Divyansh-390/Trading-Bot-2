import sys
import time
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from API.roostoo import RoostooClient

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "market.duckdb"


class MarketCollector:
    def __init__(self, interval=10):
        DATA_DIR.mkdir(exist_ok=True)

        self.interval = interval
        self.client = RoostooClient()

        db = duckdb.connect(str(DB_PATH))

        db.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP,
                pair VARCHAR,
                bid DOUBLE,
                ask DOUBLE,
                last_price DOUBLE,
                change DOUBLE,
                coin_trade_value DOUBLE,
                unit_trade_value DOUBLE
            )
        """)

        db.close()

    def collect(self):
        response = self.client.ticker()

        if not response.get("Success", True):
            raise RuntimeError(
                response.get("ErrMsg", "Ticker request failed")
            )

        timestamp = time.time()
        rows = []

        for pair, data in response["Data"].items():
            rows.append((
                timestamp,
                pair,
                data.get("MaxBid"),
                data.get("MinAsk"),
                data.get("LastPrice"),
                data.get("Change"),
                data.get("CoinTradeValue"),
                data.get("UnitTradeValue")
            ))

        db = duckdb.connect(str(DB_PATH))

        db.executemany(
            """
            INSERT INTO market_data
            VALUES (
                to_timestamp(?),
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows
        )

        db.close()

        return len(rows)

    def run(self):
        print("Starting market collector...")

        while True:
            try:
                count = self.collect()

                print(
                    f"Collected {count} pairs | "
                    f"Database: {DB_PATH}"
                )

            except Exception as e:
                print(f"Collector error: {e}")

            time.sleep(self.interval)


if __name__ == "__main__":
    collector = MarketCollector(interval=10)
    collector.run()