import duckdb

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "market.duckdb"


def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


def latest():
    db = get_connection()

    result = db.execute("""
        SELECT *
        FROM market_data
        ORDER BY timestamp DESC
        LIMIT 10
    """).fetchall()

    db.close()

    return result


def latest_prices():
    db = get_connection()

    result = db.execute("""
        SELECT pair, last_price
        FROM market_data
        WHERE timestamp = (
            SELECT MAX(timestamp)
            FROM market_data
        )
        ORDER BY pair
    """).fetchall()

    db.close()

    return result

def price_history(pair, limit=100):
    db = get_connection()

    result = db.execute("""
        SELECT timestamp, last_price
        FROM market_data
        WHERE pair = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [pair, limit]).fetchall()

    db.close()

    return list(reversed(result))

if __name__ == "__main__":
    prices = price_history("BTC/USD")

    print("BTC prices:", len(prices))

    for price in prices[-10:]:
        print(price)