import duckdb
import pandas as pd
import numpy as np
from itertools import combinations

try:
    from statsmodels.tsa.stattools import coint
except ImportError:
    coint = None


DB_PATH = "data/market.duckdb"

INITIAL_CAPITAL = 50000.0
FEE = 0.001

BAR = "1min"
MAX_FFILL = 3

TRAIN_FRACTION = 0.60

MIN_CORRELATION = 0.75
MAX_PVALUE = 0.10
MAX_PAIRS = 8

LOOKBACK_Z = 120
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5

MAX_HOLD_MINUTES = 180

PAIR_CAPITAL_FRACTION = 0.20


def load_data():
    db = duckdb.connect(DB_PATH, read_only=True)

    df = db.execute("""
        SELECT
            timestamp,
            pair,
            bid,
            ask,
            last_price
        FROM market_data
        WHERE bid > 0
          AND ask > 0
          AND last_price > 0
        ORDER BY timestamp, pair
    """).fetchdf()

    db.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique()}")
    print(
        f"Time range: {df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )

    return df


def normalize_pair_name(name):
    return (
        str(name)
        .upper()
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def prepare_prices(df):
    print("\nBuilding synchronized price matrix...")

    df = df.copy()
    df["pair_clean"] = df["pair"].map(normalize_pair_name)

    duplicate_count = (
        df.groupby(["timestamp", "pair_clean"])
        .size()
        .gt(1)
        .sum()
    )

    if duplicate_count:
        df = (
            df.sort_values("timestamp")
            .groupby(["timestamp", "pair_clean"], as_index=False)
            .last()
        )

    prices = df.pivot_table(
        index="timestamp",
        columns="pair_clean",
        values="last_price",
        aggfunc="last"
    )

    prices = prices.sort_index()

    prices = prices.resample(BAR).last()

    prices = prices.ffill(limit=MAX_FFILL)

    valid = prices.notna().sum()

    keep = valid[valid >= int(len(prices) * 0.70)].index

    prices = prices[keep]

    print(f"Synchronized bars: {len(prices):,}")
    print(f"Usable pairs: {len(prices.columns)}")

    return prices


def correlation_candidates(train):
    print("\nFinding correlated candidate pairs...")

    returns = np.log(train).diff()

    corr = returns.corr()

    candidates = []

    cols = list(train.columns)

    for a, b in combinations(cols, 2):
        value = corr.loc[a, b]

        if pd.isna(value):
            continue

        if value >= MIN_CORRELATION:
            candidates.append((a, b, value))

    candidates.sort(key=lambda x: x[2], reverse=True)

    print(f"Candidate pairs: {len(candidates):,}")

    return candidates


def hedge_ratio(a, b):
    x = np.log(b).values
    y = np.log(a).values

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 100:
        return None

    x = np.column_stack([np.ones(len(x)), x])

    beta = np.linalg.lstsq(x, y, rcond=None)[0][1]

    return float(beta)


def discover_pairs(train):
    print("\n" + "=" * 65)
    print("PAIR DISCOVERY")
    print("=" * 65)

    candidates = correlation_candidates(train)

    if not candidates:
        raise RuntimeError(
            "No sufficiently correlated pairs were found. "
            "Lower MIN_CORRELATION."
        )

    discovered = []

    for a, b, correlation in candidates:
        sample = train[[a, b]].dropna()

        if len(sample) < 200:
            continue

        beta = hedge_ratio(sample[a], sample[b])

        if beta is None or not np.isfinite(beta):
            continue

        spread = (
            np.log(sample[a]) -
            beta * np.log(sample[b])
        )

        if coint is not None:
            try:
                _, pvalue, _ = coint(
                    np.log(sample[a]),
                    np.log(sample[b])
                )
            except Exception:
                pvalue = 1.0
        else:
            pvalue = np.nan

        spread_std = spread.std()

        if spread_std <= 0 or not np.isfinite(spread_std):
            continue

        if coint is not None and pvalue > MAX_PVALUE:
            continue

        discovered.append({
            "a": a,
            "b": b,
            "correlation": correlation,
            "beta": beta,
            "pvalue": pvalue,
            "spread_std": spread_std
        })

    discovered.sort(
        key=lambda x: (
            x["pvalue"] if np.isfinite(x["pvalue"]) else 1.0,
            -x["correlation"]
        )
    )

    discovered = discovered[:MAX_PAIRS]

    print(f"\nSelected pairs: {len(discovered)}")

    for i, pair in enumerate(discovered, 1):
        pvalue = pair["pvalue"]

        if np.isfinite(pvalue):
            pv = f"{pvalue:.4f}"
        else:
            pv = "N/A"

        print(
            f"{i:2d}. "
            f"{pair['a']} / {pair['b']} | "
            f"corr={pair['correlation']:.3f} | "
            f"beta={pair['beta']:.3f} | "
            f"p={pv}"
        )

    if not discovered:
        if coint is None:
            raise RuntimeError(
                "statsmodels is not installed. "
                "Install it with: pip install statsmodels"
            )

        raise RuntimeError(
            "No cointegrated pairs passed the filters. "
            "This dataset may not contain enough pair-trading alpha."
        )

    return discovered


def make_spread(test, train, pair):
    a = pair["a"]
    b = pair["b"]
    beta = pair["beta"]

    combined = pd.concat(
        [
            train[[a, b]],
            test[[a, b]]
        ]
    )

    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.ffill()

    spread = (
        np.log(combined[a]) -
        beta * np.log(combined[b])
    )

    mean = spread.rolling(
        LOOKBACK_Z,
        min_periods=LOOKBACK_Z
    ).mean()

    std = spread.rolling(
        LOOKBACK_Z,
        min_periods=LOOKBACK_Z
    ).std()

    z = (spread - mean) / std

    return combined, spread, z


def get_quotes(df):
    df = df.copy()
    df["pair_clean"] = df["pair"].map(normalize_pair_name)

    quotes = df.pivot_table(
        index="timestamp",
        columns="pair_clean",
        values=["bid", "ask"],
        aggfunc="last"
    )

    return quotes.sort_index()


def execute_pair(capital, pair, quotes, timestamp, direction):
    a = pair["a"]
    b = pair["b"]

    if (a, "bid") not in quotes.columns:
        return None

    if (a, "ask") not in quotes.columns:
        return None

    if (b, "bid") not in quotes.columns:
        return None

    if (b, "ask") not in quotes.columns:
        return None

    row = quotes.loc[timestamp]

    values = [
        row[(a, "bid")],
        row[(a, "ask")],
        row[(b, "bid")],
        row[(b, "ask")]
    ]

    if any(pd.isna(x) or x <= 0 for x in values):
        return None

    allocation = capital * PAIR_CAPITAL_FRACTION

    half = allocation / 2.0

    beta = abs(pair["beta"])

    if beta <= 0:
        return None

    if direction == 1:
        a_side = "LONG"
        b_side = "SHORT"

        a_entry = row[(a, "ask")]
        b_entry = row[(b, "bid")]
    else:
        a_side = "SHORT"
        b_side = "LONG"

        a_entry = row[(a, "bid")]
        b_entry = row[(b, "ask")]

    a_qty = half / a_entry
    b_qty = half / b_entry

    return {
        "direction": direction,
        "a_side": a_side,
        "b_side": b_side,
        "a_entry": a_entry,
        "b_entry": b_entry,
        "a_qty": a_qty,
        "b_qty": b_qty,
        "capital_used": allocation,
        "entry_fee": allocation * FEE
    }


def close_pair(position, pair, quotes, timestamp):
    a = pair["a"]
    b = pair["b"]

    if timestamp not in quotes.index:
        return None

    row = quotes.loc[timestamp]

    needed = [
        (a, "bid"),
        (a, "ask"),
        (b, "bid"),
        (b, "ask")
    ]

    if any(x not in quotes.columns for x in needed):
        return None

    values = [row[x] for x in needed]

    if any(pd.isna(x) or x <= 0 for x in values):
        return None

    if position["direction"] == 1:
        a_exit = row[(a, "bid")]
        b_exit = row[(b, "ask")]

        a_pnl = (
            a_exit - position["a_entry"]
        ) * position["a_qty"]

        b_pnl = (
            position["b_entry"] - b_exit
        ) * position["b_qty"]

    else:
        a_exit = row[(a, "ask")]
        b_exit = row[(b, "bid")]

        a_pnl = (
            position["a_entry"] - a_exit
        ) * position["a_qty"]

        b_pnl = (
            b_exit - position["b_entry"]
        ) * position["b_qty"]

    gross_pnl = a_pnl + b_pnl

    exit_notional = (
        abs(position["a_qty"] * a_exit) +
        abs(position["b_qty"] * b_exit)
    )

    exit_fee = exit_notional * FEE

    total_fee = position["entry_fee"] + exit_fee

    net_pnl = gross_pnl - total_fee

    return {
        "gross_pnl": gross_pnl,
        "fees": total_fee,
        "net_pnl": net_pnl,
        "a_exit": a_exit,
        "b_exit": b_exit
    }


def backtest(df, prices, pairs):
    print("\n" + "=" * 65)
    print("PAIR TRADING BACKTEST")
    print("=" * 65)

    quotes = get_quotes(df)

    test_start = prices.index[
        int(len(prices) * TRAIN_FRACTION)
    ]

    test_prices = prices.loc[test_start:]

    pair_states = []

    for pair_id, pair in enumerate(pairs):
        _, _, z = make_spread(
            test_prices,
            prices.loc[:test_start],
            pair
        )

        z = z.loc[test_prices.index]

        pair_states.append({
            "id": pair_id,
            "pair": pair,
            "z": z
        })

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for timestamp in test_prices.index:
        # ------------------------------------------------------------
        # If we have a position, manage only that position.
        # ------------------------------------------------------------
        if position is not None:
            state = pair_states[position["pair_id"]]
            pair = state["pair"]

            z_value = state["z"].get(timestamp, np.nan)

            if pd.isna(z_value):
                continue

            hold = (
                timestamp - position["entry_time"]
            ).total_seconds() / 60.0

            should_exit = (
                abs(z_value) <= EXIT_Z
                or abs(z_value) >= STOP_Z
                or hold >= MAX_HOLD_MINUTES
            )

            if not should_exit:
                continue

            closed = close_pair(
                position,
                pair,
                quotes,
                timestamp
            )

            if closed is None:
                continue

            capital += closed["net_pnl"]

            trades.append({
                "pair": f"{pair['a']}/{pair['b']}",
                "direction": (
                    "LONG_SPREAD"
                    if position["direction"] == 1
                    else "SHORT_SPREAD"
                ),
                "entry": position["entry_time"],
                "exit": timestamp,
                "entry_z": position["entry_z"],
                "exit_z": z_value,
                "hold_minutes": hold,
                "gross_pnl": closed["gross_pnl"],
                "fees": closed["fees"],
                "net_pnl": closed["net_pnl"],
                "return": (
                    closed["net_pnl"] /
                    position["capital_used"]
                ),
                "capital": capital
            })

            position = None

            continue

        # ------------------------------------------------------------
        # No position: find the strongest available signal.
        # We choose the pair with the largest absolute z-score.
        # ------------------------------------------------------------
        signals = []

        for state in pair_states:
            z_value = state["z"].get(timestamp, np.nan)

            if pd.isna(z_value):
                continue

            if abs(z_value) < ENTRY_Z:
                continue

            signals.append(
                (
                    abs(z_value),
                    state["id"],
                    z_value
                )
            )

        if not signals:
            continue

        signals.sort(
            key=lambda x: x[0],
            reverse=True
        )

        _, pair_id, z_value = signals[0]

        pair = pairs[pair_id]

        direction = (
            1 if z_value <= -ENTRY_Z
            else -1
        )

        opened = execute_pair(
            capital,
            pair,
            quotes,
            timestamp,
            direction
        )

        if opened is None:
            continue

        position = {
            **opened,
            "entry_time": timestamp,
            "pair_id": pair_id,
            "entry_z": z_value
        }

    # ------------------------------------------------------------
    # Close any remaining position at the end of the test.
    # ------------------------------------------------------------
    if position is not None:
        state = pair_states[position["pair_id"]]
        pair = state["pair"]

        last_timestamp = test_prices.index[-1]

        closed = close_pair(
            position,
            pair,
            quotes,
            last_timestamp
        )

        if closed is not None:
            capital += closed["net_pnl"]

            hold = (
                last_timestamp -
                position["entry_time"]
            ).total_seconds() / 60.0

            z_value = state["z"].get(
                last_timestamp,
                np.nan
            )

            trades.append({
                "pair": f"{pair['a']}/{pair['b']}",
                "direction": (
                    "LONG_SPREAD"
                    if position["direction"] == 1
                    else "SHORT_SPREAD"
                ),
                "entry": position["entry_time"],
                "exit": last_timestamp,
                "entry_z": position["entry_z"],
                "exit_z": z_value,
                "hold_minutes": hold,
                "gross_pnl": closed["gross_pnl"],
                "fees": closed["fees"],
                "net_pnl": closed["net_pnl"],
                "return": (
                    closed["net_pnl"] /
                    position["capital_used"]
                ),
                "capital": capital
            })

    return capital, trades


def print_results(capital, trades):
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    total_return = (
        capital / INITIAL_CAPITAL - 1
    )

    print(f"Initial capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"Final capital   : ${capital:,.2f}")
    print(f"Total return    : {total_return:.3%}")
    print(f"Trades          : {len(trades)}")

    if not trades:
        print("\nNo trades were generated.")
        print(
            "This is useful information: the selected pairs "
            "did not produce enough mean-reversion signals."
        )
        return

    returns = np.array([
        x["return"] for x in trades
    ])

    net = np.array([
        x["net_pnl"] for x in trades
    ])

    winners = returns[returns > 0]
    losers = returns[returns <= 0]

    win_rate = (
        len(winners) / len(returns)
    )

    profit_factor = (
        winners.sum() /
        abs(losers.sum())
        if len(losers)
        else float("inf")
    )

    avg_trade = returns.mean()

    avg_win = (
        winners.mean()
        if len(winners)
        else 0
    )

    avg_loss = (
        losers.mean()
        if len(losers)
        else 0
    )

    total_fees = sum(
        x["fees"] for x in trades
    )

    print(f"Win rate        : {win_rate:.2%}")
    print(f"Average trade   : {avg_trade:.4%}")
    print(f"Average winner  : {avg_win:.4%}")
    print(f"Average loser   : {avg_loss:.4%}")
    print(f"Profit factor   : {profit_factor:.3f}")
    print(f"Total fees      : ${total_fees:,.2f}")
    print(
        f"Average hold    : "
        f"{np.mean([x['hold_minutes'] for x in trades]):.2f} min"
    )

    equity = INITIAL_CAPITAL + np.cumsum(net)

    peak = np.maximum.accumulate(
        np.insert(equity, 0, INITIAL_CAPITAL)
    )[1:]

    drawdown = equity / peak - 1

    print(f"Max drawdown    : {drawdown.min():.3%}")

    print("\nLast 15 trades:")

    for trade in trades[-15:]:
        print(
            trade["pair"],
            trade["direction"],
            trade["entry"],
            "->",
            trade["exit"],
            f"z {trade['entry_z']:.2f} -> "
            f"{trade['exit_z']:.2f}",
            f"net={trade['net_pnl']:+.2f}",
            f"capital=${trade['capital']:.2f}"
        )


def main():
    print("=" * 65)
    print("CRYPTO STATISTICAL ARBITRAGE BACKTEST")
    print("=" * 65)

    if coint is None:
        print(
            "\nERROR: statsmodels is required.\n"
            "Run:\n"
            "pip install statsmodels"
        )
        return

    df = load_data()

    prices = prepare_prices(df)

    if len(prices) < 500:
        raise RuntimeError(
            "Not enough synchronized bars for this backtest."
        )

    split = int(
        len(prices) * TRAIN_FRACTION
    )

    train = prices.iloc[:split].copy()
    test = prices.iloc[split:].copy()

    print("\nWalk-forward split:")
    print(f"Training bars : {len(train):,}")
    print(f"Testing bars  : {len(test):,}")
    print(f"Train end     : {train.index[-1]}")
    print(f"Test start    : {test.index[0]}")

    pairs = discover_pairs(train)

    final_capital, trades = backtest(
        df,
        prices,
        pairs
    )

    print_results(
        final_capital,
        trades
    )


if __name__ == "__main__":
    main()
