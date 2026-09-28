import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/market.duckdb"

INITIAL_CAPITAL = 50000
FEE = 0.0012
MIN_SCORE = 0.5
SWITCH_MARGIN = 0.2

def analyze_short_signal(df):
    data = df[[
        "timestamp",
        "pair",
        "score",          
        "mid"
    ]].copy()

    data = data.sort_values(["pair", "timestamp"])

    for minutes in [5, 10, 15]:

        results = []

        for pair, g in data.groupby("pair"):

            g = g.sort_values("timestamp").copy()

            future = g[[
                "timestamp",
                "mid"
            ]].copy()

            future["future_time"] = (
                future["timestamp"] -
                pd.Timedelta(minutes=minutes)
            )

            future = future.rename(
                columns={
                    "timestamp": "future_timestamp",
                    "mid": "future_mid"
                }
            )

            merged = pd.merge_asof(
                g,
                future.sort_values("future_time"),
                left_on="timestamp",
                right_on="future_time",
                direction="forward",
                tolerance=pd.Timedelta(seconds=15)
            )

            merged["future_return"] = (
                merged["future_mid"] /
                merged["mid"] - 1
            )

            results.append(merged)

        result = pd.concat(results)

        result = result.dropna(
            subset=["score", "future_return"]
        )

        result["decile"] = pd.qcut(
            result["score"],
            10,
            labels=False,
            duplicates="drop"
        ) + 1

        print(f"\nSHORT SIGNAL — {minutes} MINUTES")

        print(
            result.groupby("decile")["future_return"]
            .mean()
            .mul(100)
            .round(4)
        )

        corr = result["score"].corr(
            result["future_return"]
        )

        print(f"Correlation: {corr:.6f}")

        top = result[
            result["decile"] == 10
        ]["future_return"].mean()

        bottom = result[
            result["decile"] == 1
        ]["future_return"].mean()

        print(
            f"Top score future return: {top * 100:.4f}%"
        )

        print(
            f"Bottom score future return: {bottom * 100:.4f}%"
        )


def analyze_features(df):
    print("\n" + "=" * 60)
    print("INDIVIDUAL FEATURE PREDICTIVENESS")
    print("=" * 60)

    features = [
        "momentum_1m",
        "momentum_5m",
        "trend",
        "volatility",
        "spread"
    ]

    for minutes in [5, 10, 15]:

        rows = []

        for pair, g in df.groupby("pair"):

            g = g.sort_values("timestamp").copy()

            future = g[["timestamp", "mid"]].copy()

            future["future_time"] = (
                future["timestamp"]
                - pd.Timedelta(minutes=minutes)
            )

            merged = pd.merge_asof(
                g,
                future,
                left_on="timestamp",
                right_on="future_time",
                direction="forward",
                tolerance=pd.Timedelta(seconds=15),
                suffixes=("", "_future")
            )

            merged["future_return"] = (
                merged["mid_future"] / merged["mid"] - 1
            )

            rows.append(merged)

        data = pd.concat(rows)

        print(f"\n{minutes}-MINUTE FUTURE RETURN")
        print("-" * 60)

        for feature in features:

            valid = data[
                [feature, "future_return"]
            ].dropna()

            correlation = valid[feature].corr(
                valid["future_return"]
            )

            print(
                f"{feature:15s}: "
                f"{correlation:.6f}"
            )

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
        ORDER BY timestamp, pair
    """).fetchdf()

    db.close()

    return df


def add_features(df):

    df["mid"] = (df["bid"] + df["ask"]) / 2

    result = []

    for pair, g in df.groupby("pair"):

        g = g.sort_values("timestamp").copy()
        g = g.set_index("timestamp")

        price = g["mid"]

        past = g[["mid"]].reset_index()
        past = past.rename(
            columns={
                "timestamp": "past_timestamp",
                "mid": "past_mid"
            }
        )

        current = g.reset_index()

        current["target_1m"] = (
            current["timestamp"] -
            pd.Timedelta(minutes=1)
        )

        current["target_5m"] = (
            current["timestamp"] -
            pd.Timedelta(minutes=5)
        )

        m1 = pd.merge_asof(
            current[["timestamp", "target_1m"]],
            past,
            left_on="target_1m",
            right_on="past_timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=15)
        )

        m5 = pd.merge_asof(
            current[["timestamp", "target_5m"]],
            past,
            left_on="target_5m",
            right_on="past_timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=15)
        )

        g["momentum_1m"] = (
            current["mid"].values /
            m1["past_mid"].values - 1
        )

        g["momentum_5m"] = (
            current["mid"].values /
            m5["past_mid"].values - 1
        )

        returns = price.pct_change()

        g["volatility"] = (
            returns.rolling(20).std()
        )

        g["sma_20"] = (
            price.rolling(20).mean()
        )

        g["trend"] = (
            price / g["sma_20"] - 1
        )

        g["spread"] = (
            (g["ask"] - g["bid"]) / g["mid"]
        )

        g = g.reset_index()

        result.append(g)

    df = pd.concat(result, ignore_index=True)

    return df


def add_scores(df):

    def zscore(x):
        std = x.std()

        if std == 0 or pd.isna(std):
            return 0

        return (x - x.mean()) / std

    df["z_1m"] = (
        df.groupby("timestamp")["momentum_1m"]
        .transform(zscore)
    )

    df["z_5m"] = (
        df.groupby("timestamp")["momentum_5m"]
        .transform(zscore)
    )

    df["z_trend"] = (
        df.groupby("timestamp")["trend"]
        .transform(zscore)
    )

    df["z_vol"] = (
        df.groupby("timestamp")["volatility"]
        .transform(zscore)
    )

    df["z_spread"] = (
        df.groupby("timestamp")["spread"]
        .transform(zscore)
    )

    df["score"] = (
        0.35 * df["z_5m"] +
        0.25 * df["z_1m"] +
        0.20 * df["z_trend"] -
        0.10 * df["z_vol"] -
        0.10 * df["z_spread"]
    )

    return df

def analyze_score_predictiveness(df):
    print("\n" + "=" * 60)
    print("SCORE PREDICTIVENESS ANALYSIS")
    print("=" * 60)

    data = df[["timestamp", "pair", "mid", "score"]].dropna().copy()
    data = data.sort_values(["pair", "timestamp"])

    for minutes in [5, 10, 15]:

        future = data[["timestamp", "pair", "mid"]].copy()
        future["future_time"] = future["timestamp"] - pd.Timedelta(minutes=minutes)

        data_h = pd.merge_asof(
            data.sort_values("timestamp"),
            future.sort_values("future_time"),
            left_on="timestamp",
            right_on="future_time",
            by="pair",
            direction="forward",
            tolerance=pd.Timedelta(seconds=15),
            suffixes=("", "_future")
        )

        data_h["future_return"] = (
            data_h["mid_future"] / data_h["mid"] - 1
        )

        data_h = data_h.dropna(subset=["future_return"])

        data_h["score_bucket"] = pd.qcut(
            data_h["score"],
            10,
            labels=False,
            duplicates="drop"
        )

        result = (
            data_h.groupby("score_bucket")["future_return"]
            .mean()
            .sort_index()
        )

        print(f"\n{minutes}-MINUTE FUTURE RETURN")
        print("-" * 40)

        for bucket, ret in result.items():
            print(
                f"Decile {int(bucket) + 1}: "
                f"{ret * 100:.4f}%"
            )

        correlation = data_h["score"].corr(data_h["future_return"])

        print(f"Correlation: {correlation:.6f}")

        top = result.iloc[-1]
        bottom = result.iloc[0]

        print(
            f"Top - Bottom: "
            f"{(top - bottom) * 100:.4f}%"
        )

def backtest(df):

    capital = INITIAL_CAPITAL

    current_pair = None
    entry_price = None
    entry_time = None

    trades = []

    total_fees = 0

    equity = []

    timestamps = sorted(df["timestamp"].unique())

    for timestamp in timestamps:

        market = df[
            df["timestamp"] == timestamp
        ].dropna(subset=["score"])

        if market.empty:
            continue

        equity.append({
            "timestamp": timestamp,
            "capital": capital
        })

        best = market.loc[
            market["score"].idxmax()
        ]

        if current_pair is None:

            if best["score"] >= MIN_SCORE:

                current_pair = best["pair"]
                entry_price = best["ask"]
                entry_time = timestamp

                fee = capital * FEE
                capital -= fee
                total_fees += fee

            continue

        current = market[
            market["pair"] == current_pair
        ]

        if current.empty:
            continue

        current = current.iloc[0]

        holding_time = timestamp - entry_time

        if holding_time < pd.Timedelta(minutes=5):
            continue

        trade_return = current["bid"] / entry_price - 1

        stop_loss = trade_return <= -0.005

        take_profit = trade_return >= 0.01

        score_exit = current["score"] < 0

        switch_signal = (
            best["score"] >
            current["score"] + SWITCH_MARGIN
        )

        should_exit = (
            stop_loss or
            take_profit or
            score_exit or
            switch_signal
        )

        if should_exit:

            exit_price = current["bid"]

            trade_return = (
                exit_price / entry_price - 1
            )

            gross_pnl = capital * trade_return

            capital += gross_pnl

            fee = capital * FEE

            capital -= fee

            total_fees += fee

            trades.append({
                "pair": current_pair,
                "entry": entry_time,
                "exit": timestamp,
                "return": trade_return,
                "gross_pnl": gross_pnl,
                "holding_seconds": (
                    timestamp - entry_time
                ).total_seconds(),
                "capital": capital
            })

            current_pair = None
            entry_price = None
            entry_time = None

    if current_pair is not None and entry_time != timestamps[-1]:

        current = df[
            (df["timestamp"] == timestamps[-1]) &
            (df["pair"] == current_pair)
        ]

        if not current.empty:

            current = current.iloc[0]

            exit_price = current["bid"]

            trade_return = (
                exit_price / entry_price - 1
            )

            gross_pnl = capital * trade_return

            capital += gross_pnl

            fee = capital * FEE

            capital -= fee

            total_fees += fee

            trades.append({
                "pair": current_pair,
                "entry": entry_time,
                "exit": timestamps[-1],
                "return": trade_return,
                "gross_pnl": gross_pnl,
                "holding_seconds": (
                    timestamps[-1] - entry_time
                ).total_seconds(),
                "capital": capital
            })

    return capital, trades, total_fees, equity

def main():

    print("Loading market data...")

    df = load_data()

    print(f"Rows: {len(df):,}")
    print(f"Pairs: {df['pair'].nunique()}")

    print("\nCalculating timestamp-aware features...")

    df = add_features(df)

    print("Calculating scores...")

    df = add_scores(df)

    analyze_short_signal(df)

    print("Running backtest...")

    analyze_score_predictiveness(df)
    analyze_features(df)
    final_capital, trades, total_fees, equity = backtest(df)

    total_return = (
        final_capital / INITIAL_CAPITAL - 1
    )

    if trades:
        win_rate = (
            sum(t["return"] > 0 for t in trades)
            / len(trades)
        )

        avg_trade = np.mean(
            [t["return"] for t in trades]
        )
    else:
        win_rate = 0
        avg_trade = 0

    if trades:

        returns = np.array([
            t["return"] for t in trades
        ])

        winners = returns[returns > 0]
        losers = returns[returns <= 0]

        avg_win = np.mean(winners) if len(winners) else 0
        avg_loss = np.mean(losers) if len(losers) else 0

        profit_factor = (
            winners.sum() / abs(losers.sum())
            if len(losers) else float("inf")
        )

        avg_hold = np.mean([
            t["holding_seconds"]
            for t in trades
        ])

    print(f"Average winner : {avg_win:.4%}")
    print(f"Average loser  : {avg_loss:.4%}")
    print(f"Largest winner : {returns.max():.4%}")
    print(f"Largest loser  : {returns.min():.4%}")
    print(f"Profit factor  : {profit_factor:.3f}")
    print(f"Avg hold time  : {avg_hold / 60:.2f} minutes")
    print(f"Total fees     : ${total_fees:.2f}")

    print("\n========== BACKTEST ==========")
    print(f"Initial capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"Final capital   : ${final_capital:,.2f}")
    print(f"Total return    : {total_return:.3%}")
    print(f"Trades          : {len(trades)}")
    print(f"Win rate        : {win_rate:.2%}")
    print(f"Average trade   : {avg_trade:.4%}")

    equity_df = pd.DataFrame(equity)
    trades_df = pd.DataFrame(trades)

    equity_df.to_csv("backtest_equity.csv", index=False)
    trades_df.to_csv("backtest_trades.csv", index=False)

    print("\nSaved:")
    print("backtest_equity.csv")
    print("backtest_trades.csv")

    print("\nLast 10 trades:")

    for trade in trades[-10:]:
        print(
            trade["pair"],
            trade["entry"],
            trade["exit"],
            f"{trade['return']:.4%}",
            f"${trade['capital']:.2f}"
        )


if __name__ == "__main__":
    main()