import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "market.duckdb"
EQUITY_PATH = BASE_DIR / "backtest_equity.csv"
TRADES_PATH = BASE_DIR / "backtest_trades.csv"

st.set_page_config(
    page_title="Roostoo Trading Dashboard",
    layout="wide"
)

st.title("Roostoo Trading Dashboard")


@st.fragment(run_every="10s")
def dashboard():

    db = duckdb.connect(str(DB_PATH), read_only=True)

    latest = db.execute("""
        SELECT
            pair,
            bid,
            ask,
            last_price,
            change,
            coin_trade_value,
            unit_trade_value,
            timestamp
        FROM market_data
        WHERE timestamp = (
            SELECT MAX(timestamp)
            FROM market_data
        )
        ORDER BY pair
    """).df()

    pairs = db.execute("""
        SELECT DISTINCT pair
        FROM market_data
        ORDER BY pair
    """).df()["pair"].tolist()

    db.close()

    if latest.empty:
        st.warning("No market data available.")
        return

    st.metric("Pairs", len(latest))

    selected_pair = st.selectbox(
        "Select asset",
        pairs,
        index=pairs.index("BTC/USD") if "BTC/USD" in pairs else 0
    )

    db = duckdb.connect(str(DB_PATH), read_only=True)

    history = db.execute("""
    SELECT
        timestamp,
        last_price,
        bid,
        ask
    FROM market_data
    WHERE pair = ?
    ORDER BY timestamp
    """, [selected_pair]).df()

    db.close()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["last_price"],
            mode="lines",
            name="Last Price"
        )
    )

    fig.update_layout(
        title=f"{selected_pair} Price",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        uirevision=selected_pair,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"price_chart_{selected_pair}"
    )

    st.markdown("### Latest Market Data")

    st.dataframe(
        latest,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Last update: {latest['timestamp'].iloc[0]}"
    )


dashboard()

st.divider()
st.header("Backtest Results")

if not EQUITY_PATH.exists():
    st.warning("No backtest data found. Run backtester.py first.")
else:
    equity = pd.read_csv(EQUITY_PATH)
    equity["timestamp"] = pd.to_datetime(equity["timestamp"])

    trades = pd.DataFrame()

    if TRADES_PATH.exists():
        trades = pd.read_csv(TRADES_PATH)

        if "entry_time" in trades.columns:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"])

        if "exit_time" in trades.columns:
            trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    if not equity.empty:

        initial_capital = float(equity["capital"].iloc[0])
        final_capital = float(equity["capital"].iloc[-1])
        total_return = (final_capital / initial_capital - 1) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Initial Capital",
            f"${initial_capital:,.2f}"
        )

        c2.metric(
            "Final Capital",
            f"${final_capital:,.2f}"
        )

        c3.metric(
            "Return",
            f"{total_return:.2f}%"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity["timestamp"],
                y=equity["capital"],
                mode="lines",
                name="Equity"
            )
        )

        if not trades.empty:

            if "entry_time" in trades.columns:

                entry_times = trades["entry_time"].dropna()

                entry_caps = []

                for t in entry_times:
                    idx = (equity["timestamp"] - t).abs().argmin()
                    entry_caps.append(equity.iloc[idx]["capital"])

                fig.add_trace(
                    go.Scatter(
                        x=entry_times,
                        y=entry_caps,
                        mode="markers",
                        name="Entry",
                        marker=dict(
                            size=10,
                            symbol="triangle-up"
                        )
                    )
                )

            if "exit_time" in trades.columns:

                exit_times = trades["exit_time"].dropna()

                exit_caps = []

                for t in exit_times:
                    idx = (equity["timestamp"] - t).abs().argmin()
                    exit_caps.append(equity.iloc[idx]["capital"])

                fig.add_trace(
                    go.Scatter(
                        x=exit_times,
                        y=exit_caps,
                        mode="markers",
                        name="Exit",
                        marker=dict(
                            size=10,
                            symbol="triangle-down"
                        )
                    )
                )

        fig.update_layout(
            title="Backtest Equity Curve",
            xaxis_title="Time",
            yaxis_title="Capital",
            height=550,
            hovermode="x unified",
            xaxis=dict(
                type="date",
                rangeslider=dict(visible=True)
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="backtest_equity_chart"
        )

    if not trades.empty:

        st.markdown("### Backtest Trades")

        st.dataframe(
            trades,
            use_container_width=True,
            hide_index=True
        )