import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "market.duckdb"

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
            rangeslider=dict(
                visible=True
            ),
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