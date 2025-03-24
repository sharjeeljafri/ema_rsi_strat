import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Streamlit page configuration
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# Dark Mode CSS with Branding Styles
st.markdown("""
    <style>
    /* Global Dark Mode Styles */
    body {
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0;
        background-color: #1E1E1E;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    h1 {
        font-size: 2.5em;
        color: #4A90E2;
        font-weight: 700;
        margin-bottom: 0.5em;
    }
    h2 {
        font-size: 1.8em;
        color: #4A90E2;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .sidebar .sidebar-content {
        background-color: #2A2A2A;
        border-right: 1px solid #444444;
        padding: 20px;
    }
    .metric-card {
        background-color: #2A2A2A;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .chart-container {
        background-color: #2A2A2A;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stDataFrame {
        background-color: #2A2A2A;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 10px;
        color: #E0E0E0;
    }
    .stDataFrame table {
        color: #E0E0E0;
    }
    .stDataFrame th {
        background-color: #333333;
        color: #E0E0E0;
    }
    .stDataFrame td {
        background-color: #2A2A2A;
        color: #E0E0E0;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        color: #A0A0A0;
        font-size: 0.9em;
        border-top: 1px solid #444444;
        margin-top: 30px;
    }
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #333333;
        color: #E0E0E0;
    }
    /* Branding Styles */
    .top-branding {
        font-size: 1.2em;
        color: #BB86FC;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .sidebar-branding {
        display: flex;
        align-items: center;
        margin-top: 20px;
        font-size: 0.9em;
        color: #A0A0A0;
    }
    .sidebar-branding a {
        margin-left: 8px;
    }
    .sidebar-branding img {
        width: 24px;
        height: 24px;
        vertical-align: middle;
    }
    </style>
""", unsafe_allow_html=True)

# Top Left Branding
st.markdown('<div class="top-branding">Syed Sharjeel Jafri</div>', unsafe_allow_html=True)

# Title and description
st.title("📈 Trading Strategy Dashboard")
st.markdown("Analyze stocks using EMA Crossover and RSI signals with historical performance metrics.", unsafe_allow_html=True)

# Sidebar - Strategy Parameters
with st.sidebar:
    st.header("Strategy Parameters")
    ticker = st.text_input("Stock Ticker", "AAPL", help="Enter a valid stock ticker (e.g., AAPL)").upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365), help="Select the start date for historical data")
    with col2:
        end_date = st.date_input("End Date", datetime.now(), help="Select the end date for historical data")
    
    st.subheader("Indicator Settings")
    short_ema = st.slider("Short EMA Period", 5, 50, 20, help="Period for the short Exponential Moving Average")
    long_ema = st.slider("Long EMA Period", 20, 200, 50, help="Period for the long Exponential Moving Average")
    rsi_period = st.slider("RSI Period", 5, 30, 14, help="Period for the Relative Strength Index")
    rsi_overbought = st.slider("RSI Overbought Level", 50, 90, 70, help="Upper threshold for RSI (overbought)")
    rsi_oversold = st.slider("RSI Oversold Level", 10, 50, 30, help="Lower threshold for RSI (oversold)")
    
    st.subheader("Trading Mode")
    position_mode = st.radio("Position Mode", ["Flip on Signal", "Hold Until Exit"], help="Choose how positions are managed based on signals")
    
    # Sidebar Branding with LinkedIn Logo
    st.markdown(
        """
        <div class="sidebar-branding">
            Created by Syed Sharjeel Jafri
            <a href="https://www.linkedin.com/in/syed-sharjeel-jafri" target="_blank">
                <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" alt="LinkedIn">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    run_button = st.button("Run Analysis")

# Indicator Calculations
def calculate_indicators(df, short_ema, long_ema, rsi_period):
    df['Short_EMA'] = df['Close'].ewm(span=short_ema, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_ema, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=rsi_period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(span=rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs.fillna(1e10)))
    df['Signal'] = 0
    df.loc[(df['Short_EMA'] > df['Long_EMA']) & (df['RSI'] < rsi_overbought), 'Signal'] = 1
    df.loc[(df['Short_EMA'] < df['Long_EMA']) & (df['RSI'] > rsi_oversold), 'Signal'] = -1
    return df

# Position Generation
def generate_positions(signal_series, mode="Hold Until Exit"):
    if mode == "Flip on Signal":
        return signal_series.shift(1)
    elif mode == "Hold Until Exit":
        position = []
        last_signal = 0
        for signal in signal_series:
            if signal != 0:
                last_signal = signal
            position.append(last_signal)
        return pd.Series(position, index=signal_series.index).shift(1)

# Performance Metrics
def calculate_performance(df, positions):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * positions
    total_return = (df['Strategy_Returns'] + 1).prod() - 1
    sharpe_ratio = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else np.nan
    max_drawdown = (df['Strategy_Returns'].cumsum().cummax() - df['Strategy_Returns'].cumsum()).max()
    trades = df['Signal'][df['Signal'] != 0]
    num_trades = len(trades)
    trade_returns = df['Strategy_Returns'][df['Signal'] != 0]
    win_rate = (trade_returns > 0).sum() / num_trades if num_trades > 0 else np.nan
    num_days = (df.index[-1] - df.index[0]).days
    cagr = (1 + total_return) ** (365 / num_days) - 1 if num_days > 0 else 0
    return total_return, sharpe_ratio, max_drawdown, num_trades, win_rate, cagr

# Price Chart with Matplotlib (Dark Mode)
def plot_price_chart(df):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#2A2A2A')
    ax.set_facecolor('#2A2A2A')
    ax.plot(df.index, df['Close'], label='Price', color='#4A90E2', linewidth=2)
    ax.plot(df.index, df['Short_EMA'], label=f'Short EMA ({short_ema})', color='#FF6F61', linewidth=1.5, linestyle='--')
    ax.plot(df.index, df['Long_EMA'], label=f'Long EMA ({long_ema})', color='#BB86FC', linewidth=1.5, linestyle='--')
    buys = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]
    ax.scatter(buys.index, buys['Close'], label='Buy Signal', marker='^', color='#03DAC6', s=100)
    ax.scatter(sells.index, sells['Close'], label='Sell Signal', marker='v', color='#CF6679', s=100)
    ax.set_title(f'{ticker} Price with EMA Crossovers', fontsize=14, fontweight='bold', color='#E0E0E0')
    ax.set_xlabel('Date', fontsize=12, color='#E0E0E0')
    ax.set_ylabel('Price ($)', fontsize=12, color='#E0E0E0')
    ax.legend(facecolor='#333333', edgecolor='#444444', labelcolor='#E0E0E0')
    ax.grid(True, linestyle='--', alpha=0.3, color='#E0E0E0')
    ax.tick_params(axis='x', colors='#E0E0E0', rotation=45)
    ax.tick_params(axis='y', colors='#E0E0E0')
    plt.tight_layout()
    return fig

# RSI Chart with Matplotlib (Dark Mode)
def plot_rsi_chart(df):
    fig, ax = plt.subplots(figsize=(12, 3), facecolor='#2A2A2A')
    ax.set_facecolor('#2A2A2A')
    ax.plot(df.index, df['RSI'], label='RSI', color='#BB86FC', linewidth=2)
    ax.axhline(y=rsi_overbought, color='#CF6679', linestyle='--', label='Overbought', alpha=0.7)
    ax.axhline(y=rsi_oversold, color='#03DAC6', linestyle='--', label='Oversold', alpha=0.7)
    ax.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold', color='#E0E0E0')
    ax.set_xlabel('Date', fontsize=12, color='#E0E0E0')
    ax.set_ylabel('RSI', fontsize=12, color='#E0E0E0')
    ax.legend(facecolor='#333333', edgecolor='#444444', labelcolor='#E0E0E0')
    ax.grid(True, linestyle='--', alpha=0.3, color='#E0E0E0')
    ax.tick_params(axis='x', colors='#E0E0E0', rotation=45)
    ax.tick_params(axis='y', colors='#E0E0E0')
    plt.tight_layout()
    return fig

# Data Fetching with Retry and Fallback
def fetch_data_with_retry(ticker, start, end, retries=3, delay=5):
    for attempt in range(retries):
        try:
            with st.spinner("Fetching market data..."):
                data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                return data
            else:
                raise ValueError("Empty data returned")
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Failed to fetch data for {ticker} (Attempt {attempt + 1}/{retries}): {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"Failed to fetch live data for {ticker} after {retries} attempts: {str(e)}")
                return None

# Main Logic
def main():
    if not run_button:
        st.markdown("**Set your parameters and click 'Run Analysis' to begin.**", unsafe_allow_html=True)
        return

    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
        return

    # Attempt to fetch live data
    data = fetch_data_with_retry(ticker, start_date, end_date)
    
    # If live data fetch fails, fall back to static data
    if data is None:
        st.warning(f"Unable to fetch live data for {ticker}. Falling back to static data for AAPL...")
        try:
            data = pd.read_csv("aapl_data.csv", index_col="Date", parse_dates=True)
            st.info("Using static data for AAPL from March 22, 2023, to March 22, 2025.")
        except Exception as e:
            st.error(f"Failed to load static data: {str(e)}. Please ensure aapl_data.csv is available in the repository.")
            return
    
    if data.empty:
        st.error(f"No data available for {ticker}. Please check the ticker or date range.")
        return
    
    st.success(f"Data loaded successfully! Rows: {len(data)}")
    # Debug: Display raw OHLC data
    st.write("Sample of Raw Data:")
    st.dataframe(data[['Open', 'High', 'Low', 'Close']].head())
    
    df = calculate_indicators(data.copy(), short_ema, long_ema, rsi_period)
    positions = generate_positions(df['Signal'], mode=position_mode)
    total_return, sharpe_ratio, max_drawdown, num_trades, win_rate, cagr = calculate_performance(df, positions)

    # Charts
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.pyplot(plot_price_chart(df))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.pyplot(plot_rsi_chart(df))
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Return", f"{total_return:.2%}", help="Cumulative return of the strategy")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A", help="Risk-adjusted return")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}", help="Maximum loss from peak to trough")
        st.markdown('</div>', unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Number of Trades", f"{num_trades}", help="Total trades executed")
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Win Rate", f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A", help="Percentage of winning trades")
        st.markdown('</div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("CAGR", f"{cagr:.2%}", help="Compound Annual Growth Rate")
        st.markdown('</div>', unsafe_allow_html=True)

    # Signals Table
    st.subheader("Trade Signals")
    signal_df = df[df['Signal'] != 0][['Close', 'Short_EMA', 'Long_EMA', 'RSI', 'Signal']].tail(10).copy()
    signal_df['Signal'] = signal_df['Signal'].map({1: 'Buy', -1: 'Sell'})
    st.dataframe(signal_df.style.format({'Close': '${:.2f}', 'Short_EMA': '${:.2f}', 'Long_EMA': '${:.2f}', 'RSI': '{:.2f}'}))

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown(
        """
        **Created by Syed Sharjeel Jafri**  
        Connect with me on [LinkedIn](https://www.linkedin.com/in/syed-sharjeel-jafri)
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
