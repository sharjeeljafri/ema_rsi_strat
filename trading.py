import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Streamlit app configuration
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# Title and description with emojis
st.title("📈 Trend-Following Trading Strategy Dashboard 🚀")
st.write("**EMA Crossover & RSI Signal Analysis** 🌟")

# Sidebar inputs with emojis
st.sidebar.header("⚙️ Strategy Parameters")
ticker = st.sidebar.text_input("Stock Ticker 🎯", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date 📅", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date 📅", datetime.now())
short_ema = st.sidebar.slider("Short EMA Period ⏳", 5, 50, 20)
long_ema = st.sidebar.slider("Long EMA Period ⏳", 20, 200, 50)
rsi_period = st.sidebar.slider("RSI Period 📊", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought Level 🔴", 50, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold Level 🟢", 10, 50, 30)

# Function to calculate indicators
def calculate_indicators(df, short_ema, long_ema, rsi_period):
    df['Short_EMA'] = df['Close'].ewm(span=short_ema, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_ema, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Signal'] = 0
    df['Signal'] = np.where((df['Short_EMA'] > df['Long_EMA']) & 
                           (df['RSI'] < rsi_overbought), 1, df['Signal'])
    df['Signal'] = np.where((df['Short_EMA'] < df['Long_EMA']) & 
                           (df['RSI'] > rsi_oversold), -1, df['Signal'])
    return df

# Function to calculate performance
def calculate_performance(df):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)
    total_return = (df['Strategy_Returns'] + 1).prod() - 1 if not df['Strategy_Returns'].empty else 0
    sharpe_ratio = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0
    max_drawdown = (df['Strategy_Returns'].cumsum().cummax() - df['Strategy_Returns'].cumsum()).max() if not df['Strategy_Returns'].empty else 0
    return total_return, sharpe_ratio, max_drawdown

# Main app
def main():
    st.write(f"Fetching data for **{ticker}** from {start_date} to {end_date}... ⏳")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data available for {ticker}. Please check the ticker or date range. ❌")
            return
        
        st.success(f"Data downloaded successfully! Rows: {len(data)} ✅")
        # Debug: Display raw OHLC data to check for issues
        st.write("Sample of raw OHLC data:")
        st.dataframe(data[['Open', 'High', 'Low', 'Close']].head())
        
        df = calculate_indicators(data, short_ema, long_ema, rsi_period)
        
        # Price chart with EMAs - Using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Close'], label='Price', color='green', linewidth=2)
        ax.plot(df.index, df['Short_EMA'], label=f'Short EMA ({short_ema})', color='blue', linewidth=2)
        ax.plot(df.index, df['Long_EMA'], label=f'Long EMA ({long_ema})', color='orange', linewidth=2)
        ax.set_title(f'{ticker} Price with EMA Crossovers')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # RSI Chart - Using Matplotlib
        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
        ax_rsi.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
        ax_rsi.axhline(y=rsi_overbought, color='red', linestyle='--', label='Overbought')
        ax_rsi.axhline(y=rsi_oversold, color='green', linestyle='--', label='Oversold')
        ax_rsi.set_title('Relative Strength Index (RSI)')
        ax_rsi.set_xlabel('Date')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.legend()
        ax_rsi.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_rsi)
        
        # Performance metrics with emojis
        total_return, sharpe_ratio, max_drawdown = calculate_performance(df)
        st.subheader("📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return 💰", f"{total_return:.2%}")
        col2.metric("Sharpe Ratio 📈", f"{sharpe_ratio:.2f}")
        col3.metric("Max Drawdown 📉", f"{max_drawdown:.2%}")
        
        # Trade signals with emojis
        st.subheader("🔔 Trade Signals")
        signals_df = df[df['Signal'] != 0][['Close', 'Short_EMA', 'Long_EMA', 'RSI', 'Signal']]
        signals_df['Signal'] = signals_df['Signal'].map({1: 'Buy 🟢', -1: 'Sell 🔴', 0: 'Hold ⚪'})
        st.dataframe(signals_df.tail(10).style.format({'Close': '${:.2f}', 
                                                       'Short_EMA': '${:.2f}', 
                                                       'Long_EMA': '${:.2f}', 
                                                       'RSI': '{:.2f}'}))

    except Exception as e:
        st.error(f"An error occurred: {str(e)} ❌")

if __name__ == "__main__":
    main()