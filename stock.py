
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from xgboost import XGBRegressor
import ta

# App title
st.title("Stock Market Prediction App")

# Sidebar inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Fetch data
@st.cache
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)
st.subheader(f"Raw Data for {ticker}")
st.write(data.head())

# Plot raw data
def plot_data(data):
    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Close"], label="Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig

st.subheader("Closing Price over Time")
st.pyplot(plot_data(data))

# Ensure "Close" is a 1D pandas Series
st.write(f"Type of 'Close': {type(data['Close'])}")  # Debugging: check type
st.write(f"Shape of 'Close': {data['Close'].shape}")  # Debugging: check shape

if isinstance(data["Close"], pd.DataFrame):
    close_prices = data["Close"].iloc[:, 0]
else:
    close_prices = data["Close"]

# Add technical indicators
st.subheader("Adding Technical Indicators")
try:
    data["SMA"] = ta.trend.sma_indicator(close_prices, window=14)
    data["RSI"] = ta.momentum.rsi(close_prices, window=14)
    data["MACD"] = ta.trend.macd_diff(close_prices)
    st.write("Technical indicators added successfully!")
    st.write(data[["Close", "SMA", "RSI", "MACD"]].head())
except Exception as e:
    st.error(f"Error adding technical indicators: {e}")
