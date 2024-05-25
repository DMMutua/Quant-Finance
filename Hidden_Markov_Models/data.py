import numpy as np
import pandas as pd
import yfinance as yf

def load_data(ticker="SPY"):
    """
    Loads Stock Price Action Data for a Specified Ticker Symbol
    Default Data Loaded is S&P 500 Data
    """
    data = yf.download("SPY")
    return data

def preview_data(data):
    """Visualizes the Head and Tail of the Data
    Dataframe"""
    print(data.info())
    print("DataFrame Head: ")
    print(data.head())
    print("Dataframe Tail: ")
    print(data.tail())

def engineer_features(data):
    """
    Input - price action Dataframe for a Ticker symbol.
    Computes the Following Features;
        Range - Range of the Returns for each datetime entry
        Returns - Returns and Log Returns for each datetime entry
        Price - 20-day M.A and for Prices and Close Price for each datetime entry
        Volatility - 20-day M.A and log for Volatility for each datetime entry
    Returns DF `features` with combined engineered features.
    """
    log_returns = np.log(data.Close / data.Close.shift(1))
    log_returns.name = "log_returns"

    range = (data.High - data.Low)
    range.name = "range"

    ma20_close = data.Close.rolling(window=20).mean()
    ma20_close.name = "ma20_close"
    ma50_close = data.Close.rolling(window=50).mean()
    ma50_close.name = "ma50_close"

    volatility20 = log_returns.rolling(window=20).std()
    volatility50 = log_returns.rolling(window=50).std()
    volatility20.name = "volatility20"
    volatility50.name = "volatility50"

    ma20_volatility = volatility20.rolling(window=20).mean()
    ma50_volatility = volatility50.rolling(window=50).mean()
    ma20_volatility.name = "ma20_volatility"
    ma50_volatility.name = "ma50_volatility"
    
    #Feature Combination in Single Dataframe
    features = pd.concat(
        [
            log_returns, range, ma20_close, ma50_close,
            volatility20, volatility50, ma20_volatility,
            ma50_volatility
        ], axis=1
    ).dropna()
    return features

spy = load_data("SPY")
ftrs = engineer_features(spy)
preview_data(ftrs)
