import requests
import pandas as pd
import pandas_ta as ta

def get_closing_prices(df):
    # Extract closing prices
    closing_prices = pd.DataFrame({'close': [float(df["close"][i]) for i in range(len(df))]})
    pd.set_option('display.max_columns', None)

    # Calculate SMA
    closing_prices.ta.sma(length=5, append=True)

    # Calculate RSI
    closing_prices.ta.rsi(length=14, append=True)

    # Calculate Bollinger Band
    closing_prices.ta.bbands(length=3, append=True)

    # Calculate MACD
    closing_prices.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Calculate EMA
    closing_prices.ta.ema(length=3, append=True)

    return closing_prices
