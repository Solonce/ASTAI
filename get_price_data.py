import requests
import pandas as pd
import time
import numpy as np
import json

def get_price_data(get_index=False, num=10000000000):
    def get_trading_pairs():
        url = "https://api.kraken.com/0/public/AssetPairs"
        response = requests.get(url)
        return response.json()['result'].keys()

    def get_ohlc_data(pair):
        url = "https://api.kraken.com/0/public/OHLC"
        payload = {
            "pair": pair,
            "interval": 60  # 1-minute intervals
        }
        response = requests.get(url, params=payload)
        return pd.DataFrame(response.json()['result'][pair], 
                            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])

    # get a list of all trading pairs
    pairs = get_trading_pairs()

    # for each of the top 50 pairs, get the historical data
    data = []
    for i, pair in enumerate(pairs):
        if (get_index == False and pair[-3:] == 'USD') or (get_index == True and i == num):
            df = get_ohlc_data(pair)
            data.append(df)
            time.sleep(1)  # prevent rate limiting

    return data

def get_top_pairs(len, base_pair="USD"):
    def get_trading_pairs():
            url = "https://api.kraken.com/0/public/AssetPairs"
            response = requests.get(url)

            return response.json()['result']

    def get_sorted_vol(len):
        trading_pairs = get_trading_pairs()
        usd_pairs = {k: v for k, v in trading_pairs.items() if k[-3:] == base_pair}
        pairs_string = ','.join(usd_pairs.keys())
        response = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={pairs_string}")
        response.raise_for_status()
        tickers = response.json()["result"]
        sorted_pairs = sorted(tickers.items(), key=lambda x: float(x[1]['v'][1]), reverse=True)
        sorted_pairs = sorted_pairs[:len]
        #for i, (pair, data) in enumerate(sorted_pairs[:len]):
        #    print(f"{i+1}. {pair}: 24-hour volume = {data['v'][1]}")
        return [pair[0] for pair in sorted_pairs]

    return get_sorted_vol(len)

def get_ohlc_data(pair, json=False):
        url = "https://api.kraken.com/0/public/OHLC"
        payload = {
            "pair": pair,
            "interval": 60  # 1-minute intervals
        }
        response = requests.get(url, params=payload)
        df = pd.DataFrame(response.json()['result'][pair], 
                                columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        if not json:
            return df
        else:
            return df['close'].tolist()