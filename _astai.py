import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from stock_trading_env import StockTradingEnv
from dqn_agent import DQNAgent
from price_grabber import get_closing_prices
from collections import deque
from get_price_data import get_price_data, get_top_pairs, get_ohlc_data


def generate_data(prices):
    data = get_closing_prices(prices)

    # Generate a fake stock price history
    close = data['close'].values.tolist()
    sma = data['SMA_5'].values.tolist()
    rsi = data['RSI_14'].values.tolist()
    bbl = data['BBL_3_2.0'].values.tolist()
    bbm = data['BBM_3_2.0'].values.tolist()
    bbu = data['BBU_3_2.0'].values.tolist()
    bbb = data['BBB_3_2.0'].values.tolist()
    bbp = data['BBP_3_2.0'].values.tolist()
    macd = data['MACD_12_26_9'].values.tolist()
    macdh = data['MACDh_12_26_9'].values.tolist()
    macds = data['MACDs_12_26_9'].values.tolist()
    ema = data['EMA_3'].values.tolist()

    # Combine the price history and SMA 30 into a 2D array
    data = np.array([close, rsi, bbl, bbm, bbu]).T
    
    return data

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_box(content_list):
    width = max(len(line) for line in content_list)
    print("╔" + "═" * (width + 2) + "╗")
    for line in content_list:
        print(f"║ {line.ljust(width)} ║")
    print("╚" + "═" * (width + 2) + "╝")

def main_menu():
    while True:
        clear_console()
        draw_box([
            "Welcome to the CLI Interface",
            "",
            "1. Act on the newest data",
            "2. View model data",
            "3. Exit"
        ])
        choice = input("Enter your choice: ")
        if choice == "1":
            act_on_data()
        elif choice == "2":
            view_model_data()
        elif choice == "3":
            break

def act_on_data():
    clear_console()
    pair = get_top_pairs(1)[0]
    window_size = 50
    stock_data = generate_data(get_ohlc_data(get_top_pairs(1)[0]))[-window_size:]
    agent = DQNAgent(3, window_size, model_name='main.h5')
    action = agent.act(stock_data, prediction=True)

    draw_box([
        "Acting on Newest Data",
        "",
        f"Action for {pair} is {action}",
        ""
        "Press any key to return..."
    ])
    input()

def view_model_data():
    clear_console()

    # Load the data from the json file
    f = open('iteration_data.json')
    data = json.load(f)
    step = data['step']
    current_pair = data['current_pair']
    net_rewards = data['loss_avg']

    draw_box([
        "Model Data",
        "",
        f"Step: {step}",
        f"Current Pair: {current_pair}",
        f"Loss: {net_rewards}",
        "",
        "Press any key to return..."
    ])
    input()

if __name__ == '__main__':
    main_menu()
