import numpy as np
import random
import tensorflow as tf
import sys
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from price_grabber import get_closing_prices
from get_price_data import get_price_data, get_top_pairs, get_ohlc_data
import os
from stock_trading_env import StockTradingEnv
import json
from dqn_agent import DQNAgent
import gc
import tracemalloc
import gym
from gym import spaces
from gym.utils import seeding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_env(rank, data, window_size, seed=1, current_step=0):
    def _init():
    	env = StockTradingEnv(data, window_size, current_step=current_step)
    	env.seed(seed+rank)
    	env = Monitor(env)
    	return env
    return _init

def map_values(x, _min, _max):
    return (x - _min) / (_max - _min)

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
    print("Loaded Data")
    use_set = [close, rsi, bbl, bbm, bbu]
    use_set = [[map_values(x, np.nanmin(_set), np.nanmax(_set)) for x in _set] for _set in use_set]
    print("Normalized Data")

    # Combine the price history and SMA 30 into a 2D array
    data = np.array(use_set).T
    return data

def write_data(len):
    data = {}
    pairs = get_top_pairs(len)
    for i, pair in enumerate(pairs):
        gen_data = generate_data(get_ohlc_data(pair))
        if gen_data.size == 3600:
            print(f"Generating Data Step {i}")
            gen_data = np.array([sub_array for sub_array in gen_data if not np.isnan(sub_array).any()])
            data[pair] = gen_data.tolist()
    with open("data.json", "w") as outfile:
        json.dump(data, outfile, indent = 4)

def update_iteration_data(data):
    with open("iteration_data.json", "w") as outfile:
        json.dump(data, outfile, indent = 4)

# Main loop
if __name__ == "__main__":
    print("Started")
    if not os.path.isfile('data.json'):
    	write_data(50)

    f = open('data.json')
    price_data = json.load(f)
    print("Loaded Data")
    model_name = "main.h5"
    window_size = 50

    if not os.path.isfile(model_name):
        agent = DQNAgent(3, window_size)
        sorted_pairs = list(price_data.keys())
    else:
        f = open('iteration_data.json')
        data = json.load(f)
        agent = DQNAgent(3, window_size, is_model=True, current_iter=data['current_pair'], current_step=data['step'], model_name=model_name, loss=float(data['loss_avg']), epsilon=float(data['epsilon']), learning_rate=float(data['learning_rate']))
        agent.current_pair = data['current_pair']
        sorted_pairs = list(price_data.keys())[agent.current_pair:]
    print("Loaded Agent")
    for i, pair in enumerate(sorted_pairs):
        stock_price_data_np = price_data[pair]

        num_processes = 2
        if not os.path.isfile(model_name):
        	print("No model found")
        	envs = SubprocVecEnv([make_env(i, stock_price_data_np, window_size) for i in range(num_processes)])
        else:
        	print("Model Found")
        	envs = SubprocVecEnv([make_env(i, stock_price_data_np, window_size, current_step=agent.step) for i in range(num_processes)])	
        print("Loaded Subprocesses")
        episodes = 10
        batch_size = 32

        for e in range(episodes):
            states = envs.reset()
            for time in range(1000):
                states = [state[time:window_size+(time)] for state in states]
                actions = [agent.act(state) for state in states]
                next_states, rewards, dones, infos = envs.step(actions)
                [agent.remember(state, action, reward, done, n_reward) for state, action, reward, done, n_reward in zip(states, actions, rewards, dones, [info['n_rewards'] for info in infos])]
                states = next_states
                agent.step = infos[0]['step']
                agent.current_pair = i
                [print(f"episode: {e}/{episodes}, action: {action}, reward: {np.round(reward, 2)}, net reward: {np.round(info['net reward'], 2)} score: {agent.step}, e: {agent.epsilon}, done: {done}, open orders: {info['orders']}") for action, done, reward, info in zip(actions, dones, rewards, infos)]
                if True in dones or agent.step>=(len(stock_price_data_np)-window_size):
                    update_iteration_data({"step": 0, "current_pair": agent.current_iter, "Net Rewards": [info['net reward'] for info in infos], "loss": str(agent.loss), "loss_avg": str(agent.loss_avg), "epsilon": str(agent.epsilon), "learning_rate": str(agent.learning_rate)})
                    break
                else:
                    update_iteration_data({"step": infos[0]['step'], "current_pair": agent.current_iter, "Net Rewards": [info['net reward'] for info in infos], "loss": str(agent.loss), "loss_avg": str(agent.loss_avg), "epsilon": str(agent.epsilon), "learning_rate":str(agent.learning_rate)})

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                agent.save_model(model_name)

        agent.current_iter += 1

    f = open('iteration_data.json')
    data = json.load(f)
    data['step'] = 0
    data['current_pair'] = 0

    with open("iteration_data.json", "w") as outfile:
    	json.dump(data, outfile, indent = 4)

    write_data(50)
