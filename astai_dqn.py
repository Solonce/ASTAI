import numpy as np
import random
import sys
from collections import deque
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
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
from stock_trading_env import StockTradingEnv
import json
from dqn_agent import DQNAgent
import gc
import tracemalloc
import gym
from gym import spaces
from gym.utils import seeding
import multiprocessing
import torch
from collections import deque

torch.set_num_threads(1)
def make_env(rank, data, window_size, seed=1, current_step=0):
    env = StockTradingEnv(data, window_size, current_step=current_step)
    env.seed(seed+rank)
    #env = Monitor(env)
    return env

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

'''
0 - np.array
1 - list
2 - np.float64
3 - bool
4 - list
'''

def write_memories(mem):
     memory_dictionary = {'memories': mem}
     for _, memories in memory_dictionary.items():
         for mem_index, memory in enumerate(memories):
             for i, _memory in enumerate(memory):
                 if type(_memory) == type(np.array([])):
                    if i == 6:
                         memory_dictionary[_][mem_index][6] = list(memory_dictionary[_][mem_index][6])
                         memory_dictionary[_][mem_index][6][0] = list(memory_dictionary[_][mem_index][6][0])
                         for j, data in enumerate(memory_dictionary[_][mem_index][6][0]):
                             memory_dictionary[_][mem_index][6][0][j] = float(data)
                             #print(type(memory_dictionary[_][mem_index][6][0][j]))
                         #memory_dictionary[_][mem_index][6][0][0] = list
                         #print(f"Overall {type(memory_dictionary[_][mem_index][6])} Base {memory_dictionary[_][mem_index][6][0]} Type {type(memory_dictionary[_][mem_index][6][0])}")
                    if i == 0:
                         memory_dictionary[_][mem_index][0] = list(memory_dictionary[_][mem_index][0])
                         for entry_index, entry in enumerate(memory_dictionary[_][mem_index][0]):
                              #print(memory_dictionary[_][mem_index][0])
                              memory_dictionary[_][mem_index][0][entry_index] = list(memory_dictionary[_][mem_index][0][entry_index])
                              #print(type(memory_dictionary[_][mem_index][0][entry_index]))

                 elif type(_memory) == type([]) and i != 0 and i != 6:
                     for entry_index, entry in enumerate(memory_dictionary[_][mem_index][i]):
                         #print(entry, i, type(entry))
                         memory_dictionary[_][mem_index][i][entry_index] = float(_memory[entry_index])
     with open("memory.json", "w") as outfile:
        json.dump(memory_dictionary, outfile, indent = 4)
     print("Memory written to JSON.")

def read_memories():
     f = open('memory.json')
     memory_dictionary = json.load(f)
     _data = deque(maxlen=100)
     for _, memories in memory_dictionary.items():
         for mem_index, memory in enumerate(memories):
             for i, _memory in enumerate(memory):
                 memory_dictionary[_][mem_index][0] = np.array(memory_dictionary[_][mem_index][0])
                 memory_dictionary[_][mem_index][2] = [np.float64(memory_dictionary[_][mem_index][2])]
                 if type(_memory) == type([]) and i == 6:
                     memory_dictionary[_][mem_index][6] = np.array(memory_dictionary[_][mem_index][6])
                     memory_dictionary[_][mem_index][6][0] = np.array(memory_dictionary[_][mem_index][6][0])
                 if type(memory_dictionary[_][mem_index][i]) == type([]):
                     for entry_index, entry in enumerate(memory_dictionary[_][mem_index][i]):
                         if i == 0:
                             memory_dictionary[_][mem_index][i][entry_index] = np.array(entry)
                         elif i == 2:
                             memory_dictionary[_][mem_index][i][entry_index] = np.int64(entry[entry_index])
                         #elif i == 4:
                             #memory_dictionary[_][mem_index][i][entry_index] = np.float64(entry[entry_index])

             try:
                 memory_dictionary[_][mem_index] = tuple(memory)
             except:
                 print("empty memory")
             _data.append(memory_dictionary[_][mem_index])
     print("Memories Applied.")
     return _data

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
        episode_start = 0
    else:
        f = open('iteration_data.json')
        data = json.load(f)
        agent = DQNAgent(3, window_size, is_model=True, current_iter=data['current_pair'], current_step=data['step'], model_name=model_name, loss=float(data['loss_avg']), epsilon=float(data['epsilon']), learning_rate=float(data['learning_rate']))
        [agent.memory.append(memory) for memory in read_memories()]
        agent.current_pair = data['current_pair']
        sorted_pairs = list(price_data.keys())[agent.current_pair:]
        episode_start = int(data['episode'])
    print("Loaded Agent")
    for i, pair in enumerate(sorted_pairs):
        stock_price_data_np = price_data[pair]

        num_processes = 1
        if not os.path.isfile(model_name):
        	print("No model found")
        	envs = make_env(i, stock_price_data_np, window_size)
        else:
        	print("Model Found")
        	envs = make_env(i, stock_price_data_np, window_size, current_step=agent.step)
        print("Loaded Subprocesses")
        episodes = 10
        batch_size = 32
        kill_size = 100

        for e in range(episode_start, episodes):
            state = envs.reset()
            print(pair)
            for time in range(1000):
                gc.collect()
                state = np.array(state[(envs.current_step):window_size+(envs.current_step)])
                action, predictions = agent.act(state)
                next_state, reward, done, _, info = envs.step(action)
                agent.remember(state, action, reward, done, info['n_rewards'], envs.get_best_reward(), predictions)
                state = next_state
                agent.step = info['step']
                agent.current_pair = i
                print(f"time_step: {time}, episode: {e}/{episodes}, action: {action}, reward: {np.round(reward, 2)}, net reward: {np.round(info['net reward'], 2)} score: {agent.step}, e: {agent.epsilon}, done: {done}, open orders: {info['orders']}")
                print(done)
                if done is True or envs.current_step>=(len(stock_price_data_np)-window_size):
                    print("DONE")
                    update_iteration_data({"episode": str(e+1), "step": 0, "current_pair": agent.current_iter, "Net Rewards": info['net reward'], "loss": str(agent.loss), "loss_avg": str(agent.loss_avg), "epsilon": str(agent.epsilon), "learning_rate": str(agent.learning_rate)})
                    kill_size = kill_size - time
                    break
                else:
                    update_iteration_data({"episode": str(e), "step": info['step'], "current_pair": agent.current_iter, "Net Rewards": info['net reward'], "loss": str(agent.loss), "loss_avg": str(agent.loss_avg), "epsilon": str(agent.epsilon), "learning_rate":str(agent.learning_rate)})

                if len(agent.memory) > batch_size:
                    minibatch = random.sample(agent.memory, batch_size)
                    agent.replay(minibatch)

                agent.save_model(model_name)
                write_memories([list(memory) for memory in agent.memory])
                if time >= kill_size:
                    quit()
        episode_start = 0
        agent.current_iter += 1
        agent.learning_rate = agent.learning_rate * agent.learning_rate_decay

    f = open('iteration_data.json')
    data = json.load(f)
    data['episode'] = 0
    data['step'] = 0
    data['current_pair'] = 0

    with open("iteration_data.json", "w") as outfile:
    	json.dump(data, outfile, indent = 4)

    write_data(50)
