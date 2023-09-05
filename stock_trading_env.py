import gym
from gym import spaces
from gym.utils import seeding
from order import trade_order
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size, current_step=0):
        self.stock_prices = data
        self.position = None  # could be 'buy', 'hold', or 'sell'
        self.current_step = current_step
        self.done = False
        self.bought_price = None
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data), len(self.stock_prices)), dtype=np.float32)  
        self.action_space = spaces.Discrete(3)
        self.render_mode = 'human'
        self.reset_info = {}
        self.orders = np.array([])
        self.max_orders = 5
        self.max_order_length = 5
        self.net_reward = 0
        self.window_size = window_size

        if current_step == 0:
            self.current_step = window_size-1

    def scaled_sigmoid(self, n, i):
        if abs(i) > n:
            i = n
        scaled_x = 10 * (abs(i) / n) - 5
        return 1 / (1 + np.exp(-scaled_x))

    def piecewise(self, p, theta, _max):
        if abs(p) <= theta:
            return 3*(1-self.scaled_sigmoid(theta, p))
        elif abs(p) > theta:
            return -3*(self.scaled_sigmoid(_max, p))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        if self.current_step >= (len(self.stock_prices) - 1-self.window_size):
            self.current_step = self.window_size-1
        self.position = None
        self.done = False
        self.orders = np.array([])
        return self.stock_prices

    def get_net_rewards(self):
        return self.net_reward

    def get_reward(self, current_price, forward_index=-1):
        reward = 0
        for order in self.orders:
            order_state = order.check_change(current_price[0]) # returns 0 nothing, 1 exit
            order_percent = order.get_percent_change(current_price[0])
            if order.order_type == "sell":
                order_percent = order_percent * -1
            if order_state != 0 or order.life_counter >= self.max_order_length:
                reward += order_percent
                if forward_index == -1:
                    self.orders = np.delete(self.orders, np.where(self.orders==order)[0][0])
                    return reward
            else:
                life = order.life_counter
                if life+forward_index >= self.max_order_length and forward_index != -1:
                    life = self.max_order_length
                reward += (order_percent * (1-order.scaled_sigmoid(self.max_order_length, order.life_counter)))
            if forward_index == -1:
                order.life_counter = order.life_counter + 1
        return reward

    def step(self, action, n=5, step=-1):
        self.current_step += 1
        if self.current_step >= (len(self.stock_prices) - 1) or self.current_step >= len(self.stock_prices)-self.window_size:
            self.done = True
            self.current_step = len(self.stock_prices)-self.window_size

        if not self.done:
            current_price = self.stock_prices[self.current_step]
        else:
            current_price = self.stock_prices[-1]

        if (action == [1] or action == [2]) and self.orders.size <= self.max_orders:
            _type = "buy"
            if action == [2]:
                _type = "sell"
            self.orders = np.append(self.orders, trade_order(_type, current_price[0]))

        reward = self.get_reward(current_price)

        n_rewards = []
        if self.current_step + n > len(self.stock_prices):
            n = (len(self.stock_prices)-1)-self.current_step

        if action == [0]:
            future_price = self.stock_prices[self.current_step+n]
            if future_price[0] == 0.0:
                future_price[0] = current_price[0]

            order_percent = ((future_price[0]-current_price[0])/future_price[0])*100
            reward += self.piecewise(order_percent, 1, 3)

        for i in range(n):
            n_rewards.append(self.get_reward(self.stock_prices[self.current_step+i+1], forward_index=i))

        self.net_reward += reward
        self.reset_info = {"net reward": self.net_reward, "orders": self.orders.size, "step": self.current_step, "n_rewards": n_rewards}


        return self.stock_prices, reward, self.done, False, self.reset_info
        
