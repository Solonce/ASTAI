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
        self.order_count = 0

        if current_step == 0:
            self.current_step = window_size-1

    def get_percent_change(self, current_price, future_price):
        if future_price == 0:
            return 0
        return ((future_price - current_price)/future_price)*100

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
        self.done=False
        self.position = None
        self.orders = np.array([])
        return self.stock_prices

    def get_best_reward(self, n=5):
        if self.current_step + n > len(self.stock_prices):
             n = (len(self.stock_prices)-1)-self.current_step
        min = 1000
        max = -1000
        action = 0
        percent_gain = 3
        for i in range(n):
            percent_change = self.get_percent_change(self.stock_prices[self.current_step][0], self.stock_prices[self.current_step+i][0])
            if percent_change > max:
                max = percent_change
            if percent_change < min:
                min = percent_change
            if max > percent_gain and min <= -percent_gain/2:
                action = 1
                break
            elif min < -percent_gain and max >= percent_gain/2:
                action = 2
                break

        if max > percent_gain or min < -percent_gain and action == 0:
            if max > abs(min):
                action = 1
            else:
                action = 2
        print(f"Best action: {action}")
        reward = 0
        if action == 1:
            reward = max
        elif action == 2:
            reward = min
        else:
            reward = (max, min)
        print(f"Reward {reward}")
        return action

    def get_net_rewards(self):
        return self.net_reward

    def get_future_reward(self, i, action):
        percent_gain = 3
        percent_change = self.get_percent_change(self.stock_prices[self.current_step][0], self.stock_prices[self.current_step+i][0])
        if action == 2:
            percent_change *= -1
        elif action == 0:
            return self.piecewise(percent_change, 1, 3)
        if percent_change <= -percent_gain/2:
           return -percent_gain/2
        elif percent_change >= percent_gain:
           return percent_change
        else:
           return (percent_change * (1-self.scaled_sigmoid(self.max_order_length, i)))

    def get_reward(self, current_price, forward_index=-1):
        reward = 0
        closed_orders = []
        for order in self.orders:
            order_percent = order.get_percent_change(current_price[0])
            order_state = order.check_change(current_price[0]) # returns 0 nothing, 1 exit
            if forward_index == -1:
                print(f"Order {order.order_id} Order type: {order.order_type} Percent: {order_percent}")
            if order.order_type == "sell":
                order_percent = order_percent * -1
            if order_state != 0 or order.life_counter >= self.max_order_length:
                if order_state == 1:
                    reward += order_percent
                else:
                    reward -= order.percent_gain/2
                if forward_index == -1:
                    self.orders = np.delete(self.orders, np.where(self.orders==order)[0][0])
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
        if self.current_step >= (len(self.stock_prices) - 1) or self.current_step >= (len(self.stock_prices)-self.window_size-1):
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
            self.orders = np.append(self.orders, trade_order(_type, current_price[0], self.order_count))
            self.order_count += 1

        reward = self.get_reward(current_price)

        n_rewards = []
        order_percent = 0
        if self.current_step + n > len(self.stock_prices):
            n = (len(self.stock_prices)-1)-self.current_step
        if n != 0:
            future_price = self.stock_prices[self.current_step+1]
            if (future_price[0] == 0.0 or future_price == 0):
                future_price[0] = current_price[0]
                index = 0
                while future_price[0] == 0.0:
                    future_price[0] = self.stock_prices[self.current_step-index][0]
                    index += 1
                order_percent = ((future_price[0]-current_price[0])/future_price[0])*100
            print(f"Percent Change: {order_percent}")
        if action == [0]:
            future_price = self.stock_prices[self.current_step+n]
            if future_price[0] == 0.0:
                future_price[0] = self.stock_prices[self.current_step+(n-1)]
                index = 0
                while future_price[0] == 0.0:
                    future_price[0] == self.stock_prices[self.current_step+(n-index)][0]
                    index += 1
            order_percent = ((future_price[0]-current_price[0])/future_price[0])*100
            print(f"Percent Change (Hold): {order_percent}")
            reward += self.piecewise(order_percent, 1, 3)

        for i in range(n):
            reward_result = self.get_future_reward(i, action[0])
            n_rewards.append(reward_result)
            if reward_result >= 3 or reward_result <= -1.5 and action[0] != 0:
                n_rewards[i] *= 1.5
                if abs(n_rewards[i]) <= 5:
                    break
            if i == n-1:
                n_rewards[i] = abs(n_rewards[i]) * -1.5

       # print(f"N_Rewards: {n_rewards} Sum Rewards: {sum(n_rewards)}")

        self.net_reward += reward
        self.reset_info = {"net reward": self.net_reward, "orders": self.orders.size, "step": self.current_step, "n_rewards": n_rewards}

        
        return self.stock_prices, reward, self.done, False, self.reset_info
        
