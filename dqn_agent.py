from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import random


class DQNAgent:
    def __init__(self, action_size, window_size, is_model=False, current_iter = 0, current_step=0, model_name="", loss=0):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.window_size = window_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00000001
        self.step = current_step
        self.current_iter = current_iter
        self.loss = 0
        self.loss_avg = loss
        if not is_model:
        	self.model = self._build_model()
        else:
        	self.model = load_model(model_name, compile = False)
        	self.model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0), loss='mse')

    def _build_model(self):
        model = Sequential()
        model.add(GRU(128, input_shape=(self.window_size, 5), return_sequences=True, dynamic=True)) 
        model.add(Dropout(0.2))
        model.add(GRU(256, return_sequences=True, dynamic=True))
        model.add(Dropout(0.2))
        model.add(GRU(512, dynamic=True))
        model.add(Dropout(0.2))
        model.add(Flatten(input_shape=(self.window_size, 5)))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done, n_rewards):
        self.memory.append((state, action, reward, next_state, done, n_rewards))

    def act(self, state, prediction=False):

        if np.random.rand() <= self.epsilon and prediction==False:
            act_values = [random.randrange(self.action_size)]
        else:
            state = state.reshape(1, 50, 5)
            prediction = self.model.predict(state)
            act_values = [np.argmax(prediction)]
        return act_values

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print("Running Minibatch")
        for i, (state, action, reward, next_state, done, n_rewards) in enumerate(minibatch):
            next_state = np.expand_dims(next_state, axis=1)
            state = np.expand_dims(state, axis=1)
            state[np.isnan(state)] = 0
            next_state[np.isnan(next_state)] = 0
            #print(f"\rMinibatch Iter: {i}", end="", flush=True)
            target = sum([self.gamma**k * rew for k, rew in enumerate(n_rewards)])

            if not done:
                target += (self.gamma**len(n_rewards)) * np.amax(self.model.predict(state))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            _loss = self.model.train_on_batch(state, target_f)
            self.loss = _loss
            if self.loss_avg == 0:
                self.loss_avg = _loss
            else:
               self.loss_avg = (self.loss_avg+_loss)/2

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, model_name):
    	self.model.save(model_name)
