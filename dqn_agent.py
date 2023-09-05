from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from collections import deque
import tensorflow as tf
import numpy as np
import random


class DQNAgent:
    def __init__(self, action_size, window_size, is_model=False, current_iter = 0, current_step=0, model_name="", loss=0, epsilon=1.0, learning_rate=0.001):
        self.action_size = action_size
        self.memory = deque(maxlen=100)
        self.window_size = window_size
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.learning_rate_decay = 0.9995
        self.step = current_step
        self.current_iter = current_iter
        self.loss = 0
        self.loss_avg = loss
        if not is_model:
            self.model = self._build_model()
        else:
            self._load_model(model_name)

    def _load_model(self, model_name):
        self.model = load_model(model_name)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0), loss='mse', run_eagerly=True)

    def _build_model(self):
        model = Sequential()
        model.add(GRU(128, input_shape=(self.window_size, 5), return_sequences=True, name="gru_1"))
        model.add(Dropout(0.2))
        model.add(GRU(256, return_sequences=True, name="gru_2"))
        model.add(Dropout(0.2))
        model.add(GRU(512, name="gru_3"))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0), loss='mse', run_eagerly=True)
        return model

    def remember(self, state, action, reward, done, n_rewards):
        self.memory.append((state, action, reward, done, n_rewards))

    def act(self, state, prediction=False):
        state = state.reshape(1, 50, 5)
        if np.random.rand() <= self.epsilon and prediction==False:
            act_values = [random.randrange(self.action_size)]
        else:
            prediction = self.model.predict(state, verbose=0)
            act_values = [np.argmax(prediction)]
        return act_values

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        loss_arr = []
        print("Running Minibatch")
        for i, (state, action, reward, done, n_rewards) in enumerate(minibatch):
            print(f"\rMinibatch Iteration: {i}", end="", flush=True)
            state[np.isnan(state)] = 0
            state = np.expand_dims(state, axis=0)
            predictions = self.model.predict(state, verbose=0)
            target = sum([self.gamma**k * rew for k, rew in enumerate(n_rewards)])

            if not done:
                target += (self.gamma**len(n_rewards)) * np.amax(predictions)

            target_f = predictions
            target_f[0][action] = target
            _loss = self.model.train_on_batch(state, target_f)
            loss_arr.append(_loss)
            self.loss = _loss

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.loss_avg = (self.loss_avg+(sum(loss_arr)/len(loss_arr)))/2
        print(f"\nAverage Loss For Minibatch Iteration: {self.loss_avg}")

    def save_model(self, model_name):
    	self.model.save(model_name)
