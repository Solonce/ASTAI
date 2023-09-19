# ASTAI - Advanced Stock Trading AI - N-Step Q Learning
![image](https://github.com/Solonce/ASTAI/assets/53124218/3d97c0ed-6054-455b-8246-5265762e661c)


# Project Description

## What is it?
This is a proof of concept tool that will provide trading advice on the crypto market, and soon the stock market. It looks at historical data and applies indicators (Bollinger Bands, RSI at the moment) to make a determination on what action should be taken. These actions will include a short, long, or hold. When you short the market, you are predicting the market will go down, and when you long the market you are predicting the market will go up. If the action is to hold, then no action will be performed as there is no prediction that the market will increase or decrease a discernable amount. The goal of this project is to have a tool that can accurately predict the next movements of the market, and make trading advice based on its training. 

Currently, it trains on the top 50 highest traded pairs with a base pair of USD on the Kraken Cryptocurrency exchange.

While it is natively set up to read live crypto data using the Kraken API, it would be very simple to grab historical data from a stock market API to get the closing values of each candle from the past 720 candles.

## What's the problem it solves?
Trading markets can be insanely hard to predict, some say even impossible. This tool aims to being accurate more than 50% of the time, leading togood/accurate advice on what to do in the market to make the most amount of profit on an actively moving market. Essentially, this tools solves the difficulity of interacting with a trading market by telling you exactly what action to take.

## How it solves that problem.
It solves that problem using N-Step Deep Q Learning. It is a technique that rewards a machine learning model based on how rewarding an action is over 'n' steps.

## N Step Q-Learning
N-Step learning is an approach to Q-Learning that allows the model to base reinforcements on the future rather than just the current step. This allows for more complex reinforcement models as well as giving the model the ability to make choices that will be profitable for the future.

## Why is it beneficial?
Being able to consider the future is very beneficial to the project and model in particular because the goal is to predict the next movements of the market. This allows us to positively or negatively reinforce the model depending on how the market will continue after the initialization of the order.


# Screenshots

## Screenshot 1
Here you can see some basic output from the script.
![image](https://github.com/Solonce/ASTAI/assets/53124218/96047de1-8d78-4a71-8518-a10e7e6c7a9a)


## Screenshot 2
Here you can see the output and formatting of ```iteration_data.json```.

![image](https://github.com/Solonce/ASTAI/assets/53124218/332714a6-4b36-4461-8ab4-24f2c409cc7c)

## Screenshot 3
Here we can see the daemon service is up and running.
![image](https://github.com/Solonce/ASTAI/assets/53124218/b402382d-da52-4352-9847-4705df99a935)


# Getting Started

## Requirements
**Python:**
The main language the script is programmed in.
>version: 3.9.16

**Tensorflow:**
The package behind most of the model processing
>version 2.10.0

**Pandas**
This is the package that creates and interacts with the pandas DataFrame object.
>version 2.0.3

**Gym**
This package allows us to seed the model and have a gym env base.
>version 0.26.2

**Requests**
This package allows us to get live crypto/stock data.
>version 2.25.1

**Numpy:**
The package that aids tensorflow in most of its computation.
>version 1.24.3


## Installation 
I would call the github repo. To do this run this command in terminal:

```
git clone https://www.https://github.com/Solonce/ASTAI/
```

# Usage

## How to use
The program currently is used and viewed by interacting with iteration_data.json. As that contains all current vital info. As this model is set up just for training at the time of writing this, ```avg_loss``` 

To build your own model and have it train continuously, you need to do the following.

```
rm main.h5
```
```
rm iteration_data.json
```

Then you will need to create a file called ```astai_service.service```
```
[Unit]
Description=ASTAI Script Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /path/to/astai_dqn.py
User=yourusername
WorkingDirectory=/path/to/script/directory
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

```

Then you're going to want to ```start```/```enable``` the service. Enabling it will cause it to start on boot.

```
systemctl start astai_service.service
```

To check the status of the service you can run
```
systemctl status astai_service.service
```

Then, to check the logs, you can run
```
journalctl -u astai_service.service -n 100 --no-pager
```

## Known Bugs
-Very slight memory leak, takes about an hour to emerge. It is possible to workaround, but not ideal.


# FAQ

## What is Q Learning
Q Learning is an advanced machine learning method that consists of many smaller parts. Let's discuss those.

**Q-Values**
Q-Values is the list of possible rewards for all actions given a model and a state. Typically, the highest reward action is the chosen action.

**Epsilon**
The Epsilon value deals with the "Exploration" or "Exploitation" of the model. Essentially, if the model has a higher epsilon it will be more explorative, more likely to explore a random choice. Or, if the model has a lower epsilon, it will exploit the knowledge of the model and choose less randomly.

**Learning Rate**
The learning rate is the amount of old data is overwritten by new experiences/data in the model.

**Discount Factor**
This is a value that discounts the rewards of an action the further into the future it is calculated. For example in this model, the discount factor is a scaled sigmoid value between 1 and 0 that lowers as an open order persists until it reaches 0, in which the order is closed. 

**Bellman Equation**
This is an equation that expresses the current predicted reward from an action, and the discounted future rewards. It is vital when calculating with Q Learning.

Q Learning is achieved by passing the q values through the bellman equation each iteration of the training period, until the values converge.

# Who Am I?
Solomon Ince
ML/AI Enthusiast
Professional Cook
