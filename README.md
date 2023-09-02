# ASTAI - Advanced Stock Trading AI - N-Step Q Learning
![image](https://github.com/Solonce/ASTAI/assets/53124218/3d97c0ed-6054-455b-8246-5265762e661c)


# Project Description

## What is it?
This is a proof of concept tool that will provide trading advice on the crypto market, and soon the stock market. It looks at historical data and indicators (Bollinger Bands, RSI at the moment) to make this determination. The goal of this project is to have a tool that can accurately predict next the movements of the market, and make trading advice based on its training. It will tell you if it would reccoemnd to short or long a specific trading pair on the crypto market. 

While it is natively set up to read live crypto data using the Kraken API. However, it would be as simple as grabbing the closing values of each candle from the past 720 candles.

## Whats the problem it solves?
This tool hopes to solve the problem of the difficulty that comes when navigating the stock market.

## How it solves that problem.
It should be as easy as using it to get its suggeted movement given the normalized current market conditions. It gets that suggeted movement from to the model that will eventually be included with the github repo. It uses N-Step Q Learning to find out the action that should be taken.

## N Step Q-Learning
N-Step learning is an approach to Q-Learning that allows the model to base reinforcments on the future rather than just the current step. This allows for more complex reinforcement models as well as giving the model the ability to make choices that will be profitable for the future.

## Why is it beneficial?
Being able to consider the future is very benefiicial to the project and model in particular because the goal to predict the next movements of the market. This allows us to positively or negatively reinforce the model depending on how the market will continue after the initalization of the order.


## Examples / Demos


# Screenshots

## Screenshot 1
Here you can see some basic output from the script.
![image](https://github.com/Solonce/ASTAI/assets/53124218/96047de1-8d78-4a71-8518-a10e7e6c7a9a)


## Suncreenshot 2
Here you can see the output and formatting of ```iteration_data.json```.

![image](https://github.com/Solonce/ASTAI/assets/53124218/332714a6-4b36-4461-8ab4-24f2c409cc7c)

## Screenshot 3

# Getting Started

## Requirements
**Python:**
The main language the script is programmed in.
>version: 3.9.16

**Tensorflow:**
The package behind most of the model processing
>version 2.10.0

**Numpy:**
The package that aids tensorflow in most of its computation.
>version 1.24.3


## Installation 
I wold call the github repo. To do this run this command in terminal:

```
git clone https://www.https://github.com/Solonce/ASTAI/
```

# Usage

## How to use
The program currently is used and viewed by iteracting with iteration_data.json. As that contains all current vital info. As this model is set up just for training at the time of writing this, ```avg_loss``` 

## Examples


# Testing

## How to Test


# Roadmap

## Upcoming features

## Knwon Bugs
-Very slight memory leak, takes about an hour to emerge. It is possible to workaround, but not ideal.

## Future Improvements



# FAQ

## What is Q Learning

# Who Am I?



#

##Advanced Stock Strading AI:
This tool sets out to train/make accurate predictions on the stock market.


