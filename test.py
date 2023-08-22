import random

num_trades_day = 3
success_rate = [i * 0.1 for i in range(5, 10)]
proff_loss = [[1 + (0.005 * i), 1 - (0.0025*i)] for i in range(10)]
days_running = 30

def percent(x, y):
	return ((x-y)/y)+1

for index, rate in enumerate(success_rate):
	print('---------------------------------')
	for pl in proff_loss:
		init_amt = 100
		wallet_amount = init_amt
		for i in range(days_running):
			for j in range(num_trades_day):
				x = random.random()
				if x < rate:
					wallet_amount = wallet_amount * pl[0]
				else:
					wallet_amount = wallet_amount * pl[1]
		print(rate, pl, percent(wallet_amount, init_amt))