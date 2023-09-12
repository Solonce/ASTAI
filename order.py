import numpy as np

class trade_order():
	def __init__(self, order_type, entry_price, percent_gain=2):
		self.order_type = order_type
		self.percent_gain = percent_gain
		self.entry_price = entry_price
		self.life_counter = 0

	def scaled_sigmoid(self, n, i):
		if abs(i) > n:
			i = n
		scaled_x = 10 * (abs(i) / n) - 5
		return 1 / (1 + np.exp(-scaled_x))

	def get_percent_change(self, current_price):
            if current_price == 0:
                return 0
            return ((current_price-self.entry_price)/current_price)*100

	def check_change(self, current_price):
		percent_diff = self.get_percent_change(current_price)
		if self.order_type == "buy":
			if percent_diff >= self.percent_gain:
				return 1
			elif percent_diff <= -(self.percent_gain/2):
				return 2
			pass
		elif self.order_type == "sell":
			if percent_diff <= -self.percent_gain:
				return 1
			elif percent_diff >= (self.percent_gain/2):
				return 2
			pass
		return 0



