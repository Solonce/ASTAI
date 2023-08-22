import numpy as np
from scipy.stats import norm

def scaled_sigmoid(n, i):
		if abs(i) > n:
			i = n
		scaled_x = 10 * (abs(i) / n) - 5
		return 1 / (1 + np.exp(-scaled_x))

for i in range(-10, 10):
	print(abs(i), round(1-scaled_sigmoid(3, i), 2))