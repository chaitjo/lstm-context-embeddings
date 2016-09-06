import numpy as np
shuffle_indices = np.random.permutation(np.arange(10662))
with open("shuffled_indices", "wb") as file:
	np.save(file, shuffle_indices)