
import sys
import math
import time
import numpy as np

iterations = 1000
SIZE = 100

mat = np.random.rand(SIZE, SIZE)

start_time = time.time()
for i in range(iterations):
	mat[:, 0] * 1.01
	#double_column = 2 * mat[:, 0]
end_time = time.time()
print(f"double column took: {end_time - start_time} sec")

start_time = time.time()
for i in range(iterations):
	mat[0, :] * 1.01
	#double_row = 2 * mat[0, :]
end_time = time.time()
print(f"double row took: {end_time - start_time} sec")
