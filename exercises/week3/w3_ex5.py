
import sys
import math
from time import perf_counter as time
import numpy as np
import matplotlib.pyplot as plt

iterations = 100
sizes = (np.logspace(2, 8, 30)).astype(int)
print(sizes)

row_times = []

for size in sizes:
	print(f"Timing row/column operations for row vector of size: {1}x{size}")
	mat = np.random.rand(1, size)

	start_time = time()
	for i in range(iterations):
		double_row = 2 * mat[0, :]
	end_time = time()
	row_time = (end_time - start_time) / iterations
	row_times.append(row_time)
	print(f"double row took: {row_time} sec. per iteration")

# FLOPS calculation: Length of the array divided by the time it took...
row_flops = (sizes / np.array(row_times)) / 1e6 # number of matrix elements, normalized by 1e6
matrix_sizes_KiB = (sizes * sizes[0].itemsize) / 1000

plt.figure()
plt.loglog(matrix_sizes_KiB, row_flops)

plt.axvline(x = 32, color = 'k', label = 'L1')
plt.axvline(x = 1024, color = 'k', label = 'L2')
plt.axvline(x = 19712, color = 'k', label = 'L3')

plt.xlabel("Matrix size [KiB]")
plt.ylabel("MFLOPS")
plt.savefig("CacheEffects_row_vector.pdf", dpi=300)
