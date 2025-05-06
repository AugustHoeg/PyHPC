
import sys
import math
from time import perf_counter as time
import numpy as np
import matplotlib.pyplot as plt

iterations = 1000
sizes = (np.logspace(1, 4.5, 30)).astype(int)
print(sizes)

col_times = []
row_times = []

for size in sizes:
	print(f"Timing row/column operations for matrix size: {size}x{size}")
	mat = np.random.rand(size, size)

	start_time = time()
	for i in range(iterations):
		double_column = 2 * mat[:, 0]
	end_time = time()
	col_time = (end_time - start_time) / iterations
	col_times.append(col_time)
	print(f"double column took: {col_time} sec. per iteration")

	start_time = time()
	for i in range(iterations):
		double_row = 2 * mat[0, :]
	end_time = time()
	row_time = (end_time - start_time) / iterations
	row_times.append(row_time)
	print(f"double row took: {row_time} sec. per iteration")

row_flops = (sizes / np.array(row_times)) / 1e6 # number of matrix elements, normalized by 1e6
col_flops = (sizes / np.array(col_times)) / 1e6 # number of matrix elements, normalized by 1e6
matrix_sizes_KiB = (sizes**2 * sizes[0].itemsize) / 1000

plt.figure()
plt.loglog(matrix_sizes_KiB, col_flops, label="Column double")
plt.loglog(matrix_sizes_KiB, row_flops, label="Row double")

plt.axvline(x = 32, color = 'k', label = 'L1')
plt.axvline(x = 1024, color = 'k', label = 'L2')
plt.axvline(x = 19712, color = 'k', label = 'L3')

plt.legend()
plt.xlabel("Matrix size [KiB]")
plt.ylabel("MFLOPS")
plt.savefig("CacheEffects.pdf", dpi=300)

plt.figure()
plt.loglog(matrix_sizes_KiB, row_flops / col_flops, label="Row/Column ratio")

plt.axvline(x = 32, color = 'k', label = 'L1')
plt.axvline(x = 1024, color = 'k', label = 'L2')
plt.axvline(x = 19712, color = 'k', label = 'L3')

plt.legend()
plt.xlabel("Matrix size [KiB]")
plt.ylabel("MFLOPS ratio")
plt.savefig("CacheEffects_ratio.pdf", dpi=300)