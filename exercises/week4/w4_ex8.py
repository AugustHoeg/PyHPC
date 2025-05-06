from time import perf_counter as time
import numpy as np

import matplotlib.pyplot as plt

from w4_ex2 import distance_matrix, distance_matrix_v2, distance_matrix_v3, distance_matrix_v4

loop_times = []
noloop_times = []
sizes = np.logspace(1, 4, 30)
n_repeat = 5
for size in sizes:
    p1 = np.random.rand(int(size), 2)
    p2 = np.random.rand(int(size), 2)
    t = time()
    for _ in range(n_repeat):
        distance_matrix_v3(p1, p2)
    loop_times.append((time() - t) / n_repeat)
    t = time()
    for _ in range(n_repeat):
        distance_matrix_v4(p1, p2)
    noloop_times.append((time() - t) / n_repeat)

print('ns =', list(sizes))
print('loop_times =', loop_times)
print('noloop_times =', noloop_times)

loop_flops = (sizes / np.array(loop_times)) / 1e6 # number of matrix elements, normalized by 1e6
noloop_flops = (sizes / np.array(noloop_times)) / 1e6 # number of matrix elements, normalized by 1e6
matrix_sizes_KiB = (sizes**2 * p1[..., 0].itemsize) / 1000  # distance matrix size in KiB

plt.figure()
plt.loglog(matrix_sizes_KiB, loop_flops, label="loop flops")
plt.loglog(matrix_sizes_KiB, noloop_flops, label="no loop flops")

plt.axvline(x = 32, color = 'k', label = 'L1')
plt.axvline(x = 1024, color = 'k', label = 'L2')
plt.axvline(x = 19712, color = 'k', label = 'L3')

plt.legend()
plt.xlabel("Input size [KiB]")
plt.ylabel("MFLOPS")
plt.savefig("Loop_vs_no_loop.pdf", dpi=300)