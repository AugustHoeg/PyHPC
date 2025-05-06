
from time import perf_counter as time
import random
import multiprocessing
import numpy as np

import matplotlib.pyplot as plt

def serial_pi(samples=1000000):

    hits = 0

    for i in range(samples):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        if x ** 2 + y ** 2 <= 1:
            hits += 1

    pi = 4.0 * hits / samples
    return pi


def sample():
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if x**2 + y**2 <= 1:
        return 1
    else:
        return 0


def parallel_pi(samples=1000000, n_proc=10):

    pool = multiprocessing.Pool(n_proc)
    results_async = [pool.apply_async(sample) for i in range(samples)]
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits / samples

    return pi


def sample_multiple(samples_partial):
    return sum(sample() for i in range(samples_partial))


def chunked_parallel_pi(samples=1000000, n_proc=10):

    chunk_size = samples // n_proc
    pool = multiprocessing.Pool(n_proc)
    results_async = [pool.apply_async(sample_multiple, (chunk_size,)) for i in range(n_proc)]
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits / samples

    return pi


def chunked_parallel_pi_using_map(samples=1000000, n_proc=10):

    chunk_size = samples // n_proc
    pool = multiprocessing.Pool(n_proc)
    results_async = pool.map(sample_multiple, chunk_size=[chunk_size] * n_proc)
    hits = sum(r.get() for r in results_async)
    pi = 4.0 * hits / samples

    return pi


# Run performance tests of the three implementations
start_time = time()
serial_pi(samples=1000000)
time_serial = (time() - start_time)
print(f"Serial pi took: {time_serial} sec.")

start_time = time()
parallel_pi(samples=1000000, n_proc=10)
time_parallel = (time() - start_time)
print(f"Fully parallel pi took: {time_parallel} sec.")

start_time = time()
chunked_parallel_pi(samples=1000000, n_proc=10)
time_chunked = (time() - start_time)
print(f"Chunked parallel pi took: {time_chunked} sec.")

start_time = time()
chunked_parallel_pi_using_map(samples=1000000, n_proc=10)
time_chunked_map = (time() - start_time)
print(f"Chunked parallel pi using map took: {time_chunked_map} sec.")


# Bar plot
fruits = ['Serial Pi', 'Fully parallel Pi', 'Chunked parallel Pi', 'Chunked parallel Pi using map']
sales = [time_serial, time_parallel, time_chunked, time_chunked_map]

plt.bar(fruits, sales)
plt.yscale('log')
plt.title('Monte Carlo Pi estimation speeds')
plt.ylabel('Time [s]')
plt.xlabel('Implementation')
plt.savefig("pi_estimation.pdf", dpi=300)


# Run performance tests of chunked pi
max_processes = 10
times = []
for n_proc in range(1, max_processes):
    start_time = time()
    chunked_parallel_pi(samples=1000000, n_proc=n_proc)
    time_chunked = (time() - start_time)
    times.append(time_chunked)
    print(f"Chunked parallel pi w. n_proc = {n_proc} took: {time_chunked} sec.")

speedup = [times[0] / time for time in times]  # speed-up is fraction: old_time / new_time

plt.figure()
plt.plot(range(1, max_processes), speedup)
plt.ylabel("Speed-up")
plt.xlabel("No. of processes")
plt.savefig("speedup_vs_processes.pdf", dpi=300)
