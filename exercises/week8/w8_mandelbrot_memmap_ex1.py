import sys
import numpy as np
from time import perf_counter as time
from pyarrow import csv
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

# Functions from week5
def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z ** 2 + c
        if np.abs(z) > 2.0:
            return i
    return 100


def mandelbrot_escape_time_multiple(points):
    return [mandelbrot_escape_time(c) for c in points]


def generate_mandelbrot_set(points, num_processes):
    n = len(points) // num_processes
    pool = multiprocessing.Pool(num_processes)
    results = [pool.apply_async(mandelbrot_escape_time_multiple, (points[i * n:(i + 1) * n],)) for i in
               range(num_processes)]
    escape_times = np.array([p.get() for p in results])

    return escape_times


def generate_mandelbrot_set_chunk(points, num_processes, chunk_size=1600):
    '''

    '''

    num_chunks = len(points) // chunk_size
    pool = multiprocessing.Pool(num_processes)
    results = [pool.apply_async(mandelbrot_escape_time_multiple, (points[i * chunk_size:(i + 1) * chunk_size],)) for i
               in range(num_chunks)]
    escape_times = np.array([p.get() for p in results])

    return escape_times



def generate_mandelbrot_set_memmap(x_values, y_values, num_processes, chunk_size=100):
    '''

    '''
    results_async = []
    for i in range(int(np.ceil(len(x_values)/chunk_size))):

        x_vals = x_values[chunk_size * i:chunk_size * (i + 1)]
        y_vals = y_values[chunk_size * i:chunk_size * (i + 1)]
        points = np.array([complex(x, y) for x in x_vals for y in y_vals])
        results_async.append(pool.apply_async(write_mandelbrot_points, (points, )))

    escape_times = np.array([p.get() for p in results_async])

    return escape_times



def plot_mandelbrot(escape_times):
    plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":

    N = int(sys.argv[1])  #  N is the size of the mandelbrot set

    # Create NxN memmap:
    mandelbrot_array = np.memmap('mandelbrot_mm.npy', mode="w+", shape=(N,N), dtype='float64')

    xmin, xmax = -2, 2
    ymin, ymax = -2, 2

    num_proc = 4
    chunk_size = 50

    range_x = abs(xmin) + abs(xmax)
    range_y = abs(ymin) + abs(ymax)

    step_x = range_x / N
    step_y = range_y / N
    print(step_x)

    #for i in range(N//chunk_size):
    #    x_range1 = np.arange(start=xmin + step_x*chunk_size*i, stop=xmin + step_x*chunk_size*(i+1), step=step_x)
    #    print(x_range1[-1], len(x_range1))

    # Precompute points
    x_values = np.linspace(xmin, xmax, N)
    y_values = np.linspace(ymin, ymax, N)

    #print(x_values[0], x_values[-1], len(x_values))
    #for i in range(int(np.ceil(N/chunk_size))):
    #    x = x_values[chunk_size*i:chunk_size*(i+1)]



    points = np.array([complex(x, y) for x in x_values for y in y_values])

    # Compute set
    mandelbrot_set = generate_mandelbrot_set(points, num_proc)

    # Save set as image
    mandelbrot_set = mandelbrot_set.reshape((height, width))
    plot_mandelbrot(mandelbrot_set)