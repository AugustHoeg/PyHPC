from time import perf_counter as time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_escape_time(c):
    z = 0
    for i in range(100):
        z = z**2 + c
        if np.abs(z) > 2.0:
            return i
    return 100

def mandelbrot_escape_time_multiple(points):
    return [mandelbrot_escape_time(c) for c in points]

def generate_mandelbrot_set(points, num_processes):

    n = len(points) // num_processes
    pool = multiprocessing.Pool(num_processes)
    results = [pool.apply_async(mandelbrot_escape_time_multiple, (points[i*n:(i+1)*n], )) for i in range(num_processes)]
    escape_times = np.array([p.get() for p in results])

    return escape_times

def generate_mandelbrot_set_chunk(points, num_processes, chunk_size=1600):

    '''
    
    '''

    num_chunks = len(points) // chunk_size
    pool = multiprocessing.Pool(num_processes)
    results = [pool.apply_async(mandelbrot_escape_time_multiple, (points[i*chunk_size:(i+1)*chunk_size], )) for i in range(num_chunks)]
    escape_times = np.array([p.get() for p in results])

    return escape_times

def plot_mandelbrot(escape_times):
    plt.imshow(escape_times, cmap='hot', extent=(-2, 2, -2, 2))
    plt.axis('off')
    plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    width = 800
    height = 800
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    num_proc = 4

    # Precompute points
    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)
    points = np.array([complex(x, y) for x in x_values for y in y_values])

    # Compute set
    mandelbrot_set = generate_mandelbrot_set(points, num_proc)

    # Save set as image
    mandelbrot_set = mandelbrot_set.reshape((height, width))
    plot_mandelbrot(mandelbrot_set)

    # Run performance tests of mandelbrot set generation
    max_processes = 10
    times_non_chunked = []
    times_chunked = []
    for n_proc in range(1, max_processes):
        start_time = time()
        generate_mandelbrot_set(points, n_proc)
        time_non_chunked = (time() - start_time)
        times_non_chunked.append(time_non_chunked)
        print(f"Non-chunked Mandelbrot generation w. n_proc = {n_proc} took: {time_non_chunked} sec.")

        start_time = time()
        generate_mandelbrot_set_chunk(points, n_proc, chunk_size=1600)
        time_chunked = (time() - start_time)
        times_chunked.append(time_chunked)
        print(f"Chunked Mandelbrot generation w. n_proc = {n_proc} took: {time_chunked} sec.")

    speedup_non_chunked = [times_non_chunked[0] / time for time in times_non_chunked]  # speed-up is fraction: old_time / new_time
    speedup_chunked = [times_chunked[0] / time for time in times_chunked]  # speed-up is fraction: old_time / new_time

    plt.figure()
    plt.plot(range(1, max_processes), speedup_non_chunked, label='Non-chunked')
    plt.plot(range(1, max_processes), speedup_chunked, label='Chunked')
    plt.legend()
    plt.ylabel("Speed-up")
    plt.xlabel("No. of processes")
    plt.savefig("mandelbrot_speedup_vs_processes.pdf", dpi=300)