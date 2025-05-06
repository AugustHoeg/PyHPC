import ctypes
import multiprocessing as mp
import sys
from time import perf_counter as time
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr, dtype='float32')


def reduce_step(args):
    b, e, s, elemshape = args
    arr = tonumpyarray(shared_arr).reshape((-1,) + elemshape)
    #print("arr", arr.shape, arr[b:e:s])
    # Change the code below to compute a step of the reduction
    # ---------------------------8<---------------------------
    arr[b, ...] = sum(arr[b:e:s, ...])  # sum of neighbouring elements.
    # In the case of CelebA, where we have input (200, 128, 128, 3), we sum over the image dimension

def reduce_celeba(n_processes=1, chunk=64):

    # Create shared array
    data = np.load(sys.argv[1])

    elemshape = data.shape[1:]
    shared_arr = mp.RawArray(ctypes.c_float, data.size)
    arr = tonumpyarray(shared_arr).reshape(data.shape)
    np.copyto(arr, data)
    del data

    # Run parallel sum
    t = time()
    pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))

    # Change the code below to compute a step of the reduction
    # ---------------------------8<---------------------------
    max_steps = int(np.ceil(np.log2(arr.shape[0])))
    #print("arr shape", arr.shape)
    #print("max steps", max_steps)

    s = 1
    for step in range(max_steps):
        pool.map(reduce_step, [(i, i + chunk*s, s, elemshape) for i in range(0, len(arr), chunk*s)], chunksize=1)
        s = s * chunk

    # Write output
    run_time = time() - t
    final_image = arr[0]
    final_image /= len(arr)  # For mean
    Image.fromarray((255 * final_image.astype(float)).astype('uint8')).save('result.png')

    return run_time

if __name__ == "__main__":

    max_processes = 10
    times = []
    for n_proc in range(1, max_processes):
        run_time = reduce_celeba(n_proc, chunk=64)
        times.append(run_time)
        print(f"Reduce CelebA w. n_proc = {n_proc} took: {run_time} sec.")

    speedup = [times[0] / time for time in times]  # speed-up is fraction: old_time / new_time

    plt.figure()
    plt.plot(range(1, max_processes), speedup)
    plt.ylabel("Speed-up")
    plt.xlabel("No. of processes")
    plt.savefig("CelebA_speedup_vs_processes.pdf", dpi=300)
