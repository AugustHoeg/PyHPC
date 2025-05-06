import ctypes
import multiprocessing as mp
import sys
from time import perf_counter as time
import numpy as np
from PIL import Image


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

if __name__ == '__main__':
    n_processes = 1
    chunk = 64

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
    print("arr shape", arr.shape)
    print("max steps", max_steps)

    s = 1
    for step in range(max_steps):
        pool.map(reduce_step, [(i, i + chunk*s, s, elemshape) for i in range(0, len(arr), chunk*s)], chunksize=1)
        s = s * chunk

    if sys.argv[1].find("dummydata") > 0:
        print("data after reduction", arr.reshape(-1))
    else:
        arr /= arr.shape[0]  # take the mean
        # Write output
        print(time() - t)
        final_image = arr[0]
        # final_image /= len(arr) # For mean
        Image.fromarray(
            (255 * final_image.astype(float)).astype('uint8')
        ).save('result.png')