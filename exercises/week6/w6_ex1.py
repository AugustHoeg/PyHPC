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
    #print("arr", arr.reshape(-1))
    # Change the code below to compute a step of the reduction
    # ---------------------------8<---------------------------
    arr[b] = np.sum(arr[b:e:s])  # sum of neighbouring elements


if __name__ == '__main__':
    n_processes = 1
    chunk = 2

    # Create shared array
    data = np.load(sys.argv[1])
    #print("data before reduction", data.reshape(-1))

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

    print("elemshape", elemshape)
    print("arr", arr.shape)
    print(int(np.log2(len(arr))))
    s = 1
    for step in range(4):
        pool.map(reduce_step, [(i, i + chunk*s, s, elemshape) for i in range(0, len(arr), chunk*s)], chunksize=1)
        print("data after reduction", arr.reshape(-1), "s", s)
        s = s * chunk

    if sys.argv[1].find("dummydata") > 0:
        print("data after reduction", arr.reshape(-1))
    else:
        # Write output
        print(time() - t)
        final_image = arr[0]
        # final_image /= len(arr) # For mean
        Image.fromarray(
            (255 * final_image.astype(float)).astype('uint8')
        ).save('result.png')