import os
import sys
import math
from time import perf_counter as time
import numpy as np
import matplotlib.pyplot as plt
import blosc

def write_numpy(arr, file_name):
    np.save(f"{file_name}.npy", arr)
    os.sync()


def write_blosc(arr, file_name, cname="zstd"): # cname="zstd" or "lz4"
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    os.sync()

def read_numpy(file_name):
    return np.load(f"{file_name}.npy")


def read_blosc(file_name):
    with open(f"{file_name}.bl", "rb") as r:
        b_arr = r.read()
    return blosc.unpack_array(b_arr)


iterations = 1

n = int(sys.argv[1])
#array = np.zeros((n, n, n), dtype=np.uint8) # Array of zeros
array = np.tile(np.arange(256, dtype='uint8'), (n // 256) * n * n).reshape(n, n, n) # Tiled array of numbers
#array = np.random.randint(0, 256, (n, n, n), dtype=np.uint8) # Array of random integers between 0 and 256

# When array is random, we get no space saving with blosc due to bad compression ratio.

start_time = time()
write_numpy(array, file_name="numpy_arr")
end_time = time()
print(f"Write with numpy took: {(end_time - start_time) / iterations} sec. per iteration")

start_time = time()
write_blosc(array, file_name="blosc_arr")
end_time = time()
print(f"Write with blosc took: {(end_time - start_time) / iterations} sec. per iteration")

start_time = time()
read_numpy("numpy_arr")
end_time = time()
print(f"Read with numpy took: {(end_time - start_time) / iterations} sec. per iteration")

start_time = time()
read_blosc("blosc_arr")
end_time = time()
print(f"Read with blosc took: {(end_time - start_time) / iterations} sec. per iteration")

#