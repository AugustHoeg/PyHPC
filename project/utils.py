import zarr
import numpy as np
from zarr.storage import DirectoryStore, MemoryStore
from ome_zarr.io import parse_url

def cache_zarr(path):
    disk_store = DirectoryStore(path)
    memory_store = MemoryStore()
    zarr.copy_store(disk_store, memory_store)
    z = zarr.open(memory_store, mode='r')
    return z


if __name__ == "__main__":
    import zarr, numpy as np, time
    from zarr.storage import MemoryStore

    # Setup
    np_arr = np.random.rand(128, 128, 128)
    zarr_arr = zarr.array(np_arr, chunks=(64, 64, 64), store=MemoryStore())

    # Slice benchmark
    start = time.time()
    _ = np_arr[32:96, 32:96, 32:96]
    print("Numpy slice: %0.20f" % (time.time() - start))

    start = time.time()
    _ = zarr_arr[32:96, 32:96, 32:96]
    print("Zarr slice: %0.20f" % (time.time() - start))