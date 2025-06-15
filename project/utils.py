import zarr
import numpy as np
from zarr.storage import DirectoryStore, MemoryStore
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale, write_multiscale_labels
from numcodecs import Blosc


def cache_zarr(path):
    disk_store = DirectoryStore(path)
    memory_store = MemoryStore()
    zarr.copy_store(disk_store, memory_store)
    z = zarr.open(memory_store, mode='r')
    return z


def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), cname='lz4'):

    # Define the chunk sizes for each level
    chunk_sizes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    print("Chunk sizes: ", chunk_sizes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    # for Blosc, use cname='zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy'
    storage_opts = [
        {"chunks": chunk_sizes[i], "compression": Blosc(cname=cname, clevel=3, shuffle=Blosc.BITSHUFFLE)}
        for i in range(len(image_pyramid))
    ]

    # Write the image data to the Zarr group
    write_multiscale(
            image_pyramid,
            group=image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    if label_pyramid is not None:
        # Now write the label pyramid under /volume/labels/mask/
        write_multiscale_labels(
            label_pyramid,
            group=image_group,
            name="mask",
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )


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