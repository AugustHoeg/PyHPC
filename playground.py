import zarr
import numpy as np

from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


if __name__ == "__main__":

    # Create a Zarr array in write mode
    data_shape = (100, 100, 100)  # Order is assumed Z, Y, X
    chunk_shape = (20, 20, 20)  # Order is assumed Z, Y, X
    array = zarr.open('array.zarr', mode='w', shape=data_shape, chunks=chunk_shape, dtype=np.float32)
    print(array.info)

    # print sizes
    print(array.shape)
    print(array.chunks)
    print(array.cdata_shape)

    # Test OME-Zarr
    path = "ome_array.zarr"

    rng = np.random.default_rng(0)
    data = rng.poisson(lam=10, size=data_shape).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes="zyx", storage_options=dict(chunks=chunk_shape), compute=True)

    print("Done")
