from time import perf_counter as time
import zarr
import numpy as np

from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader

def get_ome_data(file_path, directory=None, level=0):
    # Reading using ome-zarr reader
    if directory is not None:
        data_path = file_path + "/" + directory
    reader = Reader(parse_url(data_path, mode="r"))
    # nodes may include images, labels etc
    nodes = list(reader())
    print("nodes: \n", nodes)
    # first node will be the image pixel data
    image_node = nodes[level]
    dask_data = image_node.data
    return dask_data



def playground_01():
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

if __name__ == "__main__":

    # playground_01()

    # Read OME Zarr
    file_path = "ome_array_pyramid.zarr"

    store = parse_url(file_path, mode="r").store
    root = zarr.group(store=store)

    print(root.info)  # Print the metadata of the Zarr group
    print(root.tree())  # Print the structure of the Zarr group

    # Reading directly using zarr:
    root = zarr.open(file_path, mode="r")

    # Access multiscale image levels
    level0 = root['volume']['0']
    level1 = root['volume']['1']
    level2 = root['volume']['2']

    data = get_ome_data(file_path, directory='volume', level=0)

    patch_size = (64, 64, 64)
    valid_range = np.array(level0.shape) - patch_size
    no_samples = 100

    start_time = time()
    for i in range(no_samples):
        if i % 10 == 0:
            print("Reading patch ", i)
        # randomly sample a patch
        crop_start = np.random.randint(0, np.array(level0.shape) - patch_size)  #  (0,0,0)
        crop_end = crop_start + patch_size

        patch = level0[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
        #patch = data[0][crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]].compute()
    time_elapsed = time() - start_time
    print(f"Time taken {time_elapsed} sec.")
    print(f"Time taken per patch {time_elapsed / no_samples} sec. (average)")

    print("Done")
