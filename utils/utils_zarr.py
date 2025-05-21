import numpy as np
import zarr
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4
from skimage.transform import downscale_local_mean

def write_ome_multiscale(file_path="ome_array_pyramid.zarr", pyramid_paths=None, chunk_size=(648, 648, 648)):

    # Create/open a Zarr array in write mode
    store = parse_url(file_path, mode="w").store
    root = zarr.group(store=store)

    # Read image pyramid
    image_pyramid = []
    for data_path in pyramid_paths:
        data = np.load(data_path)
        image_pyramid.append(data)

    # Create image group for the volume
    image_group = root.create_group("volume")

    # Define the chunk sizes for each level
    chunk_sizes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    print("Chunk sizes: ", chunk_sizes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    storage_opts = [
        {"chunks": chunk_sizes[i], "compression": Blosc(cname='lz4', clevel=3, shuffle=Blosc.BITSHUFFLE)}
        for i in range(len(image_pyramid))
    ]

    # Write the image data to the Zarr group
    write_multiscale(
            image_pyramid,
            group=image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    print("Done")
