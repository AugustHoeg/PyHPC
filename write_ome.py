import numpy as np
import zarr
import dask.array as da
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4

# Create/open a Zarr array in write mode
store = parse_url("array.zarr", mode="w").store
root = zarr.group(store=store)

# Create a random 3D volume
data = np.random.rand(512, 512, 512).astype(np.float32)  # shape (z, y, x)

# Create image group for the volume
image_group = root.create_group("volume")

# Define storage options for each level
storage_opts = [
        {"chunks": (256, 256, 256), "compressor": Zstd(level=5)},
        {"chunks": (128, 128, 128), "compressor": Zstd(level=3)},
        {"chunks": (64, 64, 64), "compressor": Zstd(level=1)},
    ]

# Define scaler for downsampling
scaler = Scaler(copy_metadata=True,
                downscale=2,
                in_place=False,
                labeled=False,
                max_layer=len(storage_opts),
                method='local_mean')

# Write the image data to the Zarr group
write_image(data,
            group=root,
            axes="zyx",
            storage_options=storage_opts,
            scaler=scaler,
            )

print("Done")

















# Example 3D volume wrapped into (z, y, x)
img = da.random.random((512, 512, 512), chunks=(256, 256, 256))

# Define scale factors for each level (N = 2 gives 3 total scales)
scale_factors = [[2, 2, 2], [2, 2, 2]]  # From level 0 to 2

# Define different chunks for each resolution level
chunks = [
    (256, 256, 256),  # Level 0
    (1, 1, 128, 128, 128),  # Level 1
    (1, 1, 64, 64, 64),     # Level 2
]

# Instantiate the scaler
scaler = Scaler(copy_metadata=True,
                downscale=2,
                in_place=False,
                labeled=False,
                max_layer=4,
                method='local_mean')

# Write with local mean downsampling and per-scale chunking
write_image(
    image=img,
    group="ome_zarr.zarr",
    axes=["t", "c", "z", "y", "x"],
    scale_factors=scale_factors,
    chunks=chunks,
    scaler=scaler,
)
