import numpy as np
import zarr
import dask.array as da
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4

# Create/open a Zarr array in write mode
file_path = "ome_array.zarr"

store = parse_url(file_path, mode="w").store
root = zarr.group(store=store)

# Create a random 3D volume
data = np.random.rand(512, 512, 512).astype(np.float32)  # shape (z, y, x)

# Create image group for the volume
image_group = root.create_group("volume")

# Define storage options for each level
storage_opts = [
        {"chunks": (256, 256, 256), "compression": Zstd(level=5)},
        {"chunks": (128, 128, 128), "compression": Zstd(level=3)},
        {"chunks": (64, 64, 64), "compression": Zstd(level=1)},
]

# Define scaler for downsampling
# THIS ONLY DOWNSAMPLES ALONG X,Y AND NOT Z
scaler = Scaler(copy_metadata=True,
                downscale=2,
                in_place=False,
                labeled=False,
                max_layer=len(storage_opts) - 1,  # Must match the number of storage options minus 1
                method='local_mean')

# Write the image data to the Zarr group
write_image(data,
            group=image_group,
            axes="zyx",
            storage_options=storage_opts,
            scaler=scaler,
            )

print("Done")
