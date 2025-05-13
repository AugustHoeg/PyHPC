import numpy as np
import zarr
import dask.array as da
from ome_zarr.writer import write_image, write_multiscale
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url
from numcodecs import Zstd, Blosc, LZ4
from skimage.transform import downscale_local_mean

# Create/open a Zarr array in write mode
file_path = "ome_array_pyramid.zarr"

store = parse_url(file_path, mode="w").store
root = zarr.group(store=store)

# Create a random 3D volume
data = np.random.rand(512, 512, 512).astype(np.float32)  # shape (z, y, x)

# Create image pyramid
image_pyramid = [data]
for i in range(2):
    image_pyramid.append(downscale_local_mean(image_pyramid[i], (2, 2, 2)))

# Create image group for the volume
image_group = root.create_group("volume")

# Define storage options for each level
# Compressions: LZ4(), Zstd(level=3)

storage_opts = [
        {"chunks": (256, 256, 256), "compression": LZ4()},
        {"chunks": (128, 128, 128), "compression": LZ4()},
        {"chunks": (64, 64, 64), "compression": LZ4()},
]

# Write the image data to the Zarr group
write_multiscale(
        image_pyramid,
        group=image_group,
        axes=["z", "y", "x"],
        storage_options=storage_opts
    )

print("Done")
