import os
import re
import glob
import numpy as np
import nibabel as nib
import zarr
from zarr.storage import DirectoryStore
from skimage.transform import downscale_local_mean
from project.utils import write_ome_pyramid

def write_ome_MRI(chunk_size=(256, 256, 256), group_name="volume", cname='lz4', file_end=""):

    """
    Write MRI images to OME-Zarr format.

    :param chunk_size: Defines the chunk size for the OME-Zarr at level '0' e.g. top level. For example (256, 256, 256)
    :param group_name: Defines the group name in the OME-Zarr structure, e.g., "volume"
    :param cname: Defines compression algorithm of Blosc, can be cname='zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy'
    :return:
    """

    # Read MRI images
    image_paths = sorted(glob.glob(os.path.join("data", "MRI_images", "*", "1", "NIFTI", "image.nii.gz")))
    image_filenames = [re.compile("...._CT").search(path).group(0) for path in image_paths]

    os.makedirs(f"data/MRI_images/{chunk_size[0]}/", exist_ok=True)

    for image_path, image_filename in zip(image_paths, image_filenames):
        # Create/open a Zarr array in write mode
        file_path = f"data/MRI_images/{chunk_size[0]}/{image_filename}{file_end}.zarr"

        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping...")
            continue

        store = DirectoryStore(file_path)
        root = zarr.group(store=store)

        # Read the NIFTI image
        nifti = nib.load(image_path)
        data_nifti = nifti.get_fdata()  # shape (z, y, x)

        # Convert to C-order
        data = np.ascontiguousarray(data_nifti).astype(np.float32)
        # Print shape of the data
        print(f"Image shape: {data.shape}")

        # Crop to 512^3 if larger
        data = data[:512, :512, :512]

        # Create image pyramid using downscale_local_mean
        image_pyramid = [data]
        for i in range(2):
            image_pyramid.append(downscale_local_mean(image_pyramid[i], (2, 2, 2)))

        # Create image group for the volume
        image_group = root.create_group(group_name)

        write_ome_pyramid(
            image_group=image_group,
            image_pyramid=image_pyramid,
            label_pyramid=None,  # No labels for MRI
            chunk_size=chunk_size,
            cname=cname  # Compression codec
        )

        print(f"Done writing {image_filename} to OME-Zarr format at {file_path}")

    print("Done")


if __name__ == "__main__":

    import qim3d
    image_path = "data/MRI_images/0190_CT/1/NIFTI/image.nii.gz"
    nifti = nib.load(image_path)
    data_nifti = nifti.get_fdata()  # shape (z, y, x)
    data = np.ascontiguousarray(data_nifti).astype(np.float32)
    qim3d.io.export_ome_zarr("data/MRI_images/0190_CT_qim.zarr", data, chunk_size=256, downsample_rate=2, order=1, replace=False, method='scaleZYX')

    # # Example usage of different chunk sizes
    # write_ome_MRI(chunk_size=(256, 256, 256), group_name="volume")  # Adjust chunk size and group name as needed
    # write_ome_MRI(chunk_size=(128, 128, 128), group_name="volume")  # Adjust chunk size and group name as needed
    # write_ome_MRI(chunk_size=(64, 64, 64), group_name="volume")  # Adjust chunk size and group name as needed

    # Example usage of different compression algorithms
    # write_ome_MRI(chunk_size=(128, 128, 128), group_name="volume", cname='lz4', file_end="_lz4")
    # write_ome_MRI(chunk_size=(128, 128, 128), group_name="volume", cname='zstd', file_end="_zstd")
    # write_ome_MRI(chunk_size=(128, 128, 128), group_name="volume", cname='blosclz', file_end="_blosclz")
    # write_ome_MRI(chunk_size=(128, 128, 128), group_name="volume", cname='zlib', file_end="_zlib")
    # write_ome_MRI(chunk_size=(128, 128, 128), group_name="volume", cname='lz4hc', file_end="_lz4hc")

    print("Done")

