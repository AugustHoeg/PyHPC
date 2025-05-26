import os
import h5py
import dask.array as da
import numpy as np
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import zarr
import numcodecs


def load_and_crop_slice(hdf5_path, dataset_name, z_idx, crop_bounds, dtype=np.float16):
    """Helper to load and crop a single slice (z-index) from HDF5."""
    with h5py.File(hdf5_path, 'r') as f:
        slice_2d = f[dataset_name][z_idx]
        h_start, h_end, w_start, w_end = crop_bounds
        return slice_2d[h_start:h_end, w_start:w_end].astype(dtype)


def hdf5_to_zarr_preprocess(
    hdf5_path,
    hdf5_dataset_name,
    zarr_path,
    crop_bounds,     # (h_start, h_end, w_start, w_end)
    output_chunks=(256, 256, 256),
    compressor=None,
    dtype=np.float16,
    use_dask_cluster=True,  # Use Dask cluster for multiprocessing
):
    # Step 0: Get HDF5 shape
    with h5py.File(hdf5_path, 'r') as f:
        d, h, w = f[hdf5_dataset_name].shape
        print(f"Original HDF5 shape: (D={d}, H={h}, W={w})")

    h_start, h_end, w_start, w_end = crop_bounds
    cropped_h = h_end - h_start
    cropped_w = w_end - w_start

    if use_dask_cluster:
        print("Using Dask cluster for multiprocessing...")
        # Step 1: Start Dask cluster (multiprocessing)
        cluster = LocalCluster(processes=True)
        client = Client(cluster)

    # Step 2: Create lazy list of cropped slices
    lazy_slices = [
        delayed(load_and_crop_slice)(hdf5_path, hdf5_dataset_name, z, crop_bounds, dtype)
        for z in range(d)
    ]
    dask_slices = [
        da.from_delayed(s, shape=(cropped_h, cropped_w), dtype=dtype)
        for s in lazy_slices
    ]
    volume = da.stack(dask_slices, axis=0)  # shape: (D, H_crop, W_crop)

    # Step 3: Compute global min/max
    print("Computing global min/max...")
    with ProgressBar():
        global_min = volume.min().compute()
        global_max = volume.max().compute()
    print(f"Global min: {global_min}, max: {global_max}")

    # Step 4: Normalize
    normalized = (volume - global_min) / (global_max - global_min)
    normalized = normalized.astype(np.float16)

    # Step 5: Write to Zarr with direct output chunking + compression
    print(f"Saving to Zarr at {zarr_path} with chunks={output_chunks}...")
    if compressor is None:
        compressor = numcodecs.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

    # Set encoding: apply chunks and compressor
    encoding = {"": {
        "chunks": output_chunks,
        "compressor": compressor
    }}

    with ProgressBar():
        normalized.to_zarr(zarr_path, overwrite=True, encoding=encoding)

    print("Conversion complete.")
    if use_dask_cluster:
        print("Closing Dask client and cluster...")
        client.close()
        cluster.close()


def hdf5_to_zarr(
    hdf5_path,
    hdf5_dataset_name,
    zarr_path,
    output_chunks=(256, 256, 256),
    compressor=None,
    dtype=np.float16,
    global_min=None,
    global_max=None,
    use_dask_cluster=True,  # Use Dask cluster for multiprocessing
):
    # Step 0: Get HDF5 shape
    with h5py.File(hdf5_path, 'r') as f:
        d, h, w = f[hdf5_dataset_name].shape
        print(f"HDF5 shape: (D={d}, H={h}, W={w})")

    if use_dask_cluster:
        print("Using Dask cluster for multiprocessing...")
        # Step 1: Start Dask cluster (multiprocessing)
        cluster = LocalCluster(processes=True)
        client = Client(cluster)

    data = h5py.File(hdf5_path, 'r')[hdf5_dataset_name]
    volume = da.from_array(data, chunks=(1, h, w))

    if global_min is None or global_max is None:
        # Step 3: Compute global min/max
        print("Computing global min/max...")
        with ProgressBar():
            global_min = volume.min().compute()
            global_max = volume.max().compute()
        print(f"Global min: {global_min}, max: {global_max}")

    # Step 4: Normalize
    normalized = (volume - global_min) / (global_max - global_min)
    normalized = normalized.astype(np.float16)

    # Step 5: Write to Zarr with direct output chunking + compression
    print(f"Saving to Zarr at {zarr_path} with chunks={output_chunks}...")
    if compressor is None:
        compressor = numcodecs.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

    # Set encoding: apply chunks and compressor
    encoding = {"": {
        "chunks": output_chunks,
        "compressor": compressor
    }}

    with ProgressBar():
        normalized.to_zarr(zarr_path, overwrite=True, encoding=encoding)

    print("Conversion complete.")
    if use_dask_cluster:
        print("Closing Dask client and cluster...")
        client.close()
        cluster.close()




# Example usage
if __name__ == "__main__":

    root = "/dtu/3d-imaging-center/projects/2024_DANFIX_130_ExtremeCT/raw_data_extern/2024031208/bone_1_20kev_20x_16bits_20sdd/bin4x4/"
    hdf5_path = os.path.join(root, "scan-6858-6870_recon.h5")

    hdf5_to_zarr(
        hdf5_path=hdf5_path,
        hdf5_dataset_name='/exchange/data',
        zarr_path=os.path.join(root, "cropped_normalized.zarr"),
        output_chunks=(256, 256, 256),
        compressor=numcodecs.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE),
        dtype=np.float16,
        global_min=-0.00032591819763183594,
        global_max=0.00014853477478027344,
        use_dask_cluster=True,  # Set to True to use Dask cluster for multiprocessing
    )

    # hdf5_to_zarr_preprocess(
    #     hdf5_path=hdf5_path,
    #     hdf5_dataset_name='/exchange/data',
    #     zarr_path='cropped_normalized.zarr',
    #     crop_bounds=(0, 512 + 1, 0, 1024 + 1),  # crop each slice from H[0:513], W[0:1025], result becomes (D, 512, 1024)
    #     output_chunks=(256, 256, 256),
    #     compressor=numcodecs.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE),
    #     dtype=np.float16,
    #     use_dask_cluster=False  # Set to True to use Dask cluster for multiprocessing
    # )
