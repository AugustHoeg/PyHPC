import numpy as np
import h5py
import multiprocessing as mp
import datetime

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _crop_task(h5file, frame_idx, crop_window, data_path):
    """
    Runs in each worker process:
    - opens the HDF5,
    - reads frame frame_idx,
    - applies window and dead-pixel fix,
    - runs geometry_correction,
    - returns corrected slice.
    """

    h_start, h_end, w_start, w_end = crop_window
    with h5py.File(h5file, 'r') as f:
        frame = f[data_path][frame_idx, h_start:h_end, w_start:w_end]
    min_val = np.min(frame)
    max_val = np.max(frame)
    return (frame, frame_idx, min_val, max_val)

def crop_hdf5(
        h5file,
        nworkers,
        worker_task=_crop_task,
        crop_bounds=(0, 100, 0, 100),
        write_file=None,
        data_path='exchange/data',
        dtype=np.float16,
        ret=False,
):
    # Step 0: Get HDF5 shape
    with h5py.File(h5file, 'r') as f:
        D, H, W = f[data_path].shape
        print(f"HDF5 shape: (D={D}, H={H}, W={W})")

    # Calculate slice shape based on crop_window
    h_start, h_end, w_start, w_end = crop_bounds
    slice_shape = (h_end - h_start, w_end - w_start)

    if ret:
        imstack = np.zeros((D, slice_shape[0], slice_shape[1]), dtype=dtype)

    # pre-create output dataset if needed
    if write_file:
        with h5py.File(write_file, 'a') as df:
            if data_path not in df:
                df.create_dataset(
                    data_path,
                    shape=(D, slice_shape[0], slice_shape[1]),
                    dtype=dtype,
                    chunks=(1, slice_shape[0], slice_shape[1])
                )
            print(f"Created write file {write_file} with shape {df[data_path].shape}")

    pool = mp.Pool(nworkers)
    results = []

    # submit tasks for all workers
    for frame_idx in range(nworkers):
        args = (h5file, frame_idx, crop_bounds, data_path)
        results.append(pool.apply_async(worker_task, args))

    read_count = nworkers
    write_count = 0

    global_min = np.inf
    global_max = -np.inf

    while write_count < D:

        frame, frame_idx, min_val, max_val = results[0].get()
        results.pop(0)
        print(f"{timestamp()} – Frame {frame_idx +1}/{D}", end='\r')

        # update global min/max
        global_min = min(global_min, min_val)
        global_max = max(global_max, max_val)

        if write_file:
            with h5py.File(write_file, 'a') as df:
                df[data_path][frame_idx, :, :] = frame

        if ret:
            imstack[frame_idx, :, :] = frame

        write_count += 1

        # submit next task if available
        if read_count < D:
            args = (h5file, read_count, crop_bounds, data_path)
            results.append(pool.apply_async(worker_task, args))
            read_count += 1

    print("write_count", write_count)
    print("Number of slices", D)
    print(f"Global min: {global_min}, Global max: {global_max}")

    return imstack, global_min, global_max if ret else None, global_min, global_max
