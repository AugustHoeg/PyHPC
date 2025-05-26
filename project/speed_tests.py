import os
from tqdm import tqdm
import numpy as np
import torch
import monai.transforms as mt
from monai.data import SmartCacheDataset, DataLoader
from time import sleep
from time import perf_counter as time

import matplotlib.pyplot as plt

from project.dataloader import ZarrDatasetBaseline
from project.dataset_prefetch_6 import ZarrDataset

def run_speed_test(dataloader, epochs=10, sleep_time=0.1):

    start_time = time()
    time_list = np.zeros((epochs, len(dataloader)))
    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        for j, batch in enumerate(dataloader):
            if sleep_time > 0:
                sleep(sleep_time)  # Assuming some processing time
            batch_time = time()
            time_list[i][j] = batch_time
    time_diff_list = time_list[:, 1:] - time_list[:, :-1]  # Calculate time difference between batches

    #time_list[1] -= start_time
    #time_list.pop(0)

    time_elapsed = time() - start_time
    print(f"Time taken in total {time_elapsed} sec.")

    # Calculate average time per batch
    avg_time_per_epoch = time_elapsed / epochs
    print(f"Average time per epoch: {avg_time_per_epoch} sec.")

    # Calculate average time per batch
    avg_time_per_batch = avg_time_per_epoch / batch_size
    print(f"Average time per batch: {avg_time_per_batch} sec.")

    # Calculate sliding window average
    window_size = 10
    sliding_window_avg = np.convolve(time_diff_list[0, :], np.ones(window_size)/window_size, mode='valid')

    plt.figure()
    plt.plot(time_diff_list[0, :], 'o-')
    plt.xlabel('Batch Number')
    plt.ylabel('Time (s)')
    plt.title('Time taken for each batch')
    plt.show()

    plt.figure()
    plt.plot(sliding_window_avg)
    plt.xlabel('Batch Number')
    plt.ylabel('Sliding Window Average Time (s)')
    plt.title('Sliding Window Average Time taken for each batch')
    plt.show()


def get_dataset(type = "baseline"):

    if type == "baseline":
        print("Using baseline dataset")
        dataset = ZarrDatasetBaseline(ome_levels,
                                      paths,
                                      patch_shape,
                                      patch_transform)

        num_workers = 0
        persistent_workers = True if num_workers > 0 else False
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=False,
                                persistent_workers=persistent_workers)

    else:
        dataset = ZarrDataset(ome_levels,
                              paths,
                              patch_shape,
                              patch_transform,
                              num_producers=8,
                              num_workers=1,
                              queue_size=64,
                              use_LRU_cache=False)


        num_workers = 0
        persistent_workers = True if num_workers > 0 else False
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=False,
                                persistent_workers=persistent_workers)

    return dataset, dataloader

if __name__ == "__main__":

    no_epochs = 100
    batch_size = 4
    patch_shape = (64, 64, 64)
    ome_levels = ['0']  # ['0', '1', '2']
    paths = ["../ome_array_pyramid.zarr"] * 8

    seed = 8883
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define patch transforms
    patch_transform = mt.Compose([
        # mt.Identityd(keys=ome_levels, allow_missing_keys=True),
        mt.EnsureChannelFirstd(keys=ome_levels, channel_dim='no_channel'),
        mt.SignalFillEmptyd(keys=ome_levels, replacement=0),  # Remove any NaNs
        mt.ScaleIntensityd(keys=ome_levels, minv=0.0, maxv=1.0),
        # mt.Rand3DElasticd(keys=ome_levels, prob=0.5, sigma_range=(5, 10), magnitude_range=(0.1, 0.2), mode='bilinear'),
        mt.RandFlipd(keys=ome_levels, prob=0.5, spatial_axis=[0, 1, 2]),
    ])

    dataset, dataloader = get_dataset(type="baseline")

    # Run speed test
    run_speed_test(dataloader, epochs=no_epochs, sleep_time=0)
