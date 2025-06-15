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

def run_speed_test(dataloader, total_iterations=1000, sleep_time=0.1, sliding_window_size=100, subtract_first_batch=True):

    current_step = 0
    start_time = time()
    time_list = np.zeros(total_iterations)

    batch_size = dataloader.batch_size

    while total_iterations > current_step:
        for batch_idx, batch in enumerate(dataloader):
            # Record the time for this batch
            time_list[current_step] = time()

            current_step += 1
            if current_step % (total_iterations // 100) == 0:
                print(f"Current iteration {current_step}/{total_iterations}")

            if current_step >= total_iterations:
                time_elapsed = time() - start_time
                print(f"Time taken in total {time_elapsed} sec.")
                break

            # Simulate some processing time
            if sleep_time > 0:
                sleep(sleep_time)  # Assuming some processing time

    # subtract the start time from the recorded times
    time_list -= start_time

    # If subtract_first_batch, set the first batch time as zero
    if subtract_first_batch:
        time_list -= time_list[0]

    # Calculate time difference between batches
    time_diff_list = time_list[1:] - time_list[:-1]

    # Calculate average time per batch
    avg_time_per_iteration = time_elapsed / total_iterations
    print(f"Average time per iteration: {avg_time_per_iteration} sec.")

    # Calculate average time per patch
    avg_time_per_patch = avg_time_per_iteration / batch_size
    print(f"Average time per patch: {avg_time_per_patch} sec.")

    # Calculate sliding window average
    sliding_window_avg = np.convolve(time_diff_list, np.ones(sliding_window_size)/sliding_window_size, mode='valid')

    result_dict = {
        'total_iterations': total_iterations,
        'time_elapsed': time_elapsed,
        'batch_size': batch_size,
        'sliding_window_size': sliding_window_size,
        'time_diff_list': time_diff_list,
        'sliding_window_avg': sliding_window_avg,
        'avg_time_per_iteration': avg_time_per_iteration,
        'avg_time_per_patch': avg_time_per_patch
    }

    return result_dict

def save_results(result_dict, output_file="speed_test_results.txt"):
    # Write results to a text file
    with open(output_file, 'w') as f:
        # Experiment name
        f.write(f"Speed test experiment: {os.path.basename(output_file)}\n")
        f.write(f"Total time taken: {result_dict['time_elapsed']} sec.\n")
        f.write(f"Total iterations: {result_dict['total_iterations']}\n")
        f.write(f"Average time per iteration: {result_dict['avg_time_per_iteration']} sec.\n")
        f.write(f"Average time per patch: {result_dict['avg_time_per_patch']} sec.\n")
        #f.write(f"Sliding window size: {sliding_window_size}\n")
        #f.write(f"Time differences: {time_diff_list.tolist()}\n")
        #f.write(f"Sliding window averages: {sliding_window_avg.tolist()}\n")


# def plot_time_plots(result_dict):
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(result_dict['time_diff_list'])
#     plt.xlabel('Iteration')
#     plt.ylabel('Time (s)')
#     plt.title('Time taken for each batch')
#     plt.ylim(0, np.max(result_dict['time_diff_list']) * 1.1)  # Set y-axis limit to 10% above max time
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(result_dict['sliding_window_avg'])
#     plt.xlabel('Iteration')
#     plt.ylabel('Sliding Window Average Time (s)')
#     plt.title('Sliding Window Average Time taken for each batch')
#     plt.ylim(0, np.max(result_dict['sliding_window_avg']) * 1.1)  # Set y-axis limit to 10% above max time
#
#     plt.show()

def plot_time_plots(result_dict, save_path=None, filename_prefix='plot'):
    plt.rcParams.update({'font.family': 'Times New Roman'})

    # First plot: raw time differences
    plt.figure(figsize=(12, 6))
    plt.plot(result_dict['time_diff_list'], color='tab:blue')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Time (s)', fontsize=14)
    plt.title('Time taken for each batch', fontsize=16)
    plt.ylim(0, np.max(result_dict['time_diff_list']) * 1.1)
    plt.grid(True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file1 = os.path.join(save_path, f'{filename_prefix}_time_diff.pdf')
        plt.savefig(file1, dpi=300, bbox_inches='tight')

    # Second plot: sliding window average
    plt.figure(figsize=(12, 6))
    plt.plot(result_dict['sliding_window_avg'], color='tab:orange')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Sliding Window Avg Time (s)', fontsize=14)
    plt.title('Sliding Window Average Time taken for each batch', fontsize=16)
    plt.ylim(0, np.max(result_dict['sliding_window_avg']) * 1.1)
    plt.grid(True)

    if save_path:
        file2 = os.path.join(save_path, f'{filename_prefix}_sliding_avg.pdf')
        plt.savefig(file2, dpi=300, bbox_inches='tight')

    # Third plot: raw time difference and sliding window average together
    plt.figure(figsize=(12, 6))
    plt.plot(result_dict['time_diff_list'], color='tab:blue')
    plt.plot(result_dict['sliding_window_avg'], color='tab:orange')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Time (s)', fontsize=14)
    plt.title('Time taken for each batch', fontsize=16)
    plt.ylim(0, np.max(result_dict['time_diff_list']) * 1.1)
    plt.legend(['Time per batch', 'Sliding Window Avg'], fontsize=12)
    plt.grid(True)

    if save_path:
        file3 = os.path.join(save_path, f'{filename_prefix}_combined.pdf')
        plt.savefig(file3, dpi=300, bbox_inches='tight')
    else:
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
