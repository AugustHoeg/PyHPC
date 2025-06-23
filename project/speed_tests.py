import os
import numpy as np
from time import sleep
from time import perf_counter as time
import matplotlib.pyplot as plt


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
    avg_time_per_iteration = np.mean(time_diff_list)  # time_elapsed / total_iterations
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

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Write results to a text file
    with open(output_file, 'w') as f:
        # Experiment name
        f.write(f"Speed test experiment: {os.path.basename(output_file)}\n")
        f.write(f"Total time taken: {result_dict['time_elapsed']} sec.\n")
        f.write(f"Total iterations: {result_dict['total_iterations']}\n")
        f.write(f"Average time per iteration: {result_dict['avg_time_per_iteration']} sec.\n")
        f.write(f"Average time per iteration: {result_dict['avg_time_per_iteration'] * 1000:.2f} ms\n")
        f.write(f"Average time per patch: {result_dict['avg_time_per_patch']} sec.\n")
        f.write(f"Average time per patch: {result_dict['avg_time_per_patch'] * 1000:.2f} ms\n")

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
    plt.rcParams.update({'font.family': 'Times'})

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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_path:
        file3 = os.path.join(save_path, f'{filename_prefix}_combined.pdf')
        plt.savefig(file3, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_time_plots_multi(result_dict_list, save_path=None, filename_prefix='plot', max_iterations=1000):
    plt.rcParams.update({'font.family': 'Times'})

    # First plot: raw time differences
    plt.figure(figsize=(6, 6))
    for data in result_dict_list:
        plt.plot(data['result_dict']['time_diff_list'][:max_iterations])
        #plt.plot(result_dict_list['result_dict']['sliding_window_avg'])
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Time [s]', fontsize=18)
    plt.title(f"Time per iteration, patch size: ${data['patch_size']}^3$", fontsize=16)
    plt.ylim(0, np.max(data['result_dict']['time_diff_list']) * 1.1)
    plt.legend([f"Chunks: ${data['chunk_size']}^3$" for data in result_dict_list], fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file1 = os.path.join(save_path, f'{filename_prefix}_time_diff_multi.pdf')
        plt.savefig(file1, dpi=300, bbox_inches='tight')

    # Second plot: sliding window average
    plt.figure(figsize=(6, 6))
    for data in result_dict_list:
        plt.plot(data['result_dict']['sliding_window_avg'][:max_iterations])
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Time [s]', fontsize=18)
    plt.title(f"Time per iteration, patch size: ${data['patch_size']}^3$", fontsize=20)
    plt.ylim(0, np.max(data['result_dict']['sliding_window_avg']) * 1.1)
    plt.legend([f"Chunks: ${data['chunk_size']}^3$" for data in result_dict_list], fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if save_path:
        file2 = os.path.join(save_path, f'{filename_prefix}_sliding_avg_multi.pdf')
        plt.savefig(file2, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_analysis_num_workers_no_cache(result_dict_list, save_path=None, filename_prefix='plot_num_workers_no_cache'):

    plt.rcParams.update({'font.family': 'Times'})

    # First plot: avg time per iteration
    times = [data['result_dict']['avg_time_per_iteration'] for data in result_dict_list]
    num_workers_list = [data['num_workers'] for data in result_dict_list]
    speedup = [times[0] / time for time in times]  # speed-up is fraction: old_time / new_time

    plt.figure(figsize=(12, 6))
    plt.plot(num_workers_list, speedup, marker='o', linestyle='-')
    plt.xlabel('No. of dataloader processes', fontsize=14)
    plt.ylabel('Speed-up', fontsize=14)
    plt.title('Average iteration time speed-up vs no. of dataloader processes', fontsize=16)
    plt.xticks(num_workers_list, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file1 = os.path.join(save_path, f'{filename_prefix}_iteration_speedup_no_cache.pdf')
        plt.savefig(file1, dpi=300, bbox_inches='tight')


def plot_analysis_compression(result_dict_list, save_path=None, filename_prefix='plot_compression_analysis'):

    # Compute disk usage for each compression method


    plt.rcParams.update({'font.family': 'Times'})

    comp_names = [data['compression'] for data in result_dict_list]
    disk_usage = [data['disk_usage'] for data in result_dict_list]
    times = [data['result_dict']['avg_time_per_iteration'] for data in result_dict_list]

    min_usage = min(disk_usage)
    max_usage = max(disk_usage)

    # # Scale dot sizes from disk usage (optional normalization for aesthetics)
    # min_size = 100
    # max_size = 500
    # dot_sizes = [
    #     min_size + (usage - min_usage) / (max_usage - min_usage) * (max_size - min_size)
    #     for usage in disk_usage
    # ]

    plt.figure(figsize=(12, 6))
    for usage, time in zip(disk_usage, times):
        plt.scatter(usage, time, s=150, alpha=0.7)

    plt.xlabel('Total disk usage [MB]', fontsize=14)
    plt.ylabel('Average iteration time [s]', fontsize=14)
    plt.title('Average iteration time vs. disk usage', fontsize=16)
    plt.xlim([min_usage - min_usage * 0.1, max_usage + max_usage * 0.1])
    plt.ylim([min(times) - min(times) * 0.1, max(times) + max(times) * 0.1])
    plt.legend(comp_names, fontsize=12, loc='upper right', title='Compression Method')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.grid(True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file1 = os.path.join(save_path, f'{filename_prefix}.pdf')
        plt.savefig(file1, dpi=300, bbox_inches='tight')




def plot_analysis_num_workers_with_cache(result_dict_list, save_path=None, filename_prefix='plot_num_workers_no_cache'):

    plt.rcParams.update({'font.family': 'Times'})

    data_workers = np.unique([data['dataloader_workers'] for data in result_dict_list])
    prod_workers = np.unique([data['producer_workers'] for data in result_dict_list])
    time_matrix = np.zeros((len(data_workers), len(prod_workers)))

    c = 0
    for i in range(len(data_workers)):
        for j in range(len(prod_workers)):
            time_matrix[i, j] = result_dict_list[c]['result_dict']['avg_time_per_iteration']
            c += 1

    # First plot: avg time per iteration
    speedup_matrix = time_matrix[0, 0] / time_matrix  # speed-up is fraction: old_time / new_time

    # Make 3d plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(data_workers, prod_workers, speedup_matrix)
    ax.set_xlabel('No. of dataloader processes', fontsize=16)
    ax.set_ylabel('No. of producer threads', fontsize=16)
    ax.set_zlabel('Speed-up', fontsize=16)
    ax.set_title('Average iteration time speed-up vs no. of dataloader and producer processes', fontsize=16)
    ax.set_xticks(data_workers)
    ax.set_yticks(prod_workers)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file1 = os.path.join(save_path, f'{filename_prefix}_iteration_speedup_with_cache.pdf')
        plt.savefig(file1, dpi=300)

    # Create bar plot in 3D
    z_min = 0.8
    z_max = speedup_matrix.max() + speedup_matrix.max() * 0.1  # Add 10% margin for aesthetics
    _xx, _yy = np.meshgrid(data_workers, prod_workers)
    x, y = _xx.ravel(), _yy.ravel()
    top = speedup_matrix.ravel() - z_min
    bottom = np.ones_like(top)
    width = prod_workers[1] - prod_workers[0]
    depth = data_workers[1] - data_workers[0]

    # Shift bars to center on x/y tick values
    x_centered = x - width / 2
    y_centered = y - depth / 2

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.bar3d(x_centered, y_centered, bottom, width, depth, top, shade=True)
    ax.set_xlabel('No. of dataloader processes', fontsize=16)
    ax.set_ylabel('No. of producer threads', fontsize=16)
    ax.set_zlabel('Speed-up', fontsize=16)
    ax.set_title('Average iteration time speed-up vs no. of dataloader and producer processes', fontsize=16)
    ax.set_xticks(data_workers)
    ax.set_yticks(prod_workers)
    ax.set_zlim([z_min, z_max])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file1 = os.path.join(save_path, f'{filename_prefix}_iteration_speedup_with_cache_barplot.pdf')
        plt.savefig(file1, dpi=300)


# def get_dataset(type = "baseline"):
#
#     if type == "baseline":
#         print("Using baseline dataset")
#         dataset = ZarrDatasetBaseline(ome_levels,
#                                       paths,
#                                       patch_shape,
#                                       patch_transform)
#
#         num_workers = 0
#         persistent_workers = True if num_workers > 0 else False
#         dataloader = DataLoader(dataset,
#                                 batch_size=batch_size,
#                                 shuffle=False,
#                                 num_workers=num_workers,
#                                 pin_memory=False,
#                                 persistent_workers=persistent_workers)
#
#     else:
#         dataset = ZarrDataset(ome_levels,
#                               paths,
#                               patch_shape,
#                               patch_transform,
#                               num_producers=8,
#                               num_workers=1,
#                               queue_size=64,
#                               use_LRU_cache=False)
#
#
#         num_workers = 0
#         persistent_workers = True if num_workers > 0 else False
#         dataloader = DataLoader(dataset,
#                                 batch_size=batch_size,
#                                 shuffle=False,
#                                 num_workers=num_workers,
#                                 pin_memory=False,
#                                 persistent_workers=persistent_workers)
#
#     return dataset, dataloader
#
# if __name__ == "__main__":
#
#     no_epochs = 100
#     batch_size = 4
#     patch_shape = (64, 64, 64)
#     ome_levels = ['0']  # ['0', '1', '2']
#     paths = ["../ome_array_pyramid.zarr"] * 8
#
#     seed = 8883
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     # Define patch transforms
#     patch_transform = mt.Compose([
#         # mt.Identityd(keys=ome_levels, allow_missing_keys=True),
#         mt.EnsureChannelFirstd(keys=ome_levels, channel_dim='no_channel'),
#         mt.SignalFillEmptyd(keys=ome_levels, replacement=0),  # Remove any NaNs
#         mt.ScaleIntensityd(keys=ome_levels, minv=0.0, maxv=1.0),
#         # mt.Rand3DElasticd(keys=ome_levels, prob=0.5, sigma_range=(5, 10), magnitude_range=(0.1, 0.2), mode='bilinear'),
#         mt.RandFlipd(keys=ome_levels, prob=0.5, spatial_axis=[0, 1, 2]),
#     ])
#
#     dataset, dataloader = get_dataset(type="baseline")
#
#     # Run speed test
#     run_speed_test(dataloader, epochs=no_epochs, sleep_time=0)
