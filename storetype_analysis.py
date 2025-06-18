import os
import glob
import datetime
import torch
import numpy as np
from project.dataset_iterable import ZarrIterableDataset
from monai.data import DataLoader
import monai.transforms as mt
from project.speed_tests import run_speed_test, plot_time_plots, save_results, plot_time_plots_multi


if __name__ == "__main__":

    # Example usage
    batch_size = 1
    chunk_size = 128

    for store_type in ['MemoryStore', 'Numpy']:

        result_dict_list = []

        for patch_size in [16, 32, 64, 128, 256]:
            patch_shape = (patch_size, patch_size, patch_size)

            paths = sorted(glob.glob(os.path.join("data", "MRI_images", f"{chunk_size}", "*_CT.zarr")))
            print(f"Testing with OME-Zarr patch size: {patch_shape} with store type: {store_type}")

            # Run speed test
            ome_levels = ['0']  # ['0', '1', '2']
            group_name = "volume"

            seed = 8883
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Define patch transforms

            patch_transform = mt.Compose([
                mt.Identityd(keys=ome_levels, allow_missing_keys=True),
                # mt.EnsureChannelFirstd(keys=ome_levels, channel_dim='no_channel'),
                # mt.SignalFillEmptyd(keys=ome_levels, replacement=0),  # Remove any NaNs
                # mt.ScaleIntensityd(keys=ome_levels, minv=0.0, maxv=1.0),
                # #mt.Rand3DElasticd(keys=ome_levels, prob=0.5, sigma_range=(5, 10), magnitude_range=(0.1, 0.2), mode='bilinear'),
                # mt.RandFlipd(keys=ome_levels, prob=0.5, spatial_axis=[0, 1, 2]),
            ])

            dataset = ZarrIterableDataset(ome_levels,
                                          group_name,
                                          paths,
                                          patch_shape,
                                          patch_transform,
                                          store_type=store_type,
                                          num_samples=1100)

            num_workers = 0
            persistent_workers = True if num_workers > 0 else False
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    persistent_workers=persistent_workers)

            total_iterations = 1100
            result_dict = run_speed_test(dataloader,
                                         total_iterations=1100,
                                         sleep_time=0,
                                         sliding_window_size=100,
                                         subtract_first_batch=True)

            result_dict_list.append({'chunk_size': chunk_size, 'patch_size': patch_size, 'result_dict': result_dict})

            # plot_time_plots(result_dict, save_path="../figures", filename_prefix=f"chunksize_analysis_patch_size_{patch_size}_chunk_size_{chunk_size}")
            current_time = datetime.datetime.now().strftime("%d-%m-%Y")
            out_file = f"results/storetype_analysis/storetype_analysis_patch_size_{patch_size}_storetype_{store_type}_{current_time}.txt"
            save_results(result_dict, output_file=out_file)

            print(f"Results saved to {out_file}")

        #plot_time_plots_multi(result_dict_list, save_path="figures", filename_prefix=f"chunksize_analysis_patch_size_{patch_size}")