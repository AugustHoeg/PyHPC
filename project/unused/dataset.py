import os
import sys
import numpy as np
import torch
import monai
import zarr
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader
from time import perf_counter as time

class ZarrDataset(monai.data.Dataset):
    def __init__(self, opt, paths, transform, num_workers: int = 0):
        self.opt = opt
        self.paths = paths
        self.transform = transform
        self.num_workers = num_workers

        super().__init__(paths, transform)

        # Check if the paths are valid
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        self.zarr_data = []
        for path in paths:
            self.zarr_data.append(zarr.open(path, mode='r'))

            store = parse_url(path, mode="r").store
            root = zarr.group(store=store)

            print(root.info)  # Print the metadata of the Zarr group
            print(root.tree())  # Print the structure of the Zarr group

        # Apply deterministic transforms?

    def __getitem__(self, index):

        nlevels = 3

        # Load the Zarr file
        zarr_data = zarr.open(self.paths[index], mode='r')

        # Load the data from the Zarr file
        level0 = zarr_data['volume'][f'{0}']
        level1 = zarr_data['volume'][f'{1}']
        level2 = zarr_data['volume'][f'{2}']

        # # Apply the transformation
        # if self.transform:
        #     data = self.transform(data)

        patch_size = (32, 32, 32)
        crop_start = np.zeros((nlevels, 3), dtype=int)
        crop_end = np.zeros((nlevels, 3), dtype=int)

        crop_start[2, :] = np.random.randint(0, np.array(level2.shape) - patch_size)  # (0,0,0)
        crop_end[2, :] = crop_start[2, :] + patch_size

        crop_start[1, :] = crop_start[2, :] * 2
        crop_end[1, :] = crop_end[2, :] * 2

        crop_start[0, :] = crop_start[1, :] * 2
        crop_end[0, :] = crop_end[1, :] * 2

        patch0 = level0[crop_start[0, 0]:crop_end[0, 0], crop_start[0, 1]:crop_end[0, 1], crop_start[0, 2]:crop_end[0, 2]]
        #patch1 = level1[crop_start[1, 0]:crop_end[1, 0], crop_start[1, 1]:crop_end[1, 1], crop_start[1, 2]:crop_end[1, 2]]
        #patch2 = level2[crop_start[2, 0]:crop_end[2, 0], crop_start[2, 1]:crop_end[2, 1], crop_start[2, 2]:crop_end[2, 2]]

        return patch0 #{"L0": patch0, "L1": patch1, "L2": patch2}


if __name__ == "__main__":

    #images = ["sample1.zarr", "sample2.zarr"]
    #paths = [{"H": img_HR, "L": img_LR} for img_HR, img_LR, in zip(images, self.LR_train)]

    # Example usage
    opt = None
    batch_size = 8
    paths = ["../ome_array_pyramid.zarr"] * batch_size
    transform = None  # monai.transforms.Identityd(keys=[], allow_missing_keys=True)  # Define your transformation here

    seed = 8883
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ZarrDataset(opt, paths, transform)

    num_workers = 1
    persistent_workers = True if num_workers > 0 else False
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            persistent_workers=persistent_workers)

    no_epochs = 10

    start_time = time()
    for i in range(no_epochs):
        for batch in dataloader:
            print("Loaded batch...")
            #for key in batch.keys():
            #    print(f"Key: {key}, Shape: {batch[key].shape}")
    time_elapsed = time() - start_time
    print(f"Time taken {time_elapsed} sec.")
    print(f"Time taken per patch {time_elapsed / no_epochs / batch_size} sec. (average)")


    print(f"Loaded {len(dataset)} items from Zarr dataset.")
