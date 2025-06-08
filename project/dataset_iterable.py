import os
import queue
from tqdm import tqdm
import sys
import random
import numpy as np
import torch
import monai.data
import monai.transforms as mt
import zarr
from zarr.storage import DirectoryStore, LRUStoreCache, FSStore, MemoryStore
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader, IterableDataset
from time import sleep
from time import perf_counter as time
from multiprocessing import Process, Queue, Event

class ZarrIterableDataset(IterableDataset):

    def __init__(self, ome_levels, group_name, paths, patch_shape, patch_transform, base_seed=8338, store_type='MemoryStore', num_samples=1000):
        self.group_name = group_name
        self.ome_levels = ome_levels  # Number of levels in the Zarr dataset
        self.paths = paths
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform
        self.base_seed = base_seed
        self.num_samples = num_samples
        self.seed = None

        super().__init__(paths, patch_transform)

        # Check if the paths are valid
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        self.zarr_data = []
        for path in paths:
            if store_type == 'Numpy':
                data = zarr.open(path, mode='r', cache_attrs=True)
                z = {self.group_name: {level: np.array(data[self.group_name][level]) for level in self.ome_levels}}
                self.zarr_data.append(z)
            elif store_type == 'MemoryStore':
                disk_store = DirectoryStore(path)
                memory_store = MemoryStore()
                zarr.copy_store(disk_store, memory_store)
                z = zarr.open(memory_store, mode='r')
                self.zarr_data.append(z)
            elif store_type == 'LRUStoreCache':
                store_size = 2 ** 28  # 256 MB
                cached_store = LRUStoreCache(FSStore(path), max_size=store_size)
                self.zarr_data.append(zarr.open(store=cached_store, mode='r', cache_attrs=True))
            else:
                self.zarr_data.append(zarr.open(path, mode='r', cache_attrs=True))

            store = parse_url(path, mode="r").store
            root = zarr.group(store=store)

            print(root.info)  # Print the metadata of the Zarr group
            print(root.tree())  # Print the structure of the Zarr group

    def _extract_patch_levels(self, data, patch_size=(32, 32, 32)):

        volume = data[self.group_name][self.ome_levels[-1]]
        start = np.random.randint(0, np.array(volume.shape) - patch_size)
        end = start + patch_size
        out_dict = {self.ome_levels[-1]: volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]}

        for i in range(len(self.ome_levels) - 2, -1, -1):  # reverse order
            volume = data[self.group_name][self.ome_levels[i]]
            start = start * 2
            end = end * 2
            out_dict[self.ome_levels[i]] = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return out_dict

    def _generate_patch(self):
        # Randomly select a zarr dataset
        z = random.choice(self.zarr_data)

        # Extract a patch from the selected dataset
        patch = self._extract_patch_levels(z, self.patch_shape)

        if self.patch_transform:
            patch = self.patch_transform(patch)

        return patch

    def set_random_seed(self, seed):
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        monai.utils.misc.set_determinism(seed)
        np.random.RandomState(seed)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            samples_per_worker = self.num_samples
        else:  # in a worker process
            # split workload
            samples_per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))

        if self.seed is None:
            worker_info = torch.utils.data.get_worker_info()
            self.seed = self.base_seed + worker_info.id if worker_info else self.base_seed
            self.set_random_seed(self.seed)

        # Generate patches
        for _ in range(samples_per_worker):
            yield self._generate_patch()

    def __len__(self):
        # Return a large number to ensure the dataset is infinite
        return self.num_samples


def main():

    # Example usage
    batch_size = 8
    patch_shape = (64, 64, 64)

    ome_levels = ['0'] #['0', '1', '2']
    paths = ["../ome_array_pyramid.zarr"] * batch_size
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
                                  store_type='Numpy',
                                  num_samples=1000)

    num_workers = 0
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
        print(f"Epoch {i + 1}/{no_epochs}")
        for batch in tqdm(dataloader, desc='Reconstructing patches\n', mininterval=2):
        #for batch in dataloader:
            pass
            # sleep(0.1)  # Assuming some processing time
            # print("Loaded batch...")
            # for key in batch.keys():
            #     print(f"Key: {key}, Shape: {batch[key].shape}")

    time_elapsed = time() - start_time
    print(f"Time taken {time_elapsed} sec.")
    print(f"Time taken per patch {time_elapsed / no_epochs / batch_size} sec. (average)")

    print(f"Loaded {len(dataset)} items from Zarr dataset.")


if __name__ == "__main__":
    main()