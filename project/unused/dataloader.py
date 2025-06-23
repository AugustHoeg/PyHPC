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
from zarr.storage import LRUStoreCache, FSStore
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader
from time import sleep
from time import perf_counter as time
from multiprocessing import Process, Queue, Event

class ZarrDatasetBaseline(monai.data.Dataset):
    def __init__(self, ome_levels, paths, patch_shape, patch_transform):

        self.ome_levels = ome_levels  # Number of levels in the Zarr dataset
        self.paths = paths
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform

        super().__init__(paths, patch_transform)

        # Check if the paths are valid
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        self.zarr_data = []
        for path in paths:
            self.zarr_data.append(zarr.open(path, mode='r', cache_attrs=True))
            store = parse_url(path, mode="r").store
            root = zarr.group(store=store)
            print(root.info)  # Print the metadata of the Zarr group
            print(root.tree())  # Print the structure of the Zarr group

    def _extract_patch_levels(self, data, patch_size=(32, 32, 32)):

        volume = data['volume'][self.ome_levels[-1]]
        start = np.random.randint(0, np.array(volume.shape) - patch_size)
        end = start + patch_size
        out_dict = {self.ome_levels[-1]: volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]}

        for i in range(len(self.ome_levels) - 2, -1, -1):  # reverse order
            volume = data['volume'][self.ome_levels[i]]
            start = start * 2
            end = end * 2
            out_dict[self.ome_levels[i]] = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return out_dict

    def __getitem__(self, index):

        patch = self._extract_patch_levels(self.zarr_data[index], self.patch_shape)

        # Apply the transformation
        if self.patch_transform:
            patch = self.patch_transform(patch)

        return patch

    def __len__(self):
        return len(self.paths)
