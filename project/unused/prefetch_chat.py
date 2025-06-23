import os
import numpy as np
import zarr
from ome_zarr.io import parse_url
from multiprocessing import Process, Queue
import monai

class ZarrDataset(monai.data.Dataset):
    def __init__(self, opt, paths, transform, num_workers: int = 0, cache_size: int = 10):
        self.opt = opt
        self.paths = paths
        self.transform = transform
        self.num_workers = num_workers
        self.cache_size = cache_size

        super().__init__(paths, transform)

        # Check if the paths are valid
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        self.zarr_data = []
        for path in paths:
            self.zarr_data.append(zarr.open(path, mode='r'))

        # Initialize multiprocessing queue and cache
        self.queue = Queue(maxsize=cache_size)

        # Start worker processes
        self.workers = []
        for _ in range(num_workers):
            worker = Process(target=self._worker_process)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _worker_process(self):
        while True:
            for path in self.paths:
                zarr_data = zarr.open(path, mode='r')
                patch = self._extract_patch(zarr_data)
                if self.transform:
                    patch = self.transform(patch)
                self.queue.put(patch)

    def _extract_patch(self, zarr_data):
        # Example: Extract a random 3D patch from the Zarr dataset
        patch_size = (32, 32, 32)
        volume = zarr_data['volume']
        start = [np.random.randint(0, dim - size) for dim, size in zip(volume.shape, patch_size)]
        end = [s + size for s, size in zip(start, patch_size)]
        patch = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return patch

    def __getitem__(self, index):

        if self.queue.empty():
            # Wait until the queue is not empty
            while self.queue.empty():
                continue

        # Get the patch from the queue
        patch = self.queue.get()
        return patch

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    # Example usage
    paths = ["../ome_array_pyramid.zarr"] * 8
    dataset = ZarrDataset(opt=None, paths=paths, transform=None, num_workers=4, cache_size=64)

    for i in range(len(dataset)):
        patch = dataset[i]
        print(f"Patch {i}: {patch.shape}")