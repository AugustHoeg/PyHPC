import os
import sys
import numpy as np
import torch
import monai
import zarr
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader
from time import sleep
from time import perf_counter as time
from multiprocessing import Process, Queue

class Producer():
    def __init__(self, cache_size: int = 64, num_workers: int = 1):
        super().__init__()

        # Each worker will have its own queue
        self.queue = Queue(maxsize=cache_size)
        self.num_workers = num_workers
        self.workers = []

    def set_workers(self, target_process):

        for _ in range(self.num_workers):
            worker = Process(target=target_process)
            worker.daemon = True
            self.workers.append(worker)

    def start_workers(self):
        # Start worker processes
        for worker in self.workers:
            worker.start()

        print(f"Started Producer with {self.num_workers} worker(s)")

    def stop_workers(self):
        # Stop the worker processes
        for worker in self.workers:
            worker.terminate()
            worker.join()


class ZarrDataset(monai.data.Dataset):
    def __init__(self, opt, paths, patch_shape, transform, num_queues: int = 1, num_workers: int = 1, queue_size: int = 64):
        self.opt = opt
        self.paths = paths
        self.patch_shape = patch_shape
        self.transform = transform
        self.num_queues = num_queues
        self.num_workers = num_workers  # Number of worker processes per queue
        self.queue_size = queue_size

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

        self.nlevels = 3  # Number of levels in the Zarr dataset

        # Estimated RAM usage
        bytes_per_ele = 4  # Assuming float32
        usage = np.prod(self.patch_shape) * num_queues * num_workers * queue_size * bytes_per_ele / 1024 / 1024  # in MB
        print(f"Estimated RAM usage: {usage:.2f} MB")

        # Start worker processes
        self.queues = []
        for i in range(num_queues):
            self._init_queue(queue_size)

        # wait for all queues to fill up
        print("Waiting for queues to fill up...")
        for queue in self.queues:
            while queue.qsize() < queue_size:
                continue

        # Apply deterministic transforms?

    def _init_queue(self, cache_size: int = 64):

        # Initialize multiprocessing queue and cache
        self.queue = Queue(maxsize=cache_size)
        self.queues.append(self.queue)

        # Start worker processes
        self.workers = []
        for _ in range(self.num_workers):
            worker = Process(target=self._worker_process)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        print(f"Started processes with {self.num_workers} worker(s)")


    def _extract_patch(self, data, patch_size=(32, 32, 32)):

        # We start with the first level
        volume = data['volume'][f'{0}']
        start = np.random.randint(0, np.array(volume.shape) - patch_size)  # (0,0,0)
        end = start + patch_size

        patch = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return patch

    def _worker_process(self):

        while True:
            if self.queue.full():
                 # Wait until the queue is not full
                 sleep(0.1)  # sleep for a bit
                 continue

            for z in self.zarr_data:
                patch = self._extract_patch(z, self.patch_shape)
                # simulate processing time
                sleep(0.1)
                #if self.transform:
                #    patch = self.transform(patch)
                self.queue.put(patch)

                #print("Produced patch, queue size: ", self.queue.qsize())

    def stop_workers(self):
        # Stop the worker processes
        for worker in self.workers:
            worker.terminate()
            worker.join()

    def __getitem__(self, index):

        # search for a non-empty queue
        i = 0
        while True:
            for queue in self.queues:
                if not queue.empty():
                    patch = queue.get()
                    return patch
                else:
                    continue
            i += 1
            if i % 500 == 0:
                print("All queues are empty...")
                i = 0

    # def __getitem__(self, index):
    #
    #     if self.queue.empty():
    #         # Wait until the queue is not empty
    #         while self.queue.empty():
    #             continue
    #
    #     # Get the patch from the queue
    #     patch = self.queue.get()
    #     return patch

def main():

    # Example usage
    opt = None
    batch_size = 8
    patch_shape = (64, 64, 64)
    paths = ["../ome_array_pyramid.zarr"] * batch_size
    transform = None  # monai.transforms.Identityd(keys=[], allow_missing_keys=True)  # Define your transformation here

    seed = 8883
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ZarrDataset(opt, paths, patch_shape, transform, num_queues=8, num_workers=1, queue_size=32)

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
        for batch in dataloader:
            print("Loaded batch...")
            # for key in batch.keys():
            #    print(f"Key: {key}, Shape: {batch[key].shape}")
    time_elapsed = time() - start_time
    print(f"Time taken {time_elapsed} sec.")
    print(f"Time taken per patch {time_elapsed / no_epochs / batch_size} sec. (average)")

    dataset.stop_workers()

    print(f"Loaded {len(dataset)} items from Zarr dataset.")


if __name__ == "__main__":
    main()

