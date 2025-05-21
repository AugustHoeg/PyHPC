import os
import sys
import random
import numpy as np
import torch
import monai
import zarr
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader
from time import sleep
from time import perf_counter as time
from multiprocessing import Process, Queue, Event

class ZarrProducer(Process):
    def __init__(self, zarr_data, patch_shape, patch_transform, queue_size: int = 64, num_workers: int = 1):
        super().__init__()

        # Define data
        self.zarr_data = zarr_data
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform

        # Each worker will have its own queue
        self.queue = Queue(maxsize=queue_size)
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = Event()  # Event to signal workers to stop


    def _worker_process(self):

        while not self.stop_event.is_set():
            z = random.choice(self.zarr_data)  # Randomly select a zarr dataset
            patch = self._extract_patch(z, self.patch_shape)
            if self.patch_transform:
                patch = self.patch_transform(patch)
            self.queue.put(patch, block=True)  # block until space is available


    def _extract_patch(self, data, patch_size=(32, 32, 32)):

        # We start with the first level
        volume = data['volume'][f'{0}']
        start = np.random.randint(0, np.array(volume.shape) - patch_size)  # (0,0,0)
        end = start + patch_size

        patch = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return patch


    def set_workers(self):

        for _ in range(self.num_workers):
            worker = Process(target=self._worker_process)
            worker.daemon = True
            self.workers.append(worker)


    def start_workers(self):
        # Start worker processes
        for worker in self.workers:
            worker.start()

        print(f"Started Producer with {self.num_workers} worker(s)")


    def stop_workers(self):
        # Stop the worker processes by setting stop event
        self.stop_event.set()
        for worker in self.workers:
            worker.join()


class ZarrDataset(monai.data.Dataset):
    def __init__(self, opt, paths, patch_shape, transform, num_producers: int = 1, num_workers: int = 1, queue_size: int = 64):
        self.opt = opt
        self.paths = paths
        self.patch_shape = patch_shape
        self.transform = transform
        self.num_producers = num_producers
        self.num_workers = num_workers  # Number of worker processes per queue
        self.queue_size = queue_size

        super().__init__(paths, transform)

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

        self.nlevels = 3  # Number of levels in the Zarr dataset

        # Estimated RAM usage
        bytes_per_ele = 4  # Assuming float32
        usage = np.prod(self.patch_shape) * num_producers * num_workers * queue_size * bytes_per_ele / 1024 / 1024  # in MB
        print(f"Estimated RAM usage: {usage:.2f} MB")

        # Start producer processes
        self._init_producers()


    def _init_producers(self):
        # Start worker processes
        self.producers = []
        for i in range(self.num_producers):
            producer = ZarrProducer(self.zarr_data,
                                    self.patch_shape,
                                    self.transform,
                                    queue_size=self.queue_size,
                                    num_workers=self.num_workers)
            self.producers.append(producer)

            producer.set_workers()
            producer.start_workers()

        # wait for all queues to fill up
        print("Waiting for producer queues to fill up...")
        for producer in self.producers:
            while producer.queue.qsize() < self.queue_size:
                continue


    def stop_producers(self):
        # Stop the producer processes
        for producer in self.producers:
            producer.stop_workers()


    def __getitem__(self, index):

        while True:
            for producer in self.producers:
                if producer.queue.empty():
                    continue
                else:
                    patch = producer.queue.get(timeout=0.1)
                    return patch

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

    dataset = ZarrDataset(opt, paths, patch_shape, transform, num_producers=8, num_workers=1, queue_size=64)

    num_workers = 0
    persistent_workers = True if num_workers > 0 else False
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            persistent_workers=persistent_workers)

    no_epochs = 100

    start_time = time()
    for i in range(no_epochs):
        for batch in dataloader:
            print("Loaded batch...")
            # for key in batch.keys():
            #    print(f"Key: {key}, Shape: {batch[key].shape}")
    time_elapsed = time() - start_time
    print(f"Time taken {time_elapsed} sec.")
    print(f"Time taken per patch {time_elapsed / no_epochs / batch_size} sec. (average)")

    dataset.stop_producers()

    print(f"Loaded {len(dataset)} items from Zarr dataset.")


if __name__ == "__main__":
    main()

