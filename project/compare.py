class ZarrProducer():
    def __init__(self, zarr_data, group_name, ome_levels, patch_shape, patch_transform, queue_size: int = 64, num_workers: int = 1, seed=8338):
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

        self.ome_levels = ome_levels  # levels in the OME-Zarr dataset
        self.group_name = group_name  # Name of the group in the Zarr file

        self.seed = seed

    def _worker_process(self, id):

        self.set_random_seed(self.seed + id)  # Set random seed for each worker
        # print("Worker seed set to: ", self.seed + id)

        while not self.stop_event.is_set():
            z = random.choice(self.zarr_data)  # Randomly select a zarr dataset
            patch = self._extract_patch_levels(z, self.patch_shape)
            if self.patch_transform:
                patch = self.patch_transform(patch)
            try:
                self.queue.put(patch)  # block for time out space is available
            except queue.Full:
                sleep(0.2)  # Sleep for a short time if the queue is full
                continue

    def _extract_patch(self, data, patch_size=(32, 32, 32)):

        # We start with the first level
        volume = data[self.group_name][self.ome_levels[0]]
        start = np.random.randint(0, np.array(volume.shape) - patch_size)  # (0,0,0)
        end = start + patch_size

        patch = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        out_dict = {self.ome_levels[0]: patch}
        return out_dict

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

    def set_workers(self):

        for id in range(self.num_workers):
            worker = Process(target=self._worker_process, args=(id,))
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
            worker.join(timeout=2)

    def set_random_seed(self, seed):
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        monai.utils.misc.set_determinism(seed)
        np.random.RandomState(seed)