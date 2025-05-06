import os
import torch
import monai
import zarr


class ZarrDataset(monai.data.Dataset):
    def __init__(self, opt, paths, transform, num_workers: int = 0):
        self.opt = opt
        self.paths = paths
        self.transform = transform
        self.num_workers = num_workers

        super().__init__(paths, transform)

        # Check if the paths are valid
        if not all(os.path.exists(path) for path in paths):
            raise ValueError("One or more paths do not exist.")

        self.zarr_data = [zarr.open(path, mode='r') for path in paths]

    def __getitem__(self, index):
        # Load the data from the Zarr file
        data = self.zarr_data[index]

        # Apply the transformation
        if self.transform:
            data = self.transform(data)

        return data


if __name__ == "__main__":


    paths = [{"H": img_HR, "L": img_LR} for img_HR, img_LR, in zip(HR_train, self.LR_train)]

    # Example usage
    opt = None
    paths = ["sample1.zarr", "sample2.zarr"]
    transform = monai.transforms.Identityd(keys=[])  # Define your transformation here

    dataset = ZarrDataset(opt, paths, transform)
    print(f"Loaded {len(dataset)} items from Zarr dataset.")
