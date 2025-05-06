import numpy as np


def standardize_rows(data, mean, std):

    mean_broad = mean[None, :]  # we dont have to index with None, but just to show we can
    print(mean_broad, mean_broad.shape)

    std_broad = std[None, :]  # we dont have to index with None, but just to show we can
    print(std_broad, std_broad.shape)

    data = (data - mean_broad) / std_broad

    return data

def outer(x, y):

    print(x, x.shape)
    print(y, y.shape)

    x = x[:, None]  # Increase along column dim to get a column vector: (2,) -> (2, 1)
    print(x.shape)

    outer = y*x  # broadcast x along columns of y (multiplication)

    return outer

def distmat_1d(x, y):

    x = x[:, None]  # Increase along column dim to get a column vector: (2,) -> (2, 1)

    distmat = abs(x - y)  # broadcast x along columns of y (subtraction)

    return distmat

if __name__ == "__main__":

    data = np.array([[1, 2, 3], [4, 5 ,6]])
    mean = np.array([0.5, 1, 3])
    std = np.array([1, 2, 3])

    data_norm = standardize_rows(data, mean, std)
    print(data_norm, "\n")

    x = np.array([1, 2])
    y = np.array([3, 4, 5])
    outer_xy = outer(x, y)
    print(outer_xy, "\n")

    x = np.array([1, 2])
    y = np.array([3, 0.5, 1])
    distmat = distmat_1d(x, y)
    print(distmat, "\n")