import sys
import numpy as np

#@profile
def distance_matrix_v2(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)

    D = np.empty((len(p1), len(p2)))

    for i in range(len(p1)):  # Only single loop, vectorized along coordinate pairs of p1,p2
        dsin2 = np.sin(0.5 * (p1[i] - p2[:])) ** 2
        cosprod = np.cos(p1[i, 0]) * np.cos(p2[:, 0])
        a = dsin2[:, 0] + cosprod * dsin2[:, 1]
        D[i, :] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    D *= 6371  # Earth radius in km
    return D

#@profile
def distance_matrix_v3(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)

    D = np.empty((len(p1), len(p2)))

    cosprod = np.cos(p1[:, 0])[:, None] * np.cos(p2[:, 0])

    for i in range(len(p1)):  # Only single loop, cosprod outside and arcsin instead of atan2
        dsin2 = np.sin(0.5 * (p1[i] - p2[:])) ** 2
        a = dsin2[:, 0] + cosprod[i] * dsin2[:, 1]
        D[i, :] = 2 * np.arcsin(np.sqrt(a))

    D *= 6371  # Earth radius in km
    return D

#@profile
def distance_matrix_v4(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)
    cosprod = np.cos(p1[:, 0])[:, None] * np.cos(p2[:, 0])
    dsin2 = np.sin(0.5 * (p1[:, None] - p2[:])) ** 2
    a = dsin2[..., 0] + cosprod * dsin2[..., 1]
    D = 2 * np.arcsin(np.sqrt(a))
    D *= 6371
    return D

def distance_matrix(p1, p2):
    p1, p2 = np.radians(p1), np.radians(p2)

    D = np.empty((len(p1), len(p2)))
    for i in range(len(p1)):
        for j in range(len(p2)):
            dsin2 = np.sin(0.5 * (p1[i] - p2[j])) ** 2
            cosprod = np.cos(p1[i, 0]) * np.cos(p2[j, 0])
            a = dsin2[0] + cosprod * dsin2[1]
            D[i, j] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    D *= 6371  # Earth radius in km
    return D


def load_points(fname):
    data = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(1, 2))
    return data


def distance_stats(D):
    # Extract upper triangular part to avoid duplicate entries
    assert D.shape[0] == D.shape[1], 'D must be square'
    idx = np.triu_indices(D.shape[0], k=1)
    distances = D[idx]
    return {
        'mean': float(distances.mean()),
        'std': float(distances.std()),
        'max': float(distances.max()),
        'min': float(distances.min()),
    }

fname = sys.argv[1]
points = load_points(fname)
#D = distance_matrix(points, points)
#D = distance_matrix_v2(points, points)
#D = distance_matrix_v3(points, points)
D = distance_matrix_v4(points, points)
stats = distance_stats(D)
print(stats)
