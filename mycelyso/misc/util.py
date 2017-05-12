import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.ckdtree import cKDTree as KDTree


def calculate_length(points, times=1, w=5):
    # adapted Cornelisse and van den Berg method
    if (len(points) - 2) > w:
        for _ in range(times):
            points[:, 0] = uniform_filter1d(points[:, 0], w, mode='nearest')
            points[:, 1] = uniform_filter1d(points[:, 1], w, mode='nearest')

    result = np.sqrt((np.diff(points, axis=0) ** 2.0).sum(axis=1)).sum()

    return result


def clean_by_radius(points, radius=15.0):
    if len(points) == 0:
        return points
    tree = KDTree(points)
    mapping = tree.query_ball_tree(tree, radius)
    unique_indices = np.unique(list(l[0] for l in sorted(mapping)))
    return points[unique_indices]
