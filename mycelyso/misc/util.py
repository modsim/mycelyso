import numpy as np
from itertools import tee
from scipy.ndimage import uniform_filter1d
from scipy.spatial.ckdtree import cKDTree as KDTree


def calculate_length(points, times=1, w=5):
    # adapted method from Cornelisse and van den Berg
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


# from the itertools help https://docs.python.org/2/library/itertools.html
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
# end