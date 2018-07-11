import numpy as np
from itertools import tee
from scipy.ndimage import uniform_filter1d
from scipy.spatial.ckdtree import cKDTree as KDTree


def calculate_length(points, times=1, w=5):
    """
    Calculates the length of a path.
    
    Paths sampled from pixel grids may contain notable measuring error, if euclidean distances are calculated
    naively. This method uses an adapted approach from [Cornelisse1984]_, by repeatedly smoothing
    the coordinates with a moving average filter before calculating the euclidean distance.
    
    .. [Cornelisse1984] Cornelisse and van den Berg (1984) Journal of Microscopy
       `10.1111/j.1365-2818.1984.tb00544.x <https://dx.doi.org/10.1111/j.1365-2818.1984.tb00544.x>`_
    
    :param points: Input points, a numpy array (X, 2) 
    :param times: Times smoothing should be applied
    :param w: window width of the moving average filter
    :return: Length of the input path
    
    >>> calculate_length(np.array([[1.0, 1.0],
    ...                            [5.0, 5.0]]))
    5.656854249492381
    """
    # adapted method from Cornelisse and van den Berg
    if (len(points) - 2) > w:
        for _ in range(times):
            points[:, 0] = uniform_filter1d(points[:, 0], w, mode='nearest')
            points[:, 1] = uniform_filter1d(points[:, 1], w, mode='nearest')

    result = np.sqrt((np.diff(points, axis=0) ** 2.0).sum(axis=1)).sum()

    return result


def clean_by_radius(points, radius=15.0):
    """
    Bins points by radius and returns only one per radius, removing duplicates.
    
    :param points: Input points 
    :param radius: Radius
    :return: Filtered points
    
    >>> clean_by_radius(np.array([[1.0, 1.0],
    ...                           [1.1, 1.1],
    ...                           [9.0, 9.0]]), radius=1.5)
    array([[1., 1.],
           [9., 9.]])
    """
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
