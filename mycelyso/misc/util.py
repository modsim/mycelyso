import numpy
from scipy.spatial.ckdtree import cKDTree as KDTree


def smooth(signal, kernel):
    """
    Generic smoothing function, smooths by convolving one signal with another.

    :param signal: input signal to be smoothed
    :type signal: numpy.ndarray
    :param kernel: smoothing kernel to be used. will be normalized to :math:`\sum=1`
    :type kernel: numpy.ndarray
    :return: The signal convolved with the kernel
    :rtype: numpy.ndarray

    >>> smooth(numpy.array([0, 0, 0, 0, 1, 0, 0, 0, 0]), numpy.ones(3))
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.33333333,
            0.33333333,  0.33333333,  0.        ,  0.        ])
    """

    return numpy.convolve(
        kernel / kernel.sum(),
        numpy.r_[signal[kernel.size - 1:0:-1], signal, signal[-1:-kernel.size:-1]],
        mode='valid')[kernel.size // 2 - 1:-kernel.size // 2][0:len(signal)]


def calculate_length(g):
    def get_length(points, times=1, w=5):
        # Cornelisse
        if (len(points) - 2) > w:
            kernel = numpy.ones(w)
            for _ in range(times):
                # keep first and last point static, otherwise the line may drift
                points[1:-1, 0] = smooth(points[1:-1, 0], kernel)
                points[1:-1, 1] = smooth(points[1:-1, 1], kernel)

        result = numpy.sqrt((numpy.diff(points, axis=0) ** 2.0).sum(axis=1)).sum()

        return result
    return get_length(g)


def clean_by_radius(points, radius=15.0):
    if len(points) == 0:
        return points
    tree = KDTree(points)
    mapping = tree.query_ball_tree(tree, radius)
    unique_indices = numpy.unique(list(l[0] for l in sorted(mapping)))
    return points[unique_indices]
