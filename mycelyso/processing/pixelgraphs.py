# -*- coding: utf-8 -*-
"""
The pixelgraphs module contains various functions to work with skeleton images, and treating the paths of the
skeleton as graphs, which can be walked along.
"""

import numpy as np

from scipy import ndimage as ndi


_COUNTING_KERNEL = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]], dtype=np.uint8
)

_COUNTING_KERNEL.flags.writeable = False

N, NW, W, SW, S, SE, E, NE = 2, 1, 128, 64, 32, 16, 8, 4

# noinspection PyPep8
_NEIGHBORHOOD_KERNEL = np.array(
    [[NW,  N, NE],
     [ W,  0,  E],
     [SW,  S, SE]], dtype=np.uint8
)

# a constant
_NEIGHBORHOOD_KERNEL.flags.writeable = False

X = 1
Y = 0

# remember: Y, X
# noinspection PyPep8
_NEIGHBORS = {
    #    YY  XX
    N:  [-1,  0],
    NW: [-1, -1],
    W:  [ 0, -1],
    SW: [ 1, -1],
    S:  [ 1,  0],
    SE: [ 1,  1],
    E:  [ 0,  1],
    NE: [-1,  1]
}

_INVERSE_NEIGHBORS = {
    N: S,
    NW: SE,
    W: E,
    SW: NE,
    S: N,
    SE: NW,
    E: W,
    NE: SW,
}


def get_connectivity_map(binary):
    """
    Returns a 'connectivity map', where each value represents the count of neighbors at a position.
    
    :param binary: Binary input image 
    :return: 
    """
    return binary * ndi.correlate(binary.astype(np.uint8), _COUNTING_KERNEL, mode='constant')


def get_neighborhood_map(binary):
    """
    Returns a 'neighborhood map', where each value binary encodes the connections at a point.
    
    :param binary: Binary input image 
    :return: 
    """
    return binary * ndi.correlate(binary.astype(np.uint8), _NEIGHBORHOOD_KERNEL, mode='constant')


def get_next_neighbor(num):
    """
    Returns the coordinates represented by a numeric neighbor bit. 
    
    :param num: Neighbor bit 
    :return: Shift (r, c)
    """
    return _NEIGHBORS[num]


def get_all_neighbor_nums(num):
    """
    Return all set neighbor bits in num.
    
    :param num: Neighborhood representation. 
    :return: Array of values
    """
    return [value for value, shift in _NEIGHBORS.items() if (num & value) == value]


def get_all_neighbors(num):
    """
    Return positions for all set neighbor bits in num
    
    :param num: Neighborhood representation
    :return: Array of shifts
    """
    return [shift for value, shift in _NEIGHBORS.items() if (num & value) == value]


def get_inverse_neighbor_shift(num):
    """
    Get the shift corresponding to the inverse direction represented by num. 
    
    :param num: Neighborhood bit
    :return: Shift (r, c)
    """
    return _INVERSE_NEIGHBORS[num]


def is_edge(connectivity):
    """
    Returns True if connectivity corresponds an edge (is two).
    
    :param connectivity: Scalar or matrix 
    :return: Boolean or matrix of boolean
    """
    return connectivity == 2


def is_junction(connectivity):
    """
    Returns True if connectivity corresponds a junction (is greater than two).
    
    :param connectivity: Scalar or matrix 
    :return: Boolean or matrix of boolean
    """
    return connectivity > 2


def is_end(connectivity):
    """
    Returns True if connectivity corresponds to an endpoint (is one).
    
    :param connectivity: Scalar or matrix 
    :return: Boolean or matrix of boolean 
    """
    return connectivity == 1


def where2d(image):
    """
    numpy.where for 2D matrices.
    
    :param image: Input images
    :return: Coordinate list where image is non-zero
    
    >>> where2d(np.array([[ 0, 0, 0],
    ...                   [ 0, 1, 1],
    ...                   [ 0, 0, 0]]))
    array([[1, 1],
           [1, 2]])
    """

    hits = np.where(image.ravel())[0]

    result = np.c_[hits, hits]

    result[:, 0] //= image.shape[1]
    result[:, 1] %= image.shape[1]

    return result
