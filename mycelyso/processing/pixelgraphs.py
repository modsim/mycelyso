# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy

from scipy import ndimage as ndi


_COUNTING_KERNEL = numpy.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]], dtype=numpy.uint8
)

_COUNTING_KERNEL.flags.writeable = False

N, NW, W, SW, S, SE, E, NE = 2, 1, 128, 64, 32, 16, 8, 4

# noinspection PyPep8
_NEIGHBORHOOD_KERNEL = numpy.array(
    [[NW,  N, NE],
     [ W,  0,  E],
     [SW,  S, SE]], dtype=numpy.uint8
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
    return binary * ndi.correlate(binary.astype(numpy.uint8), _COUNTING_KERNEL, mode='constant')


def get_neighborhood_map(binary):
    return binary * ndi.correlate(binary.astype(numpy.uint8), _NEIGHBORHOOD_KERNEL, mode='constant')


def get_next_neighbor(num):
    return _NEIGHBORS[num]


def get_all_neighbor_nums(num):
    return [value for value, shift in _NEIGHBORS.items() if (num & value) == value]


def get_all_neighbors(num):
    return [shift for value, shift in _NEIGHBORS.items() if (num & value) == value]


def get_inverse_neighbor_shift(num):
    return _INVERSE_NEIGHBORS[num]


def is_edge(connectivity):
    return connectivity == 2


def is_junction(connectivity):
    return connectivity > 2


def is_end(connectivity):
    return connectivity == 1


def where2d(image):

    hits = numpy.where(image.ravel())[0]

    result = numpy.c_[hits, hits]

    result[:, 0] //= image.shape[1]
    result[:, 1] %= image.shape[1]

    return result
