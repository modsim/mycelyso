# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy
import cv2
from .processing import label_image


def _scipy_get_connectivity_map(binary):
    counting_kernel = numpy.ones((3, 3,), dtype=numpy.uint8)
    counting_kernel[1, 1] = 0
    return binary * ndimage.correlate(binary.astype(numpy.uint8), counting_kernel, mode='constant')


def _cv2_get_connectivity_map(binary):
    counting_kernel = numpy.ones((3, 3,), dtype=numpy.uint8)
    counting_kernel[1, 1] = 0
    return binary * cv2.filter2D(binary.astype(numpy.uint8), -1, counting_kernel)

get_connectivity_map = _cv2_get_connectivity_map

#neighborhood_kernel = numpy.array(
#    [[  1,   2,   4],
#     [128,   0,   8],
#     [ 64,  32,  16]], dtype=numpy.uint8
#)

#N, NW, W, SW, S, SE, E, NE = 0, 1, 2, 3, 4, 5, 6, 7

N, NW, W, SW, S, SE, E, NE = 2, 1, 128, 64, 32, 16, 8, 4

# noinspection PyPep8
NEIGHBORHOOD_KERNEL = numpy.array(
    [[NW,  N, NE],
     [ W,  0,  E],
     [SW,  S, SE]], dtype=numpy.uint8
)

# a constant
NEIGHBORHOOD_KERNEL.flags.writeable = False

X = 1
Y = 0

# remember: Y, X
NEIGHBORS = {
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

NEIGHBOR_SHIFTS = list(NEIGHBORS.items())

INVERSE_NEIGHBORS = {
    N: S,
    NW: SE,
    W: E,
    SW: NE,
    S: N,
    SE: NW,
    E: W,
    NE: SW,
}


def _cv2_get_neighborhood_map(binary):
    return binary * cv2.filter2D(binary.astype(numpy.uint8), -1, NEIGHBORHOOD_KERNEL)

get_neighborhood_map = _cv2_get_neighborhood_map


def get_next_neighbor(num):
    return NEIGHBORS[num]


def get_all_neighbor_nums(num):
    return [value for value, shift in NEIGHBORS.items() if (num & value) == value]


def get_all_neighbors(num):
    return [shift for value, shift in NEIGHBORS.items() if (num & value) == value]


def get_inverse_neighbor_shift(num):
    return INVERSE_NEIGHBORS[num]


def where2d(image):

    hits = numpy.where(image.ravel())[0]

    result = numpy.c_[hits, hits]

    result[:, 0] //= image.shape[1]
    result[:, 1] %= image.shape[1]

    return result

def get_corners(binary, connectivity=None):

    if connectivity is None:
        connectivity = get_connectivity_map(binary)

    return where2d(connectivity == 1)


def is_edge(connectivity):
    return connectivity == 2


def is_junction(connectivity):
    return connectivity > 2


def is_end(connectivity):
    return connectivity == 1


def filter_connected_components(image, minimum_length=10):
    marker = label_image(image)

    lengths = numpy.bincount(marker.ravel())

    pairs = numpy.c_[numpy.linspace(0, lengths.shape[0] - 1, lengths.shape[0]).astype(int), lengths]

    good = pairs[pairs[:, 1] > minimum_length][:, 0]

    if good[0] == 0:
        good = good[1:]

    return numpy.in1d(marker.ravel(), good).reshape(image.shape)

