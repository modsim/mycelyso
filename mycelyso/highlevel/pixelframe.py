# -*- coding: utf-8 -*-
"""
The pixelframe module contains the PixelFrame class, a representation of one time lapse frame
at the binary mask/pixel level.
"""

import numpy as np

from skimage.measure import label

from ..processing.pixelgraphs import \
    where2d, get_connectivity_map, get_neighborhood_map, is_edge, is_end, is_junction, \
    get_next_neighbor, get_inverse_neighbor_shift, get_all_neighbor_nums


class PixelFrame(object):
    """
    A PixelFrame represents the pixel graph of one (skeletonized) image stack frame.
    """
    def __init__(self, image, timepoint=0.0, calibration=1.0):
        """
        Initializes the PixelFrame
        
        :param image: Input image (skeletonized) 
        :param timepoint: Experiment time  (in seconds) 
        :param calibration: Pixel size (in µm)
        """
        self.timepoint = timepoint
        self.calibration = calibration

        # binary image of skeleton, 0 and 1s (forced bool cast)
        self.image = (image > 0).astype(np.uint8)

        self.marker = label(image)

        self.graph_lengths = np.bincount(self.marker.ravel())

        self.graph_count = self.graph_lengths.shape[0]

        self.connectivity = get_connectivity_map(image)

        self.neighborhood_map = get_neighborhood_map(image)

        self.junctions_map = is_junction(self.connectivity)
        self.endpoints_map = is_end(self.connectivity)

        self.junctions = where2d(self.junctions_map)
        self.endpoints = where2d(self.endpoints_map)

        self.pathlets = []

        self.create_graph()

    def create_graph(self):
        """
        Creates the graph from the pixel skeleton.
        
        :return: 
        """
        conn = self.connectivity
        nm = self.neighborhood_map.copy()

        todo = [[nm[y, x], y, x] for y, x in self.endpoints]

        pathlets = []

        while len(todo) > 0:
            n, y, x = todo.pop()

            if nm[y, x] < n:
                continue

            points = [[y, x]]

            while True:
                ys, xs = get_next_neighbor(n)
                nm[y, x] -= n
                last_n = n

                y, x = y + ys, x + xs

                if not (-1 < y < nm.shape[0] and -1 < x < nm.shape[1]):
                    # there was a strange case where the algorithm wanted to jump outside of the image
                    y, x = y - ys, x - xs
                    break

                points.append([y, x])

                inverse = get_inverse_neighbor_shift(last_n)
                if inverse < nm[y, x]:
                    nm[y, x] -= inverse

                n = nm[y, x]

                if not is_edge(conn[y, x]):
                    break
                if n == 0:
                    break

            points = np.array(points)

            if is_junction(conn[y, x]):
                if nm[y, x] > 0:
                    for n in get_all_neighbor_nums(nm[y, x]):
                        todo.insert(0, [n, y, x])

            pathlets.append(points)

        self.pathlets = pathlets
