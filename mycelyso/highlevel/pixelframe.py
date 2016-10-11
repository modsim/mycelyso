# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy

from skimage.measure import label

from pilyso.processing.pixelgraphs import where2d, get_connectivity_map, get_neighborhood_map, is_junction, is_end, \
    get_next_neighbor, get_inverse_neighbor_shift, is_edge, get_all_neighbor_nums


class PixelFrame(object):
    def __init__(self, image, timepoint=0.0):
        self.timepoint = timepoint  # meta

        # binary image of skeleton, 0 and 1s (forced bool cast)
        self.image = (image > 0).astype(numpy.uint8)

        self.marker = label(image)

        self.graph_lengths = numpy.bincount(self.marker.ravel())

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
                lastn = n

                y, x = y + ys, x + xs

                if not (-1 < y < nm.shape[0] and -1 < x < nm.shape[1]):
                    # there was a strange case where the algorithm wanted to jump outside of the image
                    y, x = y - ys, x - xs
                    break

                points.append([y, x])

                inverse = get_inverse_neighbor_shift(lastn)
                if inverse < nm[y, x]:
                    nm[y, x] -= inverse

                n = nm[y, x]

                if not is_edge(conn[y, x]):
                    break
                if n == 0:
                    break

            points = numpy.array(points)

            if is_junction(conn[y, x]):
                if nm[y, x] > 0:
                    for n in get_all_neighbor_nums(nm[y, x]):
                        todo.insert(0, [n, y, x])

            pathlets.append(points)

        self.pathlets = pathlets

        if False:

            result_buffer = numpy.zeros_like(conn, dtype=numpy.uint16)

            result_buffer[self.image > 0] = 2**16 - 1

            for n, points in enumerate(pathlets):
                for y, x in points:
                    result_buffer[y, x] = n
            return result_buffer
