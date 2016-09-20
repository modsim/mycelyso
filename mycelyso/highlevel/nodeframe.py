# -*- coding: utf-8 -*-

from itertools import chain

import numpy
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial.ckdtree import cKDTree as KDTree

from ..misc.util import calculate_length, clean_by_radius
from pilyso.processing.pixelgraphs import where2d, is_end


class NodeFrame(object):

    pf = None
    timepoint = None

    junction_shift = None
    endpoint_shift = None

    data = None

    endpoint_tree = None
    junction_tree = None

    endpoint_tree_data = None
    junction_tree_data = None

    adjacency = None

    every_endpoint = None
    every_junction = None

    shortest_paths = None
    shortest_paths_num = None

    connected_components_count = None
    connected_components = None

    self_to_successor = None
    successor_to_self = None
    self_to_successor_alternatives = None

    def __init__(self, pf):
        self.pf = pf
        self.timepoint = self.pf.timepoint  # copy this information, so we can set pf to None for serialization
        self.prepare_graph()

    def prepare_graph(self):
        endpoint_tree_data = clean_by_radius(where2d(self.pf.endpoints_map), 8.0)  # TODO
        junction_tree_data = clean_by_radius(where2d(self.pf.junctions_map), 8.0)  # TODO

        e_length = len(endpoint_tree_data)
        j_length = len(junction_tree_data)

        total_length = e_length + j_length

        data = numpy.r_[endpoint_tree_data, junction_tree_data]

        endpoint_tree_data = data[:e_length]
        junction_tree_data = data[e_length:]

        if e_length > 0:
            endpoint_tree = KDTree(endpoint_tree_data)
        else:
            endpoint_tree = None

        if j_length > 0:
            junction_tree = KDTree(junction_tree_data)
        else:
            junction_tree = None

        junction_shift = e_length
        endpoint_shift = 0

        # while ends and junctions need to remain different,
        # they are put in the same graph / adjacency matrix
        # so, first come end nodes, then junction nodes
        # => shifts

        adjacency = lil_matrix((total_length, total_length), dtype=float)

        # little bit of nomenclature:
        # a pathlet (pixel graph so to say) is a path of on the image
        # its begin is the 'left' l_ side, its end is the 'right' r_ side
        # (not using begin / end not to confuse end with endpoint ...)

        for pathlet in self.pf.pathlets:
            pathlet_length = calculate_length(pathlet)

            l_side = pathlet[0]
            r_side = pathlet[-1]

            distance_threshold = 10.0  # TODO

            # experiment
            l_test_distance, l_test_index = endpoint_tree.query(l_side, k=1)
            if l_test_distance < distance_threshold:
                l_is_end = True
            else:
                # original code
                l_is_end = is_end(self.pf.connectivity[l_side[0], l_side[1]])

            # experiment
            r_test_distance, r_test_index = endpoint_tree.query(r_side, k=1)
            if r_test_distance < distance_threshold:
                r_is_end = True
            else:
                # original code
                r_is_end = is_end(self.pf.connectivity[r_side[0], r_side[1]])

            l_index_shift = endpoint_shift if l_is_end else junction_shift
            r_index_shift = endpoint_shift if r_is_end else junction_shift

            l_tree = endpoint_tree if l_is_end else junction_tree
            r_tree = endpoint_tree if r_is_end else junction_tree

            # first tuple value would be distance, but we don't care
            try:
                l_distance, l_index = l_tree.query(l_side, k=1)
                r_distance, r_index = r_tree.query(r_side, k=1)
            except AttributeError:
                continue

            if l_distance > 30 or r_distance > 30:  # TODO
                # probably does not happen
                continue

            adjacency_left_index = l_index+l_index_shift
            adjacency_right_index = r_index+r_index_shift
            adjacency[adjacency_left_index, adjacency_right_index] = pathlet_length
            adjacency[adjacency_right_index, adjacency_left_index] = pathlet_length

        self.junction_shift = junction_shift
        self.endpoint_shift = endpoint_shift

        self.data = data

        self.endpoint_tree = endpoint_tree
        self.junction_tree = junction_tree

        self.endpoint_tree_data = endpoint_tree_data
        self.junction_tree_data = junction_tree_data

        self.adjacency = adjacency

        self.every_endpoint = range(self.endpoint_shift, self.junction_shift)
        self.every_junction = range(self.junction_shift, self.junction_shift + len(self.junction_tree_data))

        cleanup_graph_after_creation = True

        if cleanup_graph_after_creation:
            self.cleanup_adjacency()

        self.adjacency = self.adjacency.tocsr()
        self.generate_derived_data()

    def cleanup_adjacency(self):
        non_empty_mask = (self.adjacency.getnnz(axis=0) + self.adjacency.getnnz(axis=1)) > 0
        empty_indices, = numpy.where(~non_empty_mask)
        # these int casts can go away, once scipy #5026 is in

        # if this ever becomes multithreaded, we should lock the trees now
        # endpoint_tree, junction_tree = None, None

        e_length = non_empty_mask[:self.junction_shift].sum()
        j_length = non_empty_mask[self.junction_shift:].sum()

        total_length = e_length + j_length

        self.junction_shift = e_length
        self.endpoint_shift = 0

        self.data = self.data[non_empty_mask]

        self.endpoint_tree_data = self.data[:e_length]
        self.junction_tree_data = self.data[e_length:]

        if e_length > 0:
            self.endpoint_tree = KDTree(self.endpoint_tree_data)
        else:
            self.endpoint_tree = None

        if j_length > 0:
            self.junction_tree = KDTree(self.junction_tree_data)
        else:
            self.junction_tree = None

        new_adjacency = lil_matrix((total_length, total_length), dtype=self.adjacency.dtype)

        coo = self.adjacency.tocoo()

        for n, m, value in zip(coo.row, coo.col, coo.data):
            npos, = numpy.where(n >= empty_indices)
            mpos, = numpy.where(m >= empty_indices)
            npos = 0 if len(npos) == 0 else npos[-1] + 1
            mpos = 0 if len(mpos) == 0 else mpos[-1] + 1
            new_adjacency[n-npos, m-mpos] = value

        self.adjacency = new_adjacency

        self.every_endpoint = range(self.endpoint_shift, self.junction_shift)
        self.every_junction = range(self.junction_shift, self.junction_shift + len(self.junction_tree_data))

    def generate_derived_data(self):
        self.shortest_paths = shortest_path(self.adjacency)
        self.shortest_paths_num = shortest_path(self.adjacency, unweighted=True)

        self.connected_components_count, self.connected_components = connected_components(self.adjacency)

    def is_endpoint(self, i):
        return i in self.every_endpoint

    def is_junction(self, i):
        return i in self.every_junction

    def get_connected_nodes(self, some_node):
        label = self.connected_components[some_node]
        return numpy.where(self.connected_components[self.connected_components == label])[0]

    def track(self, successor):
        junction_shift_radius = 50.0  # TODO
        endpoint_shift_radius = 150.0  # TODO

        ##
        self_len = len(self.data)

        successor_len = len(successor.data)

        self_to_successor = numpy.zeros(self_len, dtype=int)
        successor_to_self = numpy.zeros(successor_len, dtype=int)

        self_to_successor[:] = -1
        successor_to_self[:] = -1

        self_to_successor_alternatives = [[]] * self_len

        if self.junction_tree is not None and successor.junction_tree is not None:
            junction_mapping = self.junction_tree.query_ball_tree(successor.junction_tree, junction_shift_radius)
        else:
            junction_mapping = []

        if self.endpoint_tree is not None and successor.endpoint_tree is not None:
            endpoint_mapping = self.endpoint_tree.query_ball_tree(successor.endpoint_tree, endpoint_shift_radius)
        else:
            endpoint_mapping = []

        # print(self.timepoint, get_or_else(lambda: self.endpoint_tree.data), endpoint_mapping)

        for self_hit, n in enumerate(chain(endpoint_mapping, junction_mapping)):
            if len(n) == 0:
                n = -1
                ordered_n = []
            else:
                search_point = self.data[self_hit]

                hit_points = numpy.array([successor.data[h] for h in n])

                distances = numpy.sqrt(((hit_points - search_point) ** 2).sum(axis=1))

                indexed = numpy.c_[distances, n]
                ordered_n = [int(nn[1]) for nn in sorted(indexed, key=lambda t: t[0])]

                min_distance = numpy.argmin(distances)

                n = n[min_distance]
                if self_hit > self.junction_shift:
                    n += successor.junction_shift

            self_to_successor_alternatives[self_hit] = ordered_n

            self_to_successor[self_hit] = n
            successor_to_self[n] = self_hit

        self.self_to_successor = self_to_successor  # this is mainly used
        self.successor_to_self = successor_to_self
        self.self_to_successor_alternatives = self_to_successor_alternatives

    def get_networkx_graph(self, with_z=0, return_positions=False):
        from networkx import from_scipy_sparse_matrix

        g = from_scipy_sparse_matrix(self.adjacency)

        positions = {}

        for n, pos in enumerate(self.data):
            positions[n] = (float(pos[1]), float(pos[0]))
            g.node[n]['x'] = float(pos[1])
            g.node[n]['y'] = float(pos[0])

            if with_z > 0:
                g.node[n]['z'] = float(self.timepoint * with_z)
                positions[n] = (float(pos[1]), float(pos[0]), float(self.timepoint * with_z))

        if not return_positions:
            return g
        else:
            return g, positions
