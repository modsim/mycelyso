# -*- coding: utf-8 -*-
"""
documentation
"""


import warnings
from itertools import product, chain

import numpy as np
from scipy.stats import linregress

from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize as sk_skeletonize
from skimage.measure import label, regionprops

from scipy import ndimage as ndi

import networkx as nx

from tunable import Tunable
from ..misc.graphml import to_graphml_string
from ..misc.util import pairwise
from ..misc.regression import prepare_optimized_regression

from ..processing.binarization import experimental_thresholding

from .pixelframe import PixelFrame
from .nodeframe import NodeFrame

try:
    import matplotlib.pyplot as pyplot
except ImportError:
    pyplot = None


def qimshow(image, cmap='gray'):
    """
    debug function
    :param image: 
    :param cmap: 
    :return: 
    """
    if not pyplot:
        raise RuntimeError('matplotlib not installed.')
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    range_ = ax.imshow(image, cmap=cmap, interpolation='none')

    def _format_coords(x, y):
        try:
            y, x = int(y + 0.5), int(x + 0.5)
            if y < 0 or x < 0:
                raise IndexError
            value = image[y, x]
        except IndexError:
            value = float('nan')
        return 'x=%d y=%d value=%1.4f' % (x, y, value,)

    ax.format_coord = _format_coords
    matplotlib.pyplot.colorbar(range_)
    matplotlib.pyplot.show()


def set_empty_crops(image, crop_t=None, crop_b=None, crop_l=None, crop_r=None):
    return 0, image.shape[0], 0, image.shape[1]


def skip_if_image_is_below_size(min_height=4, min_width=4):
    def _inner(image, meta):
        if image.shape[0] < min_height or image.shape[1] < min_width:
            raise Skip(Meta(pos=meta.pos, t=Collected)) from None

        return image, meta

    return _inner


def binarize(image, binary=None):
    return experimental_thresholding(image)


def skeletonize(binary, skeleton=None):
    return sk_skeletonize(binary)


def image_statistics(image, calibration, result=None):
    return {
        'input_width': image.shape[1] * calibration,
        'input_height': image.shape[0] * calibration,
        'input_width_pixel': image.shape[1],
        'input_height_pixel': image.shape[0],
        'area': image.shape[1] * calibration * image.shape[0] * calibration,
        'area_pixel': image.shape[1] * image.shape[0]
    }


def quantify_binary(binary, calibration, result=None):
    ones = np.sum(binary)
    total = binary.shape[0] * binary.shape[1]
    return {
        'covered_ratio': ones / total,
        'covered_area': ones * calibration **2,
        'covered_area_pixel': ones
    }


def graph_statistics(node_frame, result=None):
    # everything / 2 because it's a digraph/graph structure
    graph_edge_length = node_frame.adjacency.sum() / 2
    graph_edge_count = node_frame.adjacency.nnz / 2

    graph_node_count = node_frame.adjacency.shape[0]
    graph_junction_count = len(node_frame.every_junction)
    graph_endpoint_count = len(node_frame.every_endpoint)

    return {
        'graph_edge_length': graph_edge_length * node_frame.calibration,
        'graph_edge_count': graph_edge_count,
        'graph_node_count': graph_node_count,
        'graph_junction_count': graph_junction_count,
        'graph_endpoint_count': graph_endpoint_count
    }


class CleanUpGaussianSigma(Tunable):
    """Clean up step: Sigma [µm] used for Gaussian filter"""
    default = 0.075


class CleanUpGaussianThreshold(Tunable):
    """Clean up step: Threshold used after Gaussian filter (values range from 0 to 1)"""
    default = 0.5


class CleanUpHoleFillSize(Tunable):
    """Clean up step: Maximum size of holes [µm²] which will be filled"""
    default = 20


def clean_up(calibration, binary):
    binary = ndi.gaussian_filter(binary * 1.0, CleanUpGaussianSigma.value / calibration) > CleanUpGaussianThreshold.value

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        binary = remove_small_holes(binary, min_size=int(CleanUpHoleFillSize.value / calibration**2), connectivity=2)

    return binary


class RemoveSmallStructuresSize(Tunable):
    """Remove structures up to this size [µm²]"""
    default = 30


def remove_small_structures(calibration, binary):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return remove_small_objects(binary, min_size=int(RemoveSmallStructuresSize.value / calibration**2), connectivity=2)  # TODO


class BorderArtifactRemovalBorderSize(Tunable):
    """Remove structures, whose centroid lies within that distance [µm] of a border"""
    default = 10


def remove_border_artifacts(calibration, binary):

    border = BorderArtifactRemovalBorderSize.value / calibration

    labeled = label(binary)
    corner_pixels = np.r_[
        labeled[0, :].ravel(),
        labeled[-1, :].ravel(),
        labeled[:, 0].ravel(),
        labeled[:, -1].ravel()
    ]

    corner_pixels = set(np.unique(corner_pixels)) - {0}

    for region in regionprops(labeled):
        if region.label in corner_pixels:
            if ((region.centroid[0] < border) or (region.centroid[0] > (binary.shape[0] - border)) or
                    (region.centroid[1] < border) or (region.centroid[1] > (binary.shape[1] - border))):
                binary[labeled == region.label] = False
    return binary


def convert_to_nodes(skeleton, timepoint, calibration, pixel_frame=None, node_frame=None):
    pixel_frame = PixelFrame(skeleton, timepoint, calibration=calibration)
    node_frame = NodeFrame(pixel_frame)
    return pixel_frame, node_frame


def track_multipoint(collected):
    for result1, result2 in pairwise(collected.values()):
        result1.node_frame.track(result2.node_frame)
    return collected


class TrackingMaximumRelativeShrinkage(Tunable):
    """"""
    default = 0.2


class TrackingMinimumTipElongationRate(Tunable):
    """"""
    default = -50.0


class TrackingMaximumTipElongationRate(Tunable):
    """"""
    default = 100.0


def individual_tracking(collected, tracked_fragments=None, tracked_fragments_fates=None):
    def any_in(needles, haystack):
        for needle in needles:
            if needle in haystack:
                return True
        return False

    tracks = {}

    last_valid = []
    fates = {}

    next_track_id = 0

    for i, (current_result, next_result) in enumerate(pairwise(collected.values())):

        frame, next_frame = current_result.node_frame, next_result.node_frame

        calibration = frame.calibration

        timepoint, next_timepoint = frame.timepoint, next_frame.timepoint
        delta_t_in_hours = (next_timepoint - timepoint) / (60.0*60.0)

        # either: look for pairs of endpoints and junctions,
        # or endpoints and endpoints, if there are no junctions yet
        def valid_pairs():
            for _e in frame.every_endpoint:
                edges = frame.adjacency[_e, :]
                edge_indices = edges.nonzero()[1]

                for _other in edge_indices:
                    no_junctions = not any_in(frame.get_connected_nodes(_other), frame.every_junction)
                    if frame.is_junction(_other) or (no_junctions and frame.is_endpoint(_other)):
                        if _e != _other:
                            yield None, _e, _other

        valid = []
        endpoints_used = set()

        for track_id, e, other in chain(last_valid, valid_pairs()):
            e_on_next, other_on_next = frame.self_to_successor[e], frame.self_to_successor[other]

            if e_on_next == other_on_next:
                continue

            if e_on_next == -1 or other_on_next == -1:
                # tracking error
                fates[track_id] = "track aborted due to missing future node"
                continue

            distance = frame.shortest_paths[e, other]

            def distance_condition(current, new):
                if new == 0.0 or current == 0.0:
                    # print(new, current)
                    return False
                return (new > current) or (abs(1.0 - (current / new)) < TrackingMaximumRelativeShrinkage.value)

            if next_frame.connected_components[e_on_next] != next_frame.connected_components[other_on_next]:
                #  careful now: either the track broke, or we need to pick an alternative, fitting variant

                for _e_on_next, _other_on_next in product(frame.self_to_successor_alternatives[e],
                                                          frame.self_to_successor_alternatives[other]):
                    if (_e_on_next != _other_on_next and (
                        next_frame.connected_components[_e_on_next] == next_frame.connected_components[
                        _other_on_next]) and (
                        next_frame.shortest_paths[_e_on_next, _other_on_next] < float('inf')) and distance_condition(
                            distance, next_frame.shortest_paths[_e_on_next, _other_on_next])):
                        e_on_next, other_on_next = _e_on_next, _other_on_next
                        break

            next_distance = next_frame.shortest_paths[e_on_next, other_on_next]

            distance_num = frame.shortest_paths_num[e, other]

            # print(track_id, i, e, other, e_on_next, other_on_next, distance, distance_num)

            if e in endpoints_used or other in endpoints_used:
                fates[track_id] = "endpoints used otherwise"
                continue

            if distance == float('inf') or next_distance == float('inf'):
                # the first one should never happen,
                # the nodes are no longer connected on the next frame?
                fates[track_id] = "formerly connected components became unconnected? (dist/next dist %.4f %.4f)" % (
                    distance, next_distance
                )
                continue

            if ((next_distance < distance) and
                        abs(1.0 - (distance / next_distance)) > TrackingMaximumRelativeShrinkage.value):
                # a later distance was SHORTER than the current, that means tracking error or cycle in graph
                fates[track_id] = "track aborted due to shortcut (cycle or tracking error) [last %f > next %f]" % (
                    distance, next_distance
                )
                continue

            mu_per_h = ((next_distance - distance) * calibration) / delta_t_in_hours

            if not (TrackingMinimumTipElongationRate.value < mu_per_h < TrackingMaximumTipElongationRate.value):
                # a later distance changed too much, which might be a tracking error
                fates[track_id] = "track aborted due to too large change in length " \
                                  "[last %f, next %f, change %f mu per h]" % (
                    distance, next_distance, mu_per_h
                )
                continue

            # split up tracks which contain just endpoints ... otherwise the elongation rate is twice as high
            if frame.is_endpoint(e) and frame.is_endpoint(other):
                path = frame.get_path(e, other)
                if next_frame.is_endpoint(e_on_next) and next_frame.is_endpoint(other_on_next):
                    if len(path) > 2:
                        other = path[-2]
                        distance = frame.shortest_paths[e, other]
                        distance_num = frame.shortest_paths_num[e, other]
                        continue

            if track_id is None:
                track_id = next_track_id
                next_track_id += 1

            if track_id not in tracks:
                tracks[track_id] = {}

            tracks[track_id][i] = (e, other, e_on_next, other_on_next, distance, distance_num)

            valid.append((track_id, e_on_next, other_on_next))

            endpoints_used.add(e)

        last_valid = valid
    return tracks, fates


class TrackingMinimumTrackedPointCount(Tunable):
    """"""
    default = 5


class TrackingMinimalMaximumLength(Tunable):
    """µm"""
    default = 10.0


class TrackingMinimalGrownLength(Tunable):
    """µm"""
    default = 5.0


# noinspection PyProtectedMember
def prepare_tracked_fragments(collected, tracked_fragments, tracked_fragments_fates, track_table=None,
                              track_table_aux_tables=None):
    key_list = list(collected.keys())
    calibration = next(iter(collected.values()))['calibration']

    track_table = []

    track_table_aux_tables = []

    for track_id, track in tracked_fragments.items():
        if len(track) < TrackingMinimumTrackedPointCount.value:
            continue

        track_list = [[i, track[i]] for i in sorted(track.keys())]

        times_lengths = np.array(
            [[collected[key_list[i]]['node_frame'].timepoint, calibration * distance]
             for i, (e, other, e_on_next, other_on_next, distance, distance_num)
             in track_list])

        # minimum length of the maximally tracked segment
        if times_lengths[:, 1].max() < TrackingMinimalMaximumLength.value:
            continue

        # minimum length grown additionally
        if (times_lengths[:, 1].max() - times_lengths[:, 1].min()) < TrackingMinimalGrownLength.value:
            continue

        # whatever the settings are, there must be growth
        if (times_lengths[:, 1].max() - times_lengths[:, 1].min()) == 0.0:
            continue

        aux_table_num = len(track_table_aux_tables)

        track_table_aux_tables.append(
            [{
                 'track_table_number': aux_table_num,
                 'timepoint': collected[key_list[i]]['node_frame'].timepoint,
                 'node_id_a': e,
                 'node_id_b': other,
                 'node_next_id_a': e_on_next,
                 'node_next_id_b': other_on_next,
                 'distance': calibration * distance,
                 'distance_num': distance_num
             } for i, (e, other, e_on_next, other_on_next, distance, distance_num) in track_list]
        )

        distance_num_helper = np.array(
            [distance_num for i, (e, other, e_on_next, other_on_next, distance, distance_num) in track_list])

        times = times_lengths[:, 0]
        lengths = times_lengths[:, 1]
        relative_lengths = times_lengths[:, 1] / times_lengths[:, 1].min()

        t_min, t_max = times.min(), times.max()

        row = {
            'timepoint_begin': t_min,
            'timepoint_end': t_max,
            'timepoint_center': t_min + (t_max - t_min) / 2,
            'minimum_distance': lengths.min(),
            'maximum_distance': lengths.max(),
            'minimum_distance_num': distance_num_helper.min(),
            'maximum_distance_num': distance_num_helper.max(),
            'duration': t_max - t_min,
            'count': len(times),
            'aux_table': aux_table_num
        }

        try:

            regression = linregress(times, lengths)._asdict()
            row.update({'plain_regression_' + k: v for k, v in regression.items()})
            regression = linregress(times, relative_lengths)._asdict()
            row.update({'normalized_regression_' + k: v for k, v in regression.items()})

            regression = linregress(times, np.log(lengths))._asdict()
            row.update({'logarithmic_plain_regression_' + k: v for k, v in regression.items()})
            regression = linregress(times, np.log(relative_lengths))._asdict()
            row.update({'logarithmic_normalized_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, lengths)
            row.update({'optimized_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, relative_lengths)
            row.update({'optimized_normalized_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, np.log(lengths))
            row.update({'optimized_logarithmic_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, np.log(relative_lengths))
            row.update({'optimized_logarithmic_normalized_regression_' + k: v for k, v in regression.items()})
        except IndexError:
            pass

        track_table.append(row)

    return track_table, track_table_aux_tables


def prepare_position_regressions(collected, result):
    fields = ['covered_ratio', 'covered_area', 'graph_edge_length', 'graph_edge_count', 'graph_node_count',
              'graph_junction_count', 'graph_endpoint_count']

    row = {}

    # noinspection PyProtectedMember
    def prepare_for_field(field_name):
        data = np.array([[f.timepoint, f[field_name]] for f in collected.values()])

        regression = linregress(data[:, 0], data[:, 1])._asdict()
        row.update({field_name + '_linear_regression_' + k: v for k, v in regression.items()})

        regression = linregress(data[:, 0], np.log(data[:, 1]))._asdict()
        row.update({field_name + '_logarithmic_regression_' + k: v for k, v in regression.items()})

        regression = prepare_optimized_regression(data[:, 0], data[:, 1])
        row.update({field_name + '_optimized_linear_regression_' + k: v for k, v in regression.items()})

        regression = prepare_optimized_regression(data[:, 0], np.log(data[:, 1]))
        row.update({field_name + '_optimized_logarithmic_regression_' + k: v for k, v in regression.items()})

    for field in fields:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                prepare_for_field(field)
        except IndexError:
            pass

    result.update(row)

    return result


def generate_graphml(node_frame, result):
    return {
        'graphml': to_graphml_string(node_frame.get_networkx_graph())
    }


# noinspection PyTypeChecker
def generate_overall_graphml(collected, result):
    time_to_z_scale = 1.0

    graphs_list = {}
    successors = {}
    node_counts = {}
    node_count_accumulator = 0

    graph = nx.Graph()

    for i, (meta, frame) in enumerate(list(collected.items())[:-1]):
        node_frame = frame.node_frame
        successors[i] = node_frame.self_to_successor
        g = node_frame.get_networkx_graph(with_z=time_to_z_scale)
        g = nx.relabel_nodes(g, lambda node_id: node_id + node_count_accumulator, copy=True)
        graphs_list[i] = g
        node_counts[i] = node_count_accumulator
        node_count_accumulator += len(g.nodes())
        graph.add_nodes_from(g.nodes(data=True))
        graph.add_edges_from(g.edges(data=True))

    for i, mapping in successors.items():
        if i + 1 == len(successors):
            break
        for relative_from_index, relative_to_index in enumerate(mapping):
            if relative_to_index == -1:
                continue
            from_index = node_counts[i] + relative_from_index
            to_index = node_counts[i + 1] + relative_to_index
            graph.add_edge(from_index, to_index)

    return {
        'overall_graphml': to_graphml_string(graph)
    }
