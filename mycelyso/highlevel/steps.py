# -*- coding: utf-8 -*-
"""
documentation
"""

import networkx as nx

from itertools import product, chain

from scipy.stats import linregress

from pilyso.processing.processing import *
from pilyso.processing.pixelgraphs import *

import numpy


from ..misc.graphml import to_graphml_string
from ..misc.pairwise import pairwise
from ..misc.regression import prepare_optimized_regression

from ..processing.binarization import experimental_thresholding



from .pixelframe import PixelFrame
from .nodeframe import NodeFrame



class MycelysoSteps(object):

    @staticmethod
    def Binarize(image, binary=None):
        return experimental_thresholding(image)

    @staticmethod
    def ImageStatistics(image, calibration, result=None):
        return {
            'input_width': image.shape[1] * calibration,
            'input_height': image.shape[0] * calibration,
            'input_width_pixel': image.shape[1],
            'input_height_pixel': image.shape[0],
            'area': image.shape[1] * calibration * image.shape[0] * calibration,
            'area_pixel': image.shape[1] * image.shape[0]
        }

    @staticmethod
    def QuantifyBinary(binary, calibration, result=None):
        ones = numpy.sum(binary)
        total = binary.shape[0] * binary.shape[1]
        return {
            'covered_ratio': ones/total,
            'covered_area': ones * calibration * calibration,
            'covered_area_pixel': ones
        }

    @staticmethod
    def GraphStatistics(node_frame, result=None):
        # everything / 2 because it's a digraph/graph structure
        graph_edge_length = node_frame.adjacency.sum() / 2
        graph_edge_count = node_frame.adjacency.nnz / 2

        graph_node_count = node_frame.adjacency.shape[0]
        graph_junction_count = len(node_frame.every_junction)
        graph_endpoint_count = len(node_frame.every_endpoint)

        return {
            'graph_edge_length': graph_edge_length,
            'graph_edge_count': graph_edge_count,
            'graph_node_count': graph_node_count,
            'graph_junction_count': graph_junction_count,
            'graph_endpoint_count': graph_endpoint_count
        }

    @staticmethod
    def CleanUp(binary):
        binary = blur_and_threshold(binary)
        binary = fill_holes_smaller_than(binary, 1000)
        return binary

    @staticmethod
    def RemoveSmallStructures(binary):
        return filter_connected_components(binary, 10)

    #mm = 2**16-1
    #cv2.imwrite(nextfree('debug_graphlets' + ('%05d' % (meta.t,)), '.png'), numpy.c_[image, image*mm, cleaned_binarization*mm, skeleton*mm].astype(numpy.uint16))

    @staticmethod
    def ConvertToNodes(skeleton, timepoint, pixel_frame=None, node_frame=None):
        pixel_frame = PixelFrame(skeleton, timepoint)
        node_frame = NodeFrame(pixel_frame)
        node_frame.pf = None  # disconnect these
        return pixel_frame, node_frame

    @staticmethod
    def TrackMultipoint(collected):
        for result1, result2 in pairwise(collected.values()):
            result1.node_frame.track(result2.node_frame)
        return collected

    @staticmethod
    def ReconnectNodePixelFrame(pixel_frame=None, node_frame=None):
        node_frame.pf = pixel_frame
        return pixel_frame, node_frame

    @staticmethod
    def DebugPlotInjector(draw_params=None):
        Debug.enable('plot_pdf')
        DebugPlot.active = True
        DebugPlot.individual_and_merge = True

        return {
            'node_size': 7.5,
            'edge_color': 'green'
        }


    @staticmethod
    def MergeAttempt(draw_params, collected):
        key_list = list(collected.keys())



        # MERGE STEP!

        for i, (meta, result) in enumerate(collected.items()):

            frame = result.node_frame

            def get_angle(e):
                other_node = frame.adjacency[e, :].nonzero()[1][0]
                pos_e = frame.data[e]
                pos_o = frame.data[other_node]

                vec = pos_e - pos_o

                # arctan2 takes y, x parameters BUT our coordinates are stored the same way ;)
                angle = numpy.arctan2(vec[0], vec[1])

                while angle < 0.0:
                    angle += 2*numpy.pi

                return float(numpy.fmod(angle, 2*numpy.pi))

            def get_next_neighbor(e):
                other_node = frame.adjacency[e, :].nonzero()[1][0]
                return other_node, frame.adjacency[e, other_node]

            radius = 30.0
            angle = numpy.deg2rad(15.0)

            def contains_cycle(adjacency):
                stack = []
                from scipy.sparse import lil_matrix
                visited = lil_matrix(adjacency.shape, dtype=bool)
                for n, row in enumerate(adjacency):
                    indices, = numpy.where(row.getnnz(axis=0))
                    for m in indices:
                        if visited[n, m]:
                            continue
                        stack = [(n, m)]

            old_G, old_pos = frame.get_networkx_graph(return_positions=True)

            frame.adjacency = frame.adjacency.tolil()

            changes = 0

            worked_at = {None}
            while len(worked_at) > 0:

                if changes > 100:
                    break

                worked_at = set()

                angles = [get_angle(e) for e in frame.every_endpoint]
                if frame.endpoint_tree:
                    neighbors = frame.endpoint_tree.query_pairs(radius)
                else:
                    neighbors = []

                if len(neighbors) == 0:
                    continue

                for a, b in neighbors:
                    a, b = a + frame.endpoint_shift, b + frame.endpoint_shift
                    if a in worked_at or b in worked_at:
                        continue

                    dist = numpy.sqrt(((frame.data[a] - frame.data[b])**2).sum())

                    angle_a, angle_b = angles[a], angles[b]

                    delta_angle = abs(numpy.fmod((angle_a + numpy.pi) - angle_b + 2*numpy.pi, numpy.pi/2))

                    if delta_angle > (numpy.pi/2)/2:
                        delta_angle -= (numpy.pi/2)
                        delta_angle = abs(delta_angle)

                    #print(angle_a - angle_b, angle_a, angle_b, delta_angle, numpy.rad2deg(delta_angle))

                    if delta_angle == 0.0: # compare to eps
                        continue

                    if True or delta_angle < angle:
                        # dist
                        worked_at.add(a)
                        worked_at.add(b)

                        changes += 1

                        n_a, dist_a = get_next_neighbor(a)
                        n_b, dist_b = get_next_neighbor(b)

                        frame.adjacency[a, n_a] = 0
                        frame.adjacency[n_a, a] = 0
                        frame.adjacency[b, n_b] = 0
                        frame.adjacency[n_b, b] = 0

                        new_dist = n_a + n_b + dist

                        frame.adjacency[n_a, n_b] = new_dist
                        frame.adjacency[n_b, n_a] = new_dist

                if len(worked_at) > 0:
                    frame.cleanup_adjacency()

            frame.adjacency = frame.adjacency.tocsr()
            frame.generate_derived_data()

            with DebugPlot() as p:
                new_G, new_pos = frame.get_networkx_graph(return_positions=True)

                import networkx

                f, (p1, p2) = p.subplots(1, 2)
                print('Plotting ', frame.timepoint)
                p1.set_title('Old (t=%.2f)' % (frame.timepoint,))

                if frame.pf:
                    p1.imshow(frame.pf.image)

                networkx.draw(old_G, pos=old_pos, ax=p1, **draw_params)

                if frame.pf:
                    p1.set_xlim([0, frame.pf.image.shape[1]])
                    p1.set_ylim([frame.pf.image.shape[0], 0])

                p2.set_title('New (t=%.2f) [changes=%d]' % (frame.timepoint, changes))

                if frame.pf:
                    p2.imshow(frame.pf.image)

                networkx.draw(new_G, pos=new_pos, ax=p2, **draw_params)

                if frame.pf:
                    p2.set_xlim([0, frame.pf.image.shape[1]])
                    p2.set_ylim([frame.pf.image.shape[0], 0])

                p.tight_layout()

        ####
        # have to redo this because we screwed the trees above
        for key1, key2 in zip(key_list, key_list[1:]):
            collected[key1]['node_frame'].track(collected[key2]['node_frame'])
        ###

        return collected

    @staticmethod
    def IndividualTracking(collected, tracked_fragments=None, tracked_fragments_fates=None):
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

            # either: look for pairs of endpoints and junctions,
            # or endpoints and endpoints, if there are no junctions yet
            def valid_pairs():
                for e in frame.every_endpoint:
                    edges = frame.adjacency[e, :]
                    edge_indices = edges.nonzero()[1]

                    for other in edge_indices:
                        no_junctions = not any_in(frame.get_connected_nodes(other), frame.every_junction)
                        if frame.is_junction(other) or (no_junctions and frame.is_endpoint(other)):
                            yield None, e, other

            valid = []
            endpoints_used = set()

            for track_id, e, other in chain(last_valid, valid_pairs()):
                e_on_next, other_on_next = frame.self_to_successor[e], frame.self_to_successor[other]

                if e_on_next == -1 or other_on_next == -1:
                    # tracking error
                    fates[track_id] = "track aborted due to missing future node"
                    #print(fates[track_id])
                    continue

                def distance_condition(current, new):
                    return (new > current) or (abs(1.0 - (current / new)) < 0.2)

                distance = frame.shortest_paths[e, other]

                if next_frame.connected_components[e_on_next] != next_frame.connected_components[other_on_next]:
                    # print("*", frame.timepoint)
                    #  careful now: either the track broke, or we need to pick an alternative, fitting variant
                    # print(next_frame.__dict__)
                    # print(next_frame.adjacency.todense())

                    for _e_on_next, _other_on_next in product(frame.self_to_successor_alternatives[e], frame.self_to_successor_alternatives[other]):
                    #    print(next_frame.connected_components[_e_on_next],next_frame.connected_components[_other_on_next])
                    #    print(distance, next_frame.shortest_paths[_e_on_next, _other_on_next])
                        if _e_on_next != _other_on_next \
                            and next_frame.connected_components[_e_on_next] == next_frame.connected_components[_other_on_next]\
                            and next_frame.shortest_paths[_e_on_next, _other_on_next] < float('inf')\
                            and distance_condition(distance, next_frame.shortest_paths[_e_on_next, _other_on_next]):
                            e_on_next, other_on_next = _e_on_next, _other_on_next
                            break

                next_distance = next_frame.shortest_paths[e_on_next, other_on_next]

                distance_num = frame.shortest_paths_num[e, other]

                #print(track_id, i, e, other, e_on_next, other_on_next, distance, distance_num)

                if e in endpoints_used or other in endpoints_used:
                    fates[track_id] = "endpoints used otherly"
                    continue

                if distance == float('inf') or next_distance == float('inf'):
                    # the first one should never happen,
                    # the nodes are no longer connected on the next frame?
                    fates[track_id] = "formerly connected components became unconnected? (dist/nextdist %.4f %.4f)" % (distance, next_distance)
                    #print(fates[track_id])
                    continue

                if not distance_condition(distance, next_distance):
                    # a later distance was SHORTER than the current, that means tracking error or cycle in graph
                    fates[track_id] = "track aborted due to shortcut (cycle or mistrack) [last %f > next %f]" % (distance, next_distance)
                    #print(fates[track_id])
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

    # pixel data:
    # filled area, total area
    # "has outgrown" flag:

    # lets generate different data

    # first: endpoint lengths. endpoint to branchpoint, tracked over time
    # biological result => "mean length before branching"

    # as well: overall length / number of branchpoints

    #endpiece_tracking = {}
    # endpiece_tracking[i] = set()
    #endpiece_tracking[i] += {(e, other, e_on_next, other_on_next)}

    # find_linear_window(x, y, begin=float('nan'), end=float('nan'), window=0.1, condition=('rvalue', 'gt', 0.95), return_begin_end=False):


    @staticmethod
    def PrepareTrackedFragments(collected, tracked_fragments, tracked_fragments_fates, track_table=None, track_table_aux_tables=None):
        key_list = list(collected.keys())
        calibration = next(iter(collected.values()))['calibration']

        minimum_track_length = 3


        track_table = []

        track_table_aux_tables = []


        for track_id, track in tracked_fragments.items():
            if len(track) < minimum_track_length:
                continue

            track_list = [[i, track[i]] for i in sorted(track.keys())]

            times_lengths = numpy.array(
                [[collected[key_list[i]]['node_frame'].timepoint, calibration * distance]
                 for i, (e, other, e_on_next, other_on_next, distance, distance_num)
                 in track_list])

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

            distance_num_helper = numpy.array([distance_num for i, (e, other, e_on_next, other_on_next, distance, distance_num) in track_list])

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

            regression = linregress(times, lengths)._asdict()
            row.update({'plain_regression_' + k: v for k, v in regression.items()})
            regression = linregress(times, relative_lengths)._asdict()
            row.update({'normalized_regression_' + k: v for k, v in regression.items()})

            regression = linregress(times, numpy.log(lengths))._asdict()
            row.update({'logarithmic_plain_regression_' + k: v for k, v in regression.items()})
            regression = linregress(times, numpy.log(relative_lengths))._asdict()
            row.update({'logarithmic_normalized_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, lengths)
            row.update({'optimized_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, relative_lengths)
            row.update({'optimized_normalized_regression_' + k: v for k, v in regression.items()})


            regression = prepare_optimized_regression(times, numpy.log(lengths))
            row.update({'optimized_logarithmic_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(times, numpy.log(relative_lengths))
            row.update({'optimized_logarithmic_normalized_regression_' + k: v for k, v in regression.items()})

            track_table.append(row)

        return track_table, track_table_aux_tables

    @staticmethod
    def DebugPrintingAnalysis(draw_params, tracked_fragments, tracked_fragments_fates, collected):

        tracks = tracked_fragments
        fates = tracked_fragments_fates

        key_list = list(collected.keys())

        # pixel data:
        # filled area, total area
        # "has outgrown" flag:

        # lets generate different data

        # first: endpoint lengths. endpoint to branchpoint, tracked over time
        # biological result => "mean length before branching"

        # as well: overall length / number of branchpoints

        #endpiece_tracking = {}
        # endpiece_tracking[i] = set()
        #endpiece_tracking[i] += {(e, other, e_on_next, other_on_next)}

        calibration = next(iter(collected.values()))['calibration']


        DebugPlot.active = True  # takes too long otherwise


        from scipy.stats import linregress

        slopes = []
        r_slopes = []
        for track_id, track in tracks.items():
            if len(track) < 3:
                continue

            #if len(track) < 8: continue



            track_list = [[i, track[i]] for i in sorted(track.keys())]
            times_lengths = numpy.array(
                [[collected[key_list[i]]['node_frame'].timepoint, calibration * distance]
                 for i, (e, other, e_on_next, other_on_next, distance, distance_num)
                 in track_list])

            if (times_lengths[:, 1].max() - times_lengths[:, 1].min()) == 0.0:
                continue

            slope, intercept, r_value, p_value, std_err = \
                linregress(times_lengths[:, 0], times_lengths[:, 1])

            slopes.append([slope, times_lengths[-1, 0], times_lengths[-1, 0] - times_lengths[0, 0]])


            r_slope, r_intercept, r_r_value, r_p_value, r_std_err = \
                linregress(times_lengths[:, 0], times_lengths[:, 1] / times_lengths[:, 1].min())

            r_slopes.append([r_slope, times_lengths[-1, 0], times_lengths[-1, 0] - times_lengths[0, 0]])


            with DebugPlot() as p:
                p.title(u"%d (from/to %d/%d) items=%d\nslope=%.2f intercept=%.2f R²=%.2f\n%s" %
                        (track_id, min(track.keys()), max(track.keys()), len(track), slope, intercept, r_value,
                         'ultimate fate: ' + fates[track_id] if track_id in fates else "none?"))
                p.plot(times_lengths[:, 0], times_lengths[:, 1])

                x = numpy.linspace(times_lengths[0, 0], times_lengths[-1, 0])
                p.plot(x, x*slope + intercept)

            with DebugPlot() as p:
                p.title(u"RELATIVE %d (from/to %d/%d) items=%d\nslope=%.2f intercept=%.2f R²=%.2f\n%s" %
                        (track_id, min(track.keys()), max(track.keys()), len(track), r_slope, r_intercept, r_value,
                         'ultimate fate: ' + fates[track_id] if track_id in fates else "none?"))
                p.plot(times_lengths[:, 0], times_lengths[:, 1] / times_lengths[:, 1].min())

                x = numpy.linspace(times_lengths[0, 0], times_lengths[-1, 0])
                p.plot(x, x*r_slope + r_intercept)



            for i, t in track_list:
                nf = collected[key_list[i]]['node_frame']
                pf = nf.pf
                if pf is None:
                    continue

                with DebugPlot() as p:
                    p.title("%d timepoint=%d (abs=%.2f)\n%s" %
                            (track_id, i, pf.timepoint,
                            'ultimate fate: ' + fates[track_id] if track_id in fates else "none?"))
                    p.imshow(pf.image)
                    e, other, e_on_next, other_on_next, distance, distance_num = t

                    if not nf.is_endpoint(e):
                        print("this should never happen")
                        continue

                    first_coordinates = nf.data[e]

                    #if nf.is_endpoint(other):
                    second_coordinates = nf.data[other]



                    import networkx
                    G, positions = nf.get_networkx_graph(return_positions=True)
                    networkx.draw(G, pos=positions, **draw_params)

                    if nf.endpoint_tree:
                        p.scatter(nf.endpoint_tree.data[:, 1], nf.endpoint_tree.data[:, 0], color='red')
                    if nf.junction_tree:
                        p.scatter(nf.junction_tree.data[:, 1], nf.junction_tree.data[:, 0], color='blue')

                    p.plot([first_coordinates[1], second_coordinates[1]], [first_coordinates[0], second_coordinates[0]])

                    p.xlim([0, pf.image.shape[1]])
                    p.ylim([pf.image.shape[0], 0])


            ####
            for i in range(i+1, min(i+3, len(key_list))):
                nf = collected[key_list[i]]['node_frame']
                pf = nf.pf
                if pf is None:
                    continue

                with DebugPlot() as p:
                    p.title("%d (aftermath) timepoint=%d (abs=%.2f)\n%s" %
                            (track_id, i, pf.timepoint,
                             'ultimate fate: ' + fates[track_id] if track_id in fates else "none?"))
                    p.imshow(pf.image)

                    import networkx
                    G, positions = nf.get_networkx_graph(return_positions=True)
                    networkx.draw(G, pos=positions, **draw_params)

                    if nf.endpoint_tree:
                        p.scatter(nf.endpoint_tree.data[:, 1], nf.endpoint_tree.data[:, 0], color='red')
                    if nf.junction_tree:
                        p.scatter(nf.junction_tree.data[:, 1], nf.junction_tree.data[:, 0], color='blue')

                    p.xlim([0, pf.image.shape[1]])
                    p.ylim([pf.image.shape[0], 0])
                ####

            print(track_id)
            print(times_lengths.tolist())
            print("*")

        slopes = numpy.array(slopes)
        r_slopes = numpy.array(r_slopes)

        try:
            with DebugPlot() as p:
                    p.title("Slopes Boxplot\nmedian=%.2f mean=%.2f stddev=%.2f" %
                            (float(numpy.median(slopes[:, 0])), float(numpy.mean(slopes[:, 0])), float(numpy.std(slopes[:, 0]))))
                    p.boxplot(slopes[:, 0])
            with DebugPlot() as p:
                    p.title("Slopes Histogram\nmedian=%.2f mean=%.2f stddev=%.2f" %
                            (float(numpy.median(slopes[:, 0])), float(numpy.mean(slopes[:, 0])), float(numpy.std(slopes[:, 0]))))
                    try:
                        p.hist(slopes[:, 0])
                    except AttributeError:
                        pass

            with DebugPlot() as p:
                    p.title("Slopes Scatter\nmedian=%.2f mean=%.2f stddev=%.2f" %
                            (float(numpy.median(slopes[:, 0])), float(numpy.mean(slopes[:, 0])), float(numpy.std(slopes[:, 0]))))
                    p.scatter(slopes[:, 1], slopes[:, 0], s=2.5*slopes[:, 2], linewidths=(0.0,))
                    p.xlabel('timepoint')
                    p.ylabel('growth rate (slope)')

            with DebugPlot() as p:
                    p.title("Relative Slopes Boxplot\nmedian=%.2f mean=%.2f stddev=%.2f" %
                            (float(numpy.median(r_slopes[:, 0])), float(numpy.mean(r_slopes[:, 0])), float(numpy.std(r_slopes[:, 0]))))
                    p.boxplot(r_slopes[:, 0])
            with DebugPlot() as p:
                    p.title("Relative Slopes Histogram\nmedian=%.2f mean=%.2f stddev=%.2f" %
                            (float(numpy.median(r_slopes[:, 0])), float(numpy.mean(r_slopes[:, 0])), float(numpy.std(r_slopes[:, 0]))))
                    try:
                        p.hist(slopes[:, 0])
                    except AttributeError:
                        pass
            with DebugPlot() as p:
                    p.title("Relative Slopes Scatter\nmedian=%.2f mean=%.2f stddev=%.2f" %
                            (float(numpy.median(r_slopes[:, 0])), float(numpy.mean(r_slopes[:, 0])), float(numpy.std(r_slopes[:, 0]))))
                    p.scatter(r_slopes[:, 1], r_slopes[:, 0], s=2.5*r_slopes[:, 2], linewidths=(0.0,))
                    p.xlabel('timepoint')
                    p.ylabel('growth rate (slope) relative')
        except RuntimeError:
            pass

        DebugPlot.call_exit_handlers()

        return collected

    @staticmethod
    def PreparePositionRegressions(collected, result):

        fields = ['covered_ratio', 'covered_area', 'graph_edge_length', 'graph_edge_count', 'graph_node_count', 'graph_junction_count', 'graph_endpoint_count']

        row = {}

        def prepare_for_field(field):
            data = numpy.array([[f.timepoint, f[field]] for f in collected.values()])

            regression = linregress(data[:, 0], data[:, 1])._asdict()
            row.update({field + '_linear_regression_' + k: v for k, v in regression.items()})

            regression = linregress(data[:, 0], numpy.log(data[:, 1]))._asdict()
            row.update({field + '_logarithmic_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(data[:, 0], data[:, 1])
            row.update({field + '_optimized_linear_regression_' + k: v for k, v in regression.items()})

            regression = prepare_optimized_regression(data[:, 0], numpy.log(data[:, 1]))
            row.update({field + '_optimized_logarithmic_regression_' + k: v for k, v in regression.items()})

        for field in fields:
            prepare_for_field(field)

        result.update(row)

        return result

    @staticmethod
    def GenerateGraphML(node_frame, result):
        return {
            'graphml': to_graphml_string(node_frame.get_networkx_graph())
        }

    @staticmethod
    def GenerateOverallGraphML(collected, result):
        time_to_z_scale = 25.0

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
