# -*- coding: utf-8 -*-
"""
documentation
"""

from pilyso.application import App, PipelineExecutionContext, PipelineEnvironment, Every, Collected, Meta, Skip
from pilyso.imagestack import ImageStack
from pilyso.steps import \
    image_source, pull_metadata_from_image, substract_start_frame, rescale_image_to_uint8, set_result, Delete
from os.path import basename, abspath

from .steps import *

from pilyso.misc.h5writer import hdf5_output, hdf5_node_name
from .. import BANNER


class Mycelyso(App):
    def options(self):
        return {
            'name': "mycelyso",
            'description': "",
            'banner': BANNER,
            'pipeline': MycelysoPipeline
        }

    def arguments(self, argparser):
        argparser.add_argument('--meta', '--meta', dest='meta', default='')
        argparser.add_argument('--output', '--output', dest='output', default='output.h5')


class MycelysoPipeline(PipelineExecutionContext):
    def __init__(self, args):
        absolute_input = abspath(args.input)
        h5nodename = hdf5_node_name(absolute_input)

        self.pipeline_environment = PipelineEnvironment(ims=ImageStack(args.input))

        per_image = self.add_stage(Meta(pos=Every, t=Every))

        # read the image
        per_image |= image_source
        per_image |= pull_metadata_from_image

        per_image |= lambda image, raw_image=None: image
        per_image |= lambda image, raw_unrotated_image=None: image

        # define what we want (per image) as results

        per_image |= set_result(
            reference_timepoint=1,
            filename_complete=absolute_input,
            filename=basename(absolute_input),
            metadata=args.meta,
            result_table={
                '_plain': [
                    'calibration', 'timepoint', 'input_height',
                    'input_width', 'area', 'covered_ratio', 'covered_area',
                    'graph_edge_length', 'graph_edge_count', 'graph_node_count',
                    'graph_junction_count', 'graph_endpoint_count',
                    'filename', 'metadata', 'shift_x', 'shift_y',
                    'crop_t', 'crop_b', 'crop_l', 'crop_r'
                ],
                'graphml': 'data',
                # 'image': 'image',
                # 'raw_unrotated_image': 'image',
                # 'raw_image': 'image',
                'skeleton': 'image',
                'binary': 'image'
            }
        )

        per_image |= substract_start_frame

        # TODO add registration routine, in case no box cropping is to be done

        # crop the box

        # per_image |= box_detector_cropper
        # per_image |= create_boxcrop_from_subtracted_image

        # per_image |= set_result(subtracted_image=Delete)

        per_image |= rescale_image_to_uint8

        per_image |= set_result(raw_unrotated_image=Delete, raw_image=Delete, subtracted_image=Delete)

        def skip_if_image_is_below_size(min_height=4, min_width=4):
            def _inner(image, meta):
                if image.shape[0] < min_height or image.shape[1] < min_width:
                    raise Skip(Meta(pos=meta.pos, t=Collected)) from None

                return image, meta
            return _inner

        # NEW MODIFICATION
        per_image |= lambda image: image[60:-60, 10:-10]

        per_image |= skip_if_image_is_below_size(4, 4)

        # generate statistics of the image
        per_image |= image_statistics

        # binarize
        per_image |= binarize

        # ... and cleanup
        per_image |= clean_up
        # generate statistics of the binarized image

        # self.add_step(Meta(t=Every, pos=Every), image_shower('binary'))

        per_image |= quantify_binary

        per_image |= skeletonize

        # self.add_step(Meta(t=Every, pos=Every), image_shower('skeleton'))

        per_image |= remove_small_structures

        # 'binary', 'skeleton' are kept!
        per_image |= convert_to_nodes

        per_image |= set_result(image=Delete)
        per_image |= set_result(pixel_frame=Delete)
        # per_image |= lambda result: print(result)
        # self.add_step(Meta(t=Every, pos=Every), MycelysoSteps.ReconnectNodePixelFrame)

        per_image |= graph_statistics
        per_image |= generate_graphml

        per_position = self.add_stage(Meta(pos=Every, t=Collected))

        per_position |= track_multipoint

        per_position |= generate_overall_graphml

        # per_position |= AnalyzeMultipoint
        #

        # per_position |= DebugPlotInjector
        # self.add_step(Meta(t=Collected, pos=Every), MergeAttempt)
        per_position |= individual_tracking

        # self.add_step(Meta(t=Collected, pos=Every), DebugPrintingAnalysis)

        per_position |= prepare_tracked_fragments

        per_position |= prepare_position_regressions

        per_position |= lambda meta, meta_pos=None: meta.pos

        per_position |= set_result(
            filename_complete=absolute_input,
            filename=basename(absolute_input),
            metadata=args.meta,
            result_table={
               '_plain': [
                   'metadata',
                   'filename_complete',
                   'filename',
                   'meta_pos',
                   '*_regression_*'
               ],
               'overall_graphml': 'data',
               'track_table': 'table',
               'track_table_aux_tables': 'table'
            }
        )

        per_position |= hdf5_output(args.output, h5nodename)

        def black_hole(result):
            for k in list(result.keys()):
                del result[k]
            del result
            return {}

        per_position |= black_hole
