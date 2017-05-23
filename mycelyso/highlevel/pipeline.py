# -*- coding: utf-8 -*-
"""
documentation
"""

from ..pilyso.application import App, PipelineExecutionContext, PipelineEnvironment, Every, Collected, Meta, Skip
from ..pilyso.imagestack import ImageStack
from ..pilyso.steps import \
    image_source, pull_metadata_from_image, substract_start_frame, rescale_image_to_uint8, set_result, Delete, \
    box_detection, create_boxcrop_from_subtracted_image
from os.path import basename, abspath

from .steps import *

from ..pilyso.misc.h5writer import hdf5_output, hdf5_node_name
from .. import __banner__


class Mycelyso(App):
    """
    The Mycelyso App, implementing a pilyso App.
    """
    def options(self):
        return {
            'name': "mycelyso",
            'description': "",
            'banner': __banner__,
            'pipeline': MycelysoPipeline
        }

    def arguments(self, argparser):
        argparser.add_argument('--meta', '--meta', dest='meta', default='')
        argparser.add_argument('--box', '--detect-box-structure', dest='box_detection',
                               default=False, action='store_true')
        argparser.add_argument('--cw', '--crop-width', dest='crop_width', default=0, type=int)
        argparser.add_argument('--ch', '--crop-height', dest='crop_height', default=0, type=int)
        argparser.add_argument('--si', '--store-image', dest='store_image', default=False, action='store_true')
        argparser.add_argument('--output', '--output', dest='output', default='output.h5')


class MycelysoPipeline(PipelineExecutionContext):
    """
    The MycelysoPipeline, defining the pipeline (with slight alterations based upon arguments passed via command line).
    """
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

        per_image |= set_empty_crops


        # define what we want (per image) as results

        result_table = {
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

        if args.store_image:
            result_table['image'] = 'image'

        per_image |= set_result(
            reference_timepoint=1,
            filename_complete=absolute_input,
            filename=basename(absolute_input),
            metadata=args.meta,
            result_table=result_table
        )

        per_image |= substract_start_frame

        if args.box_detection:
            per_image |= box_detection
            per_image |= create_boxcrop_from_subtracted_image

        per_image |= rescale_image_to_uint8

        per_image |= set_result(raw_unrotated_image=Delete, raw_image=Delete, subtracted_image=Delete)

        per_image |= lambda image: image[
                                   args.crop_height:-(args.crop_height if args.crop_height > 0 else 1),
                                   args.crop_width:-(args.crop_width if args.crop_width > 0 else 1)
                                   ]

        per_image |= lambda crop_t, crop_b, crop_l, crop_r: (
            crop_t + args.crop_height,
            crop_b - args.crop_height,
            crop_l + args.crop_width,
            crop_r - args.crop_width
        )

        per_image |= skip_if_image_is_below_size(4, 4)

        # generate statistics of the image
        per_image |= image_statistics

        # binarize
        per_image |= binarize

        # ... and cleanup
        per_image |= clean_up

        per_image |= remove_small_structures

        per_image |= remove_border_artifacts

        # generate statistics of the binarized image
        per_image |= quantify_binary

        per_image |= skeletonize

        # 'binary', 'skeleton' are kept!
        per_image |= convert_to_nodes

        if not args.store_image:
            per_image |= set_result(image=Delete)

        per_image |= set_result(pixel_frame=Delete)

        per_image |= graph_statistics
        per_image |= generate_graphml

        per_position = self.add_stage(Meta(pos=Every, t=Collected))

        per_position |= track_multipoint

        per_position |= generate_overall_graphml

        per_position |= individual_tracking

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
