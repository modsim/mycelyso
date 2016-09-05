# -*- coding: utf-8 -*-
"""
documentation
"""

import sys
import logging
import argparse
import multiprocessing

from ..imagestack.imagestack import ImageStack, Dimensions

from ..imagestack.readers import *
from ..pipeline import PipelineEnvironment, PipelineExecutor


from molyso.generic.etc import parse_range, prettify_numpy_array



class AppInterface(object):
    def options(self):
        return {}

    def arguments(self, argparser):
        return

    def setup(self, pipeline_executor):
        return

from collections import namedtuple

Meta = namedtuple('Meta', ['pos', 't'])

class App(AppInterface):
    @staticmethod
    def _internal_option_defaults():
        return {
            'name':
                "processor",
            'description':
                "processor",
            'banner':
                "",
            'pipeline': None
        }

    _internal_options = None

    @property
    def internal_options(self):
        if self._internal_options is None:
            self._internal_options = self._internal_option_defaults()
            self._internal_options.update(self.options())
        return self._internal_options

    _log = None

    @property
    def log(self):
        if self._log is None:
            self._log = logging.getLogger(self.internal_options['name'])
        return self._log

    def _create_argparser(self):
        argparser = argparse.ArgumentParser(description=self.internal_options['description'])

        def _error(message=''):
            self.log.info(self.internal_options['banner'])
            argparser.print_help()
            self.log.error("command line argument error: %s", message)
            sys.exit(1)

        argparser.error = _error

        return argparser


    @staticmethod
    def _arguments(argparser):
        argparser.add_argument('input', metavar='input', type=str, help="input file")
        argparser.add_argument('-m', '--module', dest='modules', type=str, default=None, action='append')
        argparser.add_argument('-n', '--processes', dest='processes', default=-1, type=int)
        argparser.add_argument('--prompt', '--prompt', dest='wait_on_start', default=False, action='store_true')
        argparser.add_argument('-t', '--timepoints', dest='timepoints', default='0-', type=str)
        argparser.add_argument('-p', '--positions', dest='positions', default='0-', type=str)

    args = None

    def main(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s %(message)s")

        argparser = self._create_argparser()

        self._arguments(argparser)
        self.arguments(argparser)

        self.args = argparser.parse_args()

        if self.args.wait_on_start:
            _ = input("Press enter to continue.")

        self.log.info(self.internal_options['banner'])
        self.log.info("Started %s.", self.internal_options['name'])

        ims = ImageStack(self.args.input).view(Dimensions.PositionXY, Dimensions.Time)

        #self._setup_modules()

        self.positions = parse_range(self.args.positions, maximum=ims.size[Dimensions.PositionXY])
        self.timepoints = parse_range(self.args.timepoints, maximum=ims.size[Dimensions.Time])

        self.log.info(
            "Beginning Processing:\n%s\n%s",
            prettify_numpy_array(self.positions,  "Positions : "),
            prettify_numpy_array(self.timepoints, "Timepoints: ")
        )

        if self.args.processes < 0:
            self.args.processes = multiprocessing.cpu_count()
        elif self.args.processes == 0:
            self.args.processes = False

        def progress_bar(num):
            return fancy_progress_bar(range(num))

        pe = PipelineExecutor()


        pe.multiprocessing = self.args.processes

        pe.set_workload(Meta, Meta(pos=self.positions, t=self.timepoints), Meta(t=1, pos=2))

        self.setup(pe)

        pe.run()

        self.log.info("Finished %s.", self.internal_options['name'])

    # default implementation
    def setup(self, pipeline_executor):
        pipeline_executor.initialize(self.internal_options['pipeline'], self.args)

