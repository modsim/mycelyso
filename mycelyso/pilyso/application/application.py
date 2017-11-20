# -*- coding: utf-8 -*-
"""
The application submodule contains the base class for applications, as well auxiliary functions.
"""

import argparse
import logging
import multiprocessing
# noinspection PyUnresolvedReferences
import sys
import numpy as np

from tunable import TunableManager
from ..misc.hacks.maintenance_interrupt import install_maintenance_interrupt
from ..misc.hacks.recursionlimit_raise import *
from ..misc.hacks.multiprocessing_patch import *

from collections import namedtuple

from ..imagestack.imagestack import ImageStack, Dimensions
# noinspection PyUnresolvedReferences
from ..imagestack.readers import *
from ..pipeline import PipelineExecutor


def parse_range(s, maximum=0):
    """
    Parses a range string.
    
    :param s:
    :param maximum:
    :return:
    """
    maximum -= 1
    splits = s.replace(' ', '').replace(';', ',').split(',')

    ranges = []
    remove = []

    not_values = False

    for frag in splits:
        if frag[0] == '~':
            not_values = not not_values
            frag = frag[1:]

        if '-' in frag:
            f, t = frag.split('-')

            interval = 1

            if '%' in t:
                t, _interval = t.split('%')
                interval = int(_interval)

            if t == '':
                t = maximum

            f, t = int(f), int(t)

            t = min(t, maximum)

            parsed_fragment = range(f, t + 1, interval)
        else:
            parsed_fragment = [int(frag)]

        if not_values:
            remove += parsed_fragment
        else:
            ranges += parsed_fragment

    return list(sorted(set(ranges) - set(remove)))


def prettify_numpy_array(arr, space_or_prefix):
    """
    Returns a properly indented string representation of a numpy array.

    :param arr:
    :param space_or_prefix:
    :return:
    """
    six_spaces = ' ' * 6
    prepared = repr(np.array(arr)).replace(')', '').replace('array(', six_spaces)
    if isinstance(space_or_prefix, int):
        return prepared.replace(six_spaces, ' ' * space_or_prefix)
    else:
        return space_or_prefix + prepared.replace(six_spaces, ' ' * len(space_or_prefix)).lstrip()


class AppInterface(object):
    """
    Interface to be implemented by Apps utilizing pilyso's App infrastructure.
    """
    def options(self):
        return {}

    def arguments(self, argparser):
        return

    def setup(self, pipeline_executor):
        return


Meta = namedtuple('Meta', ['pos', 't'])


class App(AppInterface):
    """
    Base class implementing most of the App "in the back" works.
    """
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
        argparser.add_argument('-tp', '--timepoints', dest='timepoints', default='0-', type=str)
        argparser.add_argument('-mp', '--positions', dest='positions', default='0-', type=str)

        TunableManager.register_argparser(argparser)

    def handle_args(self):
        pass

    args = None

    def main(self):
        """
        Main entry point for App apps.
        :return: 
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s %(message)s")

        install_maintenance_interrupt()

        argparser = self._create_argparser()

        self._arguments(argparser)
        self.arguments(argparser)

        self.log.info(self.internal_options['banner'])
        self.args = argparser.parse_args()

        self.handle_args()

        if self.args.wait_on_start:
            _ = input("Press enter to continue.")

        self.log.info("Started %s.", self.internal_options['name'])

        ims = ImageStack(self.args.input).view(Dimensions.PositionXY, Dimensions.Time)

        # self._setup_modules()

        self.positions = parse_range(self.args.positions, maximum=ims.size[Dimensions.PositionXY])
        self.timepoints = parse_range(self.args.timepoints, maximum=ims.size[Dimensions.Time])

        self.log.info("Tunable Hash: %s" % (TunableManager.get_hash()))

        self.log.info(
            "Beginning Processing:\n%s\n%s",
            prettify_numpy_array(self.positions,  "Positions : "),
            prettify_numpy_array(self.timepoints, "Timepoints: ")
        )

        if self.args.processes < 0:
            self.args.processes = multiprocessing.cpu_count()
        elif self.args.processes == 0:
            self.args.processes = False

        self.pe = PipelineExecutor()

        self.pe.multiprocessing = self.args.processes

        self.pe.set_workload(Meta, Meta(pos=self.positions, t=self.timepoints), Meta(t=1, pos=2))

        self.setup(self.pe)

        self.run()

        self.pe.close()

        self.log.info("Finished %s.", self.internal_options['name'])

    def run(self):
        self.pe.run()

    # default implementation
    def setup(self, pipeline_executor):
        pipeline_executor.initialize(self.internal_options['pipeline'], self.args)

