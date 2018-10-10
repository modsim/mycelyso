# -*- coding: utf-8 -*-
"""
The test module's __main__ contains the main() function to run the doctests.
"""

import sys
import doctest

from types import ModuleType


def collect_modules_recursive(start, blacklist=None):
    """
    Collects all modules and submodules in a recursive manner.

    :param start: the top module to start from
    :param blacklist: a string or list of strings of module (sub)names which should be ignored.
    :return:
    """

    if not isinstance(blacklist, list):
        blacklist = [blacklist]

    collector = set()

    def _inner(current):
        collector.add(current)
        for another in dir(current):
            another = getattr(current, another)
            if isinstance(another, ModuleType):
                if another.__name__.startswith(start.__name__):

                    ok = True
                    for blacklisted in blacklist:
                        if blacklisted in another.__name__:
                            ok = False

                    if ok:
                        _inner(another)

    _inner(start)

    return list(sorted(collector, key=lambda module: module.__name__))


def run_tests_recursively(start_module, blacklist=None, exit=True, quiet=False):
    """
    Runs doctests recursively.

    :param start_module: the top module to start from
    :param blacklist: a string or list of strings of module (sub)names which should be ignored.
    :param exit: whether to exit with return code
    :param quiet: whether to print infos about tests
    :return:
    """
    total_failures, total_tests = 0, 0

    for a_module in collect_modules_recursive(start_module, blacklist):
        failures, tests = doctest.testmod(a_module)
        total_failures += failures
        total_tests += tests

    if not quiet:
        print("Run %d tests in total." % (total_tests,))

    if total_failures > 0:
        if not quiet:
            print("Test failures occurred, exiting with non-zero status.")

        if exit:
            sys.exit(1)


# noinspection PyUnresolvedReferences
def main():
    """
    Runs all the doctests.

    """

    import mycelyso.highlevel.nodeframe
    import mycelyso.highlevel.pipeline
    import mycelyso.highlevel.pixelframe
    import mycelyso.highlevel.steps

    import mycelyso.misc.graphml
    import mycelyso.misc.regression
    import mycelyso.misc.util

    import mycelyso.pilyso.application.application
    import mycelyso.pilyso.application.helper

    import mycelyso.pilyso.imagestack.imagestack

    import mycelyso.pilyso.misc.h5writer
    import mycelyso.pilyso.misc.processpool

    import mycelyso.pilyso.pipeline.executor
    import mycelyso.pilyso.pipeline.pipeline

    import mycelyso.pilyso.steps.steps

    import mycelyso.processing.binarization
    import mycelyso.processing.pixelgraphs

    run_tests_recursively(mycelyso, blacklist='czifile')


if __name__ == '__main__':
    main()
