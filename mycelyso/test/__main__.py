# -*- coding: utf-8 -*-
"""
The test module's __main__ contains the main() function to run the doctests.
"""

import sys
import doctest


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

    modules_to_test = [
        mycelyso.highlevel.nodeframe,
        mycelyso.highlevel.pipeline,
        mycelyso.highlevel.pixelframe,
        mycelyso.highlevel.steps,
        #
        mycelyso.misc.graphml,
        mycelyso.misc.regression,
        mycelyso.misc.util,
        #
        mycelyso.pilyso.application.application,
        mycelyso.pilyso.application.helper,
        #
        mycelyso.pilyso.imagestack.imagestack,
        #
        mycelyso.pilyso.misc.h5writer,
        mycelyso.pilyso.misc.processpool,
        #
        mycelyso.pilyso.pipeline.executor,
        mycelyso.pilyso.pipeline.pipeline,
        #
        mycelyso.pilyso.steps.steps,
        #
        mycelyso.processing.binarization,
        mycelyso.processing.pixelgraphs,
    ]

    total_failures, total_tests = 0, 0

    for a_module in modules_to_test:
        failures, tests = doctest.testmod(a_module)
        total_failures += failures
        total_tests += tests

    print("Run %d tests in total." % (total_tests,))

    if total_failures > 0:
        print("Test failures occurred, exiting with non-zero status.")
        sys.exit(1)


if __name__ == '__main__':
    main()
