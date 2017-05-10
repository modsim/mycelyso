# -*- coding: utf-8 -*-
"""
documentation
"""

from signal import signal, SIGUSR2
import traceback


def maintenance_interrupt(the_signal, frame):

    print("Interrupted at:")
    print(''.join(traceback.format_stack(frame)))
    print("Have a look at frame.f_globals and frame.f_locals")
    try:
        from IPython import embed
        embed()
    except ImportError:
        from code import interact
        interact(local=locals())
    print("... continuing")


def install_maintenance_interrupt():
    signal(SIGUSR2, maintenance_interrupt)

