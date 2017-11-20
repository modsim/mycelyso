# -*- coding: utf-8 -*-
"""
The maintenance_interrupt submodule provides the install_maintenance_interrupt function, which allows under *NIX systems
to interrupt the running process and spawn a IPython shell within the running process.
"""

try:
    from signal import signal, SIGUSR2
except ImportError:
    signal, SIGUSR2 = None, None

import traceback


def maintenance_interrupt(the_signal, frame):

    print("Interrupted at:")
    print(''.join(traceback.format_stack(frame)))
    print("Have a look at frame.f_globals and frame.f_locals")
    try:
        from IPython import embed
        embed()
    except ImportError:
        embed = None
        from code import interact
        interact(local=locals())
    print("... continuing")


def install_maintenance_interrupt():
    if signal is None or SIGUSR2 is None:
        return  # no functionality in windows
    signal(SIGUSR2, maintenance_interrupt)

