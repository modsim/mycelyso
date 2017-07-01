# -*- coding: utf-8 -*-
"""
The recursionlimit_raise submodule raises the recursion limit of the Python interpreter upon import.
(Currently to 10**9)
"""

import sys
# graphs, recursion, let's raise the bar a bit ;)
sys.setrecursionlimit(int(1E9))
