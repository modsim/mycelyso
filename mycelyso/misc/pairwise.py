# -*- coding: utf-8 -*-
"""
documentation
"""

from itertools import tee


# from the itertools help https://docs.python.org/2/library/itertools.html
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
# end
