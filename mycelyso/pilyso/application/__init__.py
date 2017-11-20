# -*- coding: utf-8 -*-
"""
The application module contains classes to quickly bootstrap image processing applications built around pipelines,
as well as some necessary pipeline building blocks.
"""

from .application import App, AppInterface, Meta
# noinspection PyUnresolvedReferences
from ..pipeline import *
from .helper import *

# noinspection PyUnresolvedReferences
from ..pipeline.executor import Skip


