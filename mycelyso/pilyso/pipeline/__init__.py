# -*- coding: utf-8 -*-
"""
The pipeline submodule contains the infrastructure for building image processing pipelines.
"""

from .executor import PipelineExecutor, Every, Collected
from .pipeline import Pipeline, PipelineExecutionContext, PipelineEnvironment
