# -*- coding: utf-8 -*-
"""
documentation
"""

from .application import App

def pilyso_app_generator(**kwargs):
    class _PilysoApp(App):
        def options(self):
            return kwargs
    return _PilysoApp

def pipeline_to_app(pipeline):
    return pilyso_app_generator(name=pipeline.__name__.lower(), description=pipeline.__name__, pipeline=pipeline)

def run_pipeline(pipeline):
    return pipeline_to_app(pipeline)().main()

