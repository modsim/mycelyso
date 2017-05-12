# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy as np

from networkx.readwrite import GraphMLWriter


def to_graphml_writer(g):
    writer = GraphMLWriter()
    writer.xml_type.update({
        np.float32: 'double',
        np.float64: 'double'
    })
    writer.add_graph_element(g)
    return writer


def write_graphml(g, name):
    writer = to_graphml_writer(g)
    writer.dump(name)


def to_graphml_string(g):
    writer = to_graphml_writer(g)
    return str(writer)
