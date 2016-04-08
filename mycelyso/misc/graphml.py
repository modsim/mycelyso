# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy

from networkx.readwrite import GraphMLWriter

def to_graphml_writer(G):
    writer = GraphMLWriter()
    writer.xml_type.update({
        numpy.float32: 'double',
        numpy.float64: 'double'
    })
    writer.add_graph_element(G)
    return writer

def write_graphml(G, name):
    writer = to_graphml_writer(G)
    writer.dump(name)

def to_graphml_string(G):
    writer = to_graphml_writer(G)
    return str(writer)