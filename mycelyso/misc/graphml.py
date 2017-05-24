# -*- coding: utf-8 -*-
"""
The graphml module contains output routines to output GraphML structured data from internal graph representations.
"""

import numpy as np

from networkx.readwrite import GraphMLWriter


def to_graphml_writer(g):
    """
    Takes a networkx graph and returns a GraphMLWriter containing the graph.
    
    :param g: graph
    :return: GraphMLWriter instance
    """
    writer = GraphMLWriter()
    writer.xml_type.update({
        np.float32: 'double',
        np.float64: 'double'
    })
    writer.add_graph_element(g)
    return writer


def write_graphml(g, name):
    """
    Writes a networkx graph in GraphML format to a file.
    
    :param g: graph
    :param name: filename
    :return: 
    """
    writer = to_graphml_writer(g)
    writer.dump(name)


def to_graphml_string(g):
    """
    Converts a networkx graph to a GraphML representation.
    
    :param g: graph
    :return: graphml string
    """
    writer = to_graphml_writer(g)
    return str(writer)
