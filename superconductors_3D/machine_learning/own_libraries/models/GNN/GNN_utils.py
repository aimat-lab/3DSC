#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:26:09 2021

@author: Timo Sommer

Utils for Graph NNs.
"""

from pymatgen.core.structure import Structure
import pandas as pd
from superconductors_3D.utils.projectpaths import projectpath
import json
from megnet.data.crystal import CrystalGraphDisordered, get_elemental_embeddings
import megnet
from megnet.models import MEGNetModel
from  pymatgen.core.periodic_table import Species
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

def save_graph(graph_name, graph):        
    abs_graph_name = projectpath(graph_name)
    
    with open(abs_graph_name, 'w') as graph_file:
        json.dump(graph, graph_file)

def load_graph(graph_name):        
    abs_graph_name = projectpath(graph_name)
    
    with open(abs_graph_name) as graph_file:
        try:
            return json.load(graph_file)          
        except json.JSONDecodeError as e:
            raise Warning(f'Could not open file {graph_name}. Error message:', e)

def plot_graph(graph):
    """Plots a graph with nodes and edges.
    """
    G = nx.MultiGraph()
    G.add_edges_from(list(zip(graph['index1'], graph['index2'])))
    nx.draw(G)
    plt.show()

def get_graph_from_structure(cif_relpath, nfeat_bond=100, r_cutoff=4, gaussian_width=0.5, add_state_values=[], model=None, use_embedding=False):
    """Returns graph from structure. `cif_relpath` can be either a relative path to structure or the structure itself. The atoms are encoded as dictionary with their elements and occupancies.
    """
    try:
        structure = Structure.from_file(projectpath(cif_relpath))
    except TypeError:
        structure = cif_relpath
    
    if model is None:        
        gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)    
        graph_converter = CrystalGraphDisordered(cutoff=r_cutoff)
        model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
    graph = model.graph_converter.convert(structure)
    
    # json can't deal with numpy arrays, only lists
    for prop in ['bond', 'index1', 'index2']:
        graph[prop] = [val.item() for val in graph[prop]]
    graph['state'] = [[val.item() for val in graph['state'][0]]]
    
    if use_embedding:
        embedding = AtomEmbeddingMap()
        try:
            graph['atom'] = embedding.convert(graph['atom']).tolist()
        except KeyError as e:
            # Not all elements (e.g. Am) are covered by the embedding
            print('Exception: ', e)
            return np.nan
    
    for val in add_state_values:
        graph['state'][0].append(float(val))
            
    return graph


def graph_features(row, state_features, graph_dir):
    """This function takes a row as input with a graph where the nodes are represented as dictionary of element and occupancy. It returns a graph with a numerical representation of the atoms (the learnt elemental embedding from pymatgen). Additionally you can specify features for the states from the row entries.
    """
    graph_rel_path = row['graph']
    graph = load_graph(graph_rel_path)
    
    # atom features
    embedding = AtomEmbeddingMap()
    try:
        graph['atom'] = embedding.convert(graph['atom']).tolist()
    except KeyError as e:
        # Not all elements (e.g. Am) are covered by the embedding
        print('Exception: ', e)
        return np.nan
    
    # state features
    for feat in state_features:
        value = row[feat]
        graph['state'][0].append(float(value))
    
    # Save graph
    graph_basename = os.path.basename(graph_rel_path)
    if not graph_dir is None:
        new_graph_rel_path = os.path.join(graph_dir, graph_basename)
        save_graph(new_graph_rel_path, graph)
    else:
        new_graph_rel_path = graph_basename
    
    return new_graph_rel_path


class AtomEmbeddingMap(megnet.data.graph.Converter):
    """
    Class for converting a dictionary of occupancies to a numerical vector of each atom.
    """

    def __init__(self, embedding_dict: dict = None):
        """
        Args:
            embedding_dict (dict): element to element vector dictionary
        """
        if embedding_dict is None:
            embedding_dict = get_elemental_embeddings()
        self.embedding_dict = embedding_dict

    def convert(self, atoms: list) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        features = []
        for atom in atoms:
            emb = 0
            for k, v in atom.items():
                
                try:
                    # If the element has an oxidation state
                    k = Species.from_string(k).symbol
                except ValueError:
                    pass
                
                emb += np.array(self.embedding_dict[k]) * v
            features.append(emb)
        return np.array(features).reshape((len(atoms), -1))
    