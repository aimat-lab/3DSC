#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:39:17 2021

@author: Timo Sommer

This script reads in the json data provided by the MEGNet people and outputs a directory with the cifs and graphs and a df with the targets and the paths of the cifs and graphs.
"""

n_cpus = 3          # change this




# Parse input and overwrite.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
args = parser.parse_args()
n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus



from superconductors_3D.utils.projectpaths import projectpath
import os


in_json_MP_data = projectpath('superconductors_3D', 'test_MEGNet', 'Data', 'Materials_Project', 'mp.2018.6.1.json')

out_cif_dir = os.path.join('superconductors_3D', 'test_MEGNet', 'Data', 'Materials_Project', 'cifs')
out_graph_dir = os.path.join('superconductors_3D', 'test_MEGNet', 'Data', 'Materials_Project', 'graphs')
out_csv = projectpath('superconductors_3D', 'test_MEGNet', 'Data', 'Materials_Project', 'all_crystals.csv')
comment_csv = 'All crystal structures that were used in the MEGNet paper.'


import json
import pandas as pd
from tqdm import tqdm
import os
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
import joblib
import shutil
from pymatgen.core.structure import Structure
import numpy as np
from megnet.data.crystal import CrystalGraph, CrystalGraphDisordered
from megnet.models import MEGNetModel
from joblib import Parallel, delayed



def save_cif(rel_path, cif):
    abs_cif_name = projectpath(rel_path)
    with open(abs_cif_name, 'w') as cif_file:
        cif_file.write(cif)  
        
def save_graph(graph_name, graph):        
    abs_graph_name = projectpath(graph_name)
    
    with open(abs_graph_name, 'w') as graph_file:
        json.dump(graph, graph_file)            

def get_graph_from_structure(cif_relpath, disordered=False, nfeat_bond=10, r_cutoff=4, gaussian_width=0.5):
    """Returns graph from structure. If disordered=True, returns a graph with elemental embeddings.
    """
    structure = Structure.from_file(projectpath(cif_relpath))
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    
    if disordered:
        graph_converter = CrystalGraphDisordered(cutoff=r_cutoff)
    else:
        graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
    graph = model.graph_converter.convert(structure)
    
    # json can't deal with numpy arrays, only lists
    for prop in ['bond', 'index1', 'index2']:
        graph[prop] = [val.item() for val in graph[prop]]
    graph['state'] = [[val.item() for val in graph['state'][0]]]
    
    return graph

def save_data_from_entry(entry):
    # Save cif to file.
    cif_name = os.path.join(out_cif_dir, entry['material_id'] + '.cif')
    save_cif(cif_name, cif=entry['structure'])
  
    # Save original graph to file.
    original_graph_name = os.path.join(out_graph_dir, 'original_' + entry['material_id'] + '.json')
    original_graph = entry['graph']
    save_graph(original_graph_name, graph=original_graph)

    # Save normal graph that should be basically the same like the original graph.
    normal_graph_name = os.path.join(out_graph_dir, 'normal_' + entry['material_id'] + '.json')
    normal_graph = get_graph_from_structure(cif_name, disordered=False)
    save_graph(normal_graph_name, graph=normal_graph)

    # Save graph with dictionaries of elements as atom features to get the elemental embedding.
    disordered_graph_name = os.path.join(out_graph_dir, 'disordered_' + entry['material_id'] + '.json')
    disordered_graph = get_graph_from_structure(cif_name, disordered=True)
    save_graph(disordered_graph_name, graph=disordered_graph)

    assert normal_graph['bond'] == disordered_graph['bond']
    assert normal_graph['index1'] == disordered_graph['index1']
    assert normal_graph['index2'] == disordered_graph['index2']
    assert normal_graph['state'] == disordered_graph['state']
    assert normal_graph['state'] == original_graph['state']    
    
    # Store tabular data like targets, cif paths, graphs etc.
    result = {
                 'band_gap': entry['band_gap'],
                 'formation_energy_per_atom': entry['formation_energy_per_atom'],
                 'material_id': entry['material_id'],
                 'cif': cif_name,
                 'original_graph': original_graph_name,
                 'normal_graph': normal_graph_name,
                 'disordered_graph': disordered_graph_name
                 }
    return result




if __name__ == '__main__':
        
    # Read in json file of all structures.
    with open(in_json_MP_data) as f:
        data = json.load(f)
    print('Finished reading in json file')
    
    # Make new directory for cifs. If directory already exists delete it first.
    abs_out_cif_dir = projectpath(out_cif_dir)
    if not os.path.exists(abs_out_cif_dir):
        print(f'Create directory {out_cif_dir}')
        os.mkdir(abs_out_cif_dir)
    else:
        print(f'Deleting directory {abs_out_cif_dir} before recreating it.')
        shutil.rmtree(abs_out_cif_dir)
        os.mkdir(abs_out_cif_dir)
    # Make new directory for graphs. If directory already exists delete it first.
    abs_out_graph_dir = projectpath(out_graph_dir)
    if not os.path.exists(abs_out_graph_dir):
        print(f'Create directory {out_graph_dir}')
        os.mkdir(abs_out_graph_dir)
    else:
        print(f'Deleting directory {abs_out_graph_dir} before recreating it.')
        shutil.rmtree(abs_out_graph_dir)
        os.mkdir(abs_out_graph_dir)
    
    # Save as individual structures and graphs and one csv with all targets and paths to the structures and graphs.
    with Parallel(n_jobs=n_cpus, verbose=1) as parallel:
        df = parallel(delayed(save_data_from_entry)(entry) for entry in data)
        
    # Save new data.
    df = pd.DataFrame(df)
    write_to_csv(df, out_csv, comment_csv)
    print('All done!')
    
    
        
        