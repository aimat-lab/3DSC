#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:39:17 2021

@author: Timo Sommer

This script reads in the json data provided by the MEGNet people and outputs a directory with the cifs and graphs and a df with the targets and the paths of the cifs and graphs. The graphs will be built exactly as for the superconductors.
"""

n_cpus = 1          # change this




# Parse input and overwrite.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
args = parser.parse_args()
n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus



from superconductors_3D.utils.projectpaths import projectpath
import os

# Downloaded from the MEGNet paper
in_json_MP_data = projectpath('test_MEGNet', 'Data', 'Materials_Project', 'mp.2018.6.1.json')

out_graph_dir = os.path.join('data', 'transfer_learning', 'MP', 'graphs')
out_csv = projectpath('data', 'transfer_learning', 'MP', 'all_crystals.csv')
comment_csv = 'All crystal structures that were used in the MEGNet paper with graphs built exactly as for the superconductors.'


import json
import pandas as pd
from tqdm import tqdm
import os
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
import joblib
import shutil
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from megnet.data.crystal import CrystalGraph, CrystalGraphDisordered
from megnet.models import MEGNetModel
from joblib import Parallel, delayed
from superconductors_3D.machine_learning.own_libraries.models.GNN.GNN_utils import get_graph_from_structure, graph_features
from superconductors_3D.dataset_preparation.utils.crystal_utils import point_group_from_space_group, One_Hot_crystal_system_and_point_group, One_Hot_bravais_centring, state_features
from superconductors_3D.dataset_preparation.utils.check_dataset import normalise_pymatgen_spacegroups


        
def save_graph(graph_name, graph):        
    abs_graph_name = projectpath(graph_name)
    
    with open(abs_graph_name, 'w') as graph_file:
        json.dump(graph, graph_file)            

def save_data_from_entry(entry):
    structure = Structure.from_str(entry['structure'], fmt='cif')
    
    # Get spacegroup, crystal system and point group
    symmetries = SpacegroupAnalyzer(structure, symprec=0.1)
    spacegroup = symmetries.get_space_group_symbol()
    spacegroup = normalise_pymatgen_spacegroups(spacegroup)
    crystal_system = symmetries.get_crystal_system()
    point_group = point_group_from_space_group(spacegroup)
    assert all([isinstance(val, str) for val in (spacegroup, crystal_system, point_group)]), f'spacegroup: {spacegroup}, crystal_system: {crystal_system}, point_group: {point_group}'
    
    # Get one hot encoding of crystal system, point group and the bravais centring.
    One_Hot_crys_sys = One_Hot_crystal_system_and_point_group(crystal_system, point_group)
    One_Hot_bravais = One_Hot_bravais_centring(spacegroup)
    
    state_feature_dict = {'crystal_temp_2': 0}    
    state_feature_dict.update(One_Hot_crys_sys.to_dict())
    state_feature_dict.update(One_Hot_bravais.to_dict())
    state_values = [state_feature_dict[feat] for feat in state_features]
  
    # Save graph with dictionaries of elements as atom features to get the elemental embedding.
    graph_name = os.path.join(out_graph_dir, entry['material_id'] + '.json')
    graph = get_graph_from_structure(structure, add_state_values=state_values, model=model, use_embedding=True)
    save_graph(graph_name, graph=graph)
    
    # Store tabular data like targets, cif paths, graphs etc.
    result = {
                 'band_gap': entry['band_gap'],
                 'formation_energy_per_atom': entry['formation_energy_per_atom'],
                 'material_id': entry['material_id'],
                 'graph': graph_name
                 }
    result.update(state_feature_dict)
    return result




if __name__ == '__main__':
        
    # Read in json file of all structures.
    with open(in_json_MP_data) as f:
        data = json.load(f)
    print('Finished reading in json file')
    
    # Make new directory for graphs. If directory already exists delete it first.
    abs_out_graph_dir = projectpath(out_graph_dir)
    if not os.path.exists(abs_out_graph_dir):
        print(f'Create directory {out_graph_dir}')
        os.mkdir(abs_out_graph_dir)
    else:
        print(f'Deleting directory {abs_out_graph_dir} before recreating it.')
        shutil.rmtree(abs_out_graph_dir)
        os.mkdir(abs_out_graph_dir)
    
    # Setup MEGNet model for the conversion of structures, it saves a lot of time doing this only once.
    nfeat_bond=100
    r_cutoff=4
    gaussian_width=0.5
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)    
    graph_converter = CrystalGraphDisordered(cutoff=r_cutoff)
    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
    
    # Save as individual structures and graphs and one csv with all targets and paths to the structures and graphs.
    with Parallel(n_jobs=n_cpus, verbose=1) as parallel:
        df = parallel(delayed(save_data_from_entry)(entry) for entry in data)
        
    # Save new data.
    df = pd.DataFrame(df)
    write_to_csv(df, out_csv, comment_csv)
    print('All done!')
    
    
        
        