#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:17:49 2021

@author: Timo Sommer

This script finds out how many structures in a given cif database have different occupancies of the atomic sites of one element. It finds the answers to the following questions:
    - If given a structure and I want to replace some of element 1 in this structure with element 2. Can I simply replace all of the atom sites of element 1 with the corresponding fraction of element 2?
    --> In the original ICSD database, how many of the doped element sites (with several symmetrically non equivalent atom sites) are doped so that several symmetrically not equivalent sites are all doped?
        - Are they all doped with the same fraction?
    - If a structure has two 
"""

# Script abandoned because I think I don't need it anymore.



dataset_csv = '/home/timo/Masterarbeit/Rechnungen/Datasets/cif_files/ICSD/All_data_ICSD_pymatgen_cifs_normalised_cleaned.csv'
cif_col = 'cif_2'
cif_dir = '/home/timo/Masterarbeit/Rechnungen/Datasets/cif_files/ICSD/pymatgen_cleaned_cifs'

save_doped_comps = '/home/timo/Masterarbeit/Analysen/Dataset_preparation/occupancies_of_elements_in_ICSD/doped_el_compositions.csv'
comment_doped_comps = 'A dataframe with the compositions of symmetrically equivalent sites with elements that are doped somewhere in the structure. Useful for plotting etc to find out how to manipulate cif structures with additional doping atoms.'

# If all doping occupancies should be calculated again. Otherwise existing csv will be used for the plots.
rerun_all_structures = True    



import pandas as pd
import os
import pickle
import ast
import numpy as np
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from Own_functions import write_to_csv

def element_to_string(comps):
    """Makes pymatgen element object in comps to their string."""
    new_comps = []
    for comp in comps:
        new_comp = {el.name: val for el, val in comp.items()}
        new_comps.append(new_comp)
    return(new_comps)

def flatten_elements_list(el_list):
    flattened_list = [el_comp.chemical_system.split('-') for el_comp in el_list]
    flattened_list = [el for els in flattened_list for el in els]
    flattened_list = list(pd.unique(flattened_list))
    return(flattened_list)

def comps_of_doped_elements(symm_struct):
    symm_sites = symm_struct.equivalent_sites
    symm_elements = [sites[0].species.element_composition for sites in symm_sites]
    doped_elements = [el_comp for el_comp in symm_elements if len(el_comp.as_dict()) > 1]        
    all_doped_elements = flatten_elements_list(doped_elements)
    symm_comps_with_doped_elements = [dict(comp) for comp in symm_elements if any([el in comp for el in all_doped_elements])]
    return(symm_comps_with_doped_elements)

def get_doped_occupancies(dataset_csv):
    df = pd.read_csv(dataset_csv, header=1)
    n_crystals = len(df)
    datadict = {}
    n_failed = 0
    for i, row in df.iterrows():     
        cif_name = row[cif_col]
        cif_path = os.path.join(cif_dir, cif_name)
        struct = CifParser(cif_path).get_structures()[0]
        try:
            symm_analyzer = SpacegroupAnalyzer(struct)
            symm_struct = symm_analyzer.get_symmetrized_structure()
        except TypeError:
            n_failed += 1
            continue
        doped_comps = comps_of_doped_elements(symm_struct)
        
        datadict[i] = {'formula': struct.formula,
                       'doped_comps': doped_comps,
                       'cif': cif_path
                       }
        i += 1
        if i % 1000 == 0:
            print(f'Done: {i} from {n_crystals}')
    print(f'Done: {i} from {n_crystals}')
    df_doped = pd.DataFrame(datadict).T
    return(df_doped)

if __name__ == '__main__':
    
    if rerun_all_structures:
        df_doped = get_doped_occupancies(dataset_csv)
        df_doped.to_pickle(save_doped_comps)
        # write_to_csv(df_doped, save_doped_comps, comment_doped_comps)
    else:
        df_doped = pd.read_pickle(save_doped_comps)
    
# =============================================================================
#                           PLOTS
# =============================================================================
    
    def remove_single_doped_sites(comps):
        if comps == []:
            return([])
        
        all_els = []
        for comp in comps:
            all_els.extend(list(comp.keys()))
        all_els = list(pd.unique(all_els))
        counts = {}
        for el in all_els:
            counts[el] = sum([el in comp for comp in comps])
        pass
        

    # df_doped['doped_comps'] = df_doped['doped_comps'].apply(element_to_string)
    # df_doped1 = df_doped['doped_comps'].apply(remove_single_doped_sites)
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    