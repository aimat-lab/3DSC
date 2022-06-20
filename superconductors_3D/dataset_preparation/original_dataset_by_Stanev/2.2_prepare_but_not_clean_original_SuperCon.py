#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:40:06 2021

@author: timo

This script cleans the SuperCon database by normalizing the chemical formula, taking the median tc for duplicate chemical formulas and some other small stuff.
"""
from superconductors_3D.utils.projectpaths import projectpath

in_filename = projectpath('data', 'source', 'SuperCon', 'raw', 'Supercon_data_by_2018_Stanev.csv')

out_filename = projectpath('data', 'source', 'SuperCon', 'cleaned', '2.2_all_original_data_SuperCon_cleaned.csv')

comment = f'All the data from Stanev 2017 without excluding any data.'


# =============================================================================
#                                   MAIN
# =============================================================================


import pandas as pd
pd.options.mode.chained_assignment = None
import re
import pymatgen
import copy
import numpy as np
import gemmi
import warnings
from superconductors_3D.dataset_preparation.utils.group_sc import assign_rough_class
from superconductors_3D.dataset_preparation.utils.check_dataset import if_valid_formula, set_column_type, correct_symmetry_cell_setting, extract_float_values, standardise_chem_formula, get_chem_dict, consistent_spacegroup_and_crystal_system, get_normalised_spg, check_and_complement_structure, filter_entries, get_chemical_composition, normalise_pymatgen_spacegroups, normalise_AFLOW_prototype_spacegroups, normalise_quantities, prepare_df_sc
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, insert_element, movecol, frequentvals, diff_rows


def check_float(val):
    if val == "":
        val = np.nan
        return(val)
    try:
        val = float(val)
        return(val)
    except ValueError:
        raise Warning("Invalid float encountered: {}".format(val))


if __name__ == '__main__':
    
    print("Read in csv.")
    df = pd.read_csv(in_filename)
    df["origin"] = "Supercon"
    
    df = df.rename(columns={'Tc': 'tc'})
    
    df_correct = copy.deepcopy(df)
    
    # Standardise the chemical formula string
    print("Standardise the chemical formula string.")
    df_correct["formula"] = df_correct["name"].apply(standardise_chem_formula)
    
    # Add column with normalised values of formula.
    # df_correct["norm_formula"] = df_correct["formula"].apply(standardise_chem_formula, args=(True,)) 
    
    # Check if all values in a column have the same dtype
    df_correct = df_correct.apply(set_column_type, axis=0)
    
    # Get chemical system for later quick check if formulas can be similar at all.
    df_correct["chemical_composition"] = df_correct["formula"].apply(get_chemical_composition) 
    
    # Check if all numerical columns really only contain floats.
    numerical_columns = ["tc"]
    for col in numerical_columns:
        df_correct.loc[:, col] = df_correct[col].map(check_float)
    
    assert not any(df_correct['tc'].isna())    
    
    # Get column with number of elements for convenience.
    df_correct["num_elements"] = df_correct["formula"].apply(lambda x: len(get_chem_dict(x)))

    # Assign superconductors to rough classes.
    print("Assign classes to superconductors.")
    # df_correct[["sc_class", "good", "strict"]] = df_correct.apply(lambda row: assign_rough_class(row["formula"]), axis=1)
    # df_correct = movecol(df_correct, ["sc_class", "good", "strict"], "formula")
    
    # Make column if data point belongs to unique class.
    # classes = ['Ferrite', 'Cuprate', 'Other', 'Oxide', 'Chevrel', 'Heavy_fermion',
    #    'Carbon']
    # df_correct['sc_class_unique'] = df_correct['sc_class'].isin(classes)
    
    
    keep_columns = ['formula', 'tc', 'chemical_composition', 'num_elements', 'origin', 'name']
    df_correct = df_correct[keep_columns]
    df_correct = prepare_df_sc(df_correct, rename_sc={'name': 'old_formula'})
    
    
    # Export data to csv.
    write_to_csv(df_correct, out_filename, comment)
    print("Done! {} datapoints were saved in the cleaned file.".format(len(df_correct)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
