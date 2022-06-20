#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:40:06 2021

@author: timo

This script cleans the SuperCon database by normalizing the chemical formula, taking the median tc for duplicate chemical formulas and some other small stuff.
"""
from superconductors_3D.utils.projectpaths import projectpath
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

def check_quantities(formula):
    """Checks if the quantity of an element in a chemical formula is unrealistically high.
    """
    max_quantity = 150
    chemdict = get_chem_dict(formula)
    for el in chemdict.keys():
        val = chemdict[el]
        if val > max_quantity:
            # print("Quantity too high, therefore excluded: ", formula)
            return(False)
    return(True)

def clean_SuperCon(in_filename, out_filename, out_excluded_filename, comment, comment_excluded, sc_frac=1):

    print("Read in csv.")
    df = pd.read_csv(in_filename)
    
    if sc_frac != 1:
        print(f'Downsampling SuperCon for debugging to a fraction of {sc_frac}.')
        df = df.sample(frac=sc_frac)
    
    df["origin"] = "Supercon"
    
    df = df.rename(columns={'Tc': 'tc'})
    
    # Will contain all data rows that for some reason had to be filtered out.
    df_excluded = pd.DataFrame()
    
    df_correct = copy.deepcopy(df)
    
    # Standardise the chemical formula string
    print("Standardise the chemical formula string.")
    df_correct["formula"] = df_correct["name"].apply(standardise_chem_formula)
    
    # Filter entries with non valid formula
    print("Checking chemical formula.")
    excl_condition = ~ df_correct["formula"].map(if_valid_formula)
    reason = "Invalid chemical formula"
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
    
    # Add column with normalised values of formula.
    df_correct["norm_formula"] = df_correct["formula"].apply(standardise_chem_formula, args=(True,)) 
    
    # Check if all values in a column have the same dtype
    df_correct = df_correct.apply(set_column_type, axis=0)
    
    # Get chemical system for later quick check if formulas can be similar at all.
    df_correct["chemical_composition"] = df_correct["formula"].apply(get_chemical_composition) 
    
    # Filter out chemical formulas with too large quantities.
    excl_condition = ~ df_correct["formula"].apply(check_quantities)
    reason = "Quantity too high"
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
    
    
    # Check if all numerical columns really only contain floats.
    numerical_columns = ["tc"]
    for col in numerical_columns:
        df_correct.loc[:, col] = df_correct[col].map(check_float)
    
    
    # Exclude unrealistic high Tc.
    max_tc = 145
    excl_condition = (df_correct["tc"] > max_tc)
    reason = f"Tc > {max_tc} K"
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
    
    assert not any(df_correct['tc'].isna())
    
    
    # # Make new column with median tc of all entries with same chemical formula and rename this to tc and the old tc to tc_old.
    print("Get median of tc.")
    df_correct = df_correct.join(df_correct.groupby("norm_formula")["tc"].mean(), on="norm_formula", rsuffix="_mean")
    df_correct = df_correct.join(df_correct.groupby("norm_formula")["tc"].std(), on="norm_formula", rsuffix="_std")
    df_correct = df_correct.rename({"tc": "tc_old", "tc_mean": "tc"}, axis="columns")
    # Exclude duplicates and exclude whole datapoint if std of duplicates is too high. Done exactly like in 2018 Stanev.
    excl_condition = df_correct.duplicated(subset=['norm_formula'])
    reason = 'Duplicate norm_formula'
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
    max_std_tc = 5
    excl_condition = df_correct['tc_std'] > max_std_tc
    reason = f'Std of Tc > {max_std_tc}'
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
    df_correct = df_correct.drop(columns='tc_std')
    
    
    # Get column with number of elements for convenience.
    df_correct["num_elements"] = df_correct["formula"].apply(lambda x: len(get_chem_dict(x)))


    # Assign superconductors to rough classes.
    print("Assign classes to superconductors.")
    df_correct[["sc_class", "good", "strict"]] = df_correct.apply(lambda row: assign_rough_class(row["formula"]), axis=1)
    df_correct = movecol(df_correct, ["sc_class", "good", "strict"], "formula")
    
    # Make column if data point belongs to unique class.
    classes = ['Ferrite', 'Cuprate', 'Other', 'Oxide', 'Chevrel', 'Heavy_fermion',
       'Carbon']
    df_correct['sc_class_unique'] = df_correct['sc_class'].isin(classes)
    
    
    keep_columns = ['formula', 'tc', 'sc_class', 'sc_class_unique', 'norm_formula', 'chemical_composition', 'num_elements', 'origin', 'name']
    df_correct = df_correct[keep_columns]
    df_correct = prepare_df_sc(df_correct, rename_sc={'name': 'old_formula'})
    
    
    # Export data to csv.
    write_to_csv(df_correct, out_filename, comment)
    write_to_csv(df_excluded, out_excluded_filename, comment_excluded)
    print("Done! {} datapoints were saved in the cleaned file. {} datapoints were excluded and saved in another file.".format(len(df_correct), len(df_excluded)))

if __name__ == '__main__':
    
    in_filename = projectpath('data', 'source', 'SuperCon', 'raw', 'Supercon_data_by_2018_Stanev.csv')
    out_filename = projectpath('data', 'source', 'SuperCon', 'cleaned', '2.0_all_data_SuperCon_cleaned.csv')
    out_excluded_filename = projectpath('data', 'source', 'SuperCon', 'cleaned', 'excluded_2.0_all_data_SuperCon_cleaned.csv')    
    comment = f'All the cleaned data from the Supercon main table {in_filename}.'
    comment_excluded = f'All the data that was not included in {out_filename} because it was filtered out.'

    clean_SuperCon(in_filename, out_filename, out_excluded_filename, comment, comment_excluded)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
