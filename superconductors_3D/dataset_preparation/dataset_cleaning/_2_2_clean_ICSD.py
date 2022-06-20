#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:55:21 2021

@author: timo
This script is for cleaning the data csv from the ICSD database. The output is one csv file with the cleaned data and one csv file with the excluded data.
"""
from superconductors_3D.utils.projectpaths import projectpath







# =============================================================================
#                                   MAIN
# =============================================================================

from datetime import datetime
import os
import pandas as pd
import numpy as np
import gemmi
import re

import copy
from superconductors_3D.dataset_preparation.utils.check_dataset import if_valid_formula, set_column_type, correct_symmetry_cell_setting, extract_float_values, standardise_chem_formula, get_normalised_spg, check_and_complement_structure, get_chemical_composition, filter_entries, prepare_df_2
from superconductors_3D.dataset_preparation.utils.normalize_dataset_columns import rename_to_COD, rename_to_Sc
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, movecol




def normalise_spacegroups(spg_string):
    """Tries to recognise spacegroup and to return the Hermann Maguin form. If it doesn't recognise the spacegroup it sets it "".
    """
    spg = gemmi.find_spacegroup_by_name(spg_string)
    if spg != None:
        spg_string = get_normalised_spg(spg).hm
        return(spg_string)
    spg_string0 = spg_string
    if spg_string == "":
        return(spg_string)
    pattern = "[0-9]\([0-9]\)"
    if re.search(pattern, spg_string) != None:
        spg_string = re.sub("[\(\)]", "", spg_string)
    pattern1 = "[\(\{][^\)]*[\)\}]"
    if re.search(pattern1, spg_string) != None:
        spg_string = re.sub(pattern1, "", spg_string)
    pattern2 = "^:[A-Z]$"
    if re.search(pattern2, spg_string) != None:
        spg_string = re.sub(pattern2, "", spg_string)
    # if spg_string != spg_string0:
    #     print("String korrigiert: {} --> {}".format(spg_string0, spg_string))
    spg = gemmi.find_spacegroup_by_name(spg_string)
    if spg == None:
        # print("Unknown spacegroup found: {}".format(spg_string))
        spg_string = ""
        return(spg_string)
    else:
        spg_string = get_normalised_spg(spg).hm
        return(spg_string)

def clean_ICSD(in_filename, in_type_filename, out_filename, out_excluded_filename, comment, comment_excluded):
    
    important_cols = ['_database_code_icsd', '_chemical_formula_sum', 'formula_pymatgen', '_cell_measurement_temperature', '_diffrn_ambient_temperature', 'spacegroup_pymatgen',
       'crystal_system_pymatgen', 'lata_pymatgen', 'latb_pymatgen',
       'latc_pymatgen',  'cif_pymatgen_path',
       '_chemical_name_structure_type', '_exptl_crystal_density_diffrn',
       '_chemical_formula_weight', '_cell_length_a', '_cell_length_b',
       '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta',
       '_cell_angle_gamma', '_cell_volume', '_cell_formula_units_z',
       '_symmetry_space_group_name_H-M', '_space_group_IT_number',
       '_diffrn_ambient_pressure', 'file_id', 'database_id', 'cif',
       'valid_cif']

    print("Read in file.")
    df = pd.read_csv(in_filename, header=1, na_values="?")
    df = df.rename(columns=rename_to_COD)
    df = df[important_cols]
    df_excluded = pd.DataFrame()     
    
    # Add origin of data for later convenience.
    df["origin"] = "ICSD"

    # Little sanity check.
    assert all(df["valid_cif"])
    df = df.drop(columns=["valid_cif"])

    # Filter invalid chemical pymatgen formulae.
    print("Checking chemical formula (pymatgen).")
    excl_condition = ~ df["formula_pymatgen"].map(if_valid_formula)
    reason = "Invalid chemical formula (pymatgen)"
    df_correct, df_excluded = filter_entries(excl_condition, reason, df, df_excluded)
    
    # Filter invalid chemical COD formulae.
    print("Checking chemical formula (ICSD).")
    excl_condition = ~ df_correct["_chemical_formula_sum"].map(if_valid_formula)
    reason = "Invalid chemical formula (ICSD)"
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)

    # Standardise pymatgen formula and check if it is still valid.
    print("Standardise chemical formula.")
    df_correct["formula_pymatgen"] = df_correct["formula_pymatgen"].apply(standardise_chem_formula)
    assert all(df_correct["formula_pymatgen"].apply(if_valid_formula)), "Standardising broke something."
    
    # Get chemical composition (only elements, without quantities).
    df_correct["chemical_composition"] = df_correct["formula_pymatgen"].apply(get_chemical_composition)
    
    chem_comp_original = df_correct["_chemical_formula_sum"].apply(get_chemical_composition)
    excl_condition = chem_comp_original != df_correct["chemical_composition"]
    reason = "Chemical composition of original and pymatgen formula different."
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
    
    # Check if all values in a column have the same dtype
    df_correct = df_correct.apply(set_column_type, axis=0)
    
    # Check and correct spacegroup.
    print("Correct spacegroup.")
    df_correct.loc[:, "_symmetry_space_group_name_H-M"] = df_correct["_symmetry_space_group_name_H-M"].map(normalise_spacegroups)
    
    # Add column for crystal system.
    if not '_symmetry_cell_setting' in df_correct.columns:
        df_correct['_symmetry_cell_setting'] = ''

    # Check spacegroup and crystal system for consistency and complement crystal system if possible.
    print("Check and correct structure spacegroup and crystal system.")
    df_correct = check_and_complement_structure(
                                                df = df_correct,
                                                sp_name = "_symmetry_space_group_name_H-M",
                                                crys_name = "_symmetry_cell_setting",
                                                num_name = "_space_group_IT_number"
                                                )
    
    # Filter out entries without spacegroup.
    excl_condition = df_correct["_symmetry_space_group_name_H-M"] == ""
    reason = "Empty spacegroup"
    df_correct, df_excluded = filter_entries(excl_condition, reason, df_correct, df_excluded)
        
    # Check and correct important numerical columns.
    print("Clean numerical columns by uncertainty.")
    numeric_cols = ["_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_volume", "_cell_measurement_temperature", '_diffrn_ambient_temperature', '_diffrn_ambient_pressure', '_exptl_crystal_density_diffrn', '_chemical_formula_weight']
    for colname in numeric_cols:    
        df_correct[colname] = df_correct[colname].apply(extract_float_values, args=(True,))
            
    # Get column with normalised quantities.
    df_correct["norm_formula"] = df_correct["formula_pymatgen"].apply(standardise_chem_formula, args=(True,))
    
    # Import info if crystal structure is experimental or theoretical.
    df_type = pd.read_csv(in_type_filename, header=1)
    df_correct['file_id'] = df_correct['file_id'].str.split('/').str[-1].str.split('.cif').str[0].astype(int)
    is_experimental = df_correct['file_id'].isin(df_type['EXPERIMENTAL_INORGANIC']) | df_correct['file_id'].isin(df_type['EXPERIMENTAL_METALORGANIC'])
    is_theoretical = df_correct['file_id'].isin(df_type['THERORETICAL_STRUCTURES'])
    assert all(is_experimental | is_theoretical) and not any(is_experimental & is_theoretical)
    df_correct.loc[is_experimental, 'type'] = 'experimental'
    df_correct.loc[is_theoretical, 'type'] = 'theoretical'
    
    # Make temperature column consistent.
    # Unify both columns.
    df_correct.loc[df_correct['_cell_measurement_temperature'].isna(), '_cell_measurement_temperature'] = df_correct['_diffrn_ambient_temperature']
    assert df_correct['_cell_measurement_temperature'].equals(df_correct['_diffrn_ambient_temperature'])
    df_correct = df_correct.drop(columns='_diffrn_ambient_temperature')
    # If no value given assume room temperature.
    room_temperature = 293
    df_correct['no_crystal_temp_given'] = df_correct['_cell_measurement_temperature'].isna()    # For backtracking
    df_correct = movecol(df_correct, ['no_crystal_temp_given'], '_cell_measurement_temperature')
    default_room_temp = df_correct['no_crystal_temp_given']
    df_correct.loc[default_room_temp, '_cell_measurement_temperature'] = room_temperature
    
    # Exclude theoretical structures because we heavily need the cell measurement temperature.
    is_theoretical = df_correct['type'] == 'theoretical'
    reason = 'is theoretical'
    df_correct, df_excluded = filter_entries(is_theoretical, reason, df_correct, df_excluded)
    
    # Check if all recognized spacegroups are the same as in the original cif file.
    print("Check if all recognized spacegroups are the same as in the original cif file.")
    no_pymatgen_spacegroup = df_correct["spacegroup_pymatgen"] == ''
    df_correct.loc[no_pymatgen_spacegroup, "spacegroup_pymatgen"] = df_correct.loc[no_pymatgen_spacegroup, "_symmetry_space_group_name_H-M"]
    spacegroup_different = df_correct["spacegroup_pymatgen"] != df_correct["_symmetry_space_group_name_H-M"]
    reason = "Inconsistent spacegroup"
    df_correct, df_excluded = filter_entries(spacegroup_different, reason, df_correct, df_excluded)

    # Rename columns to make it more standardised like the Supercon and MP dataset.
    df_correct = prepare_df_2(df_correct, rename_to_Sc)    
        
    # Write data and excluded data to corresponding csv files.
    write_to_csv(df_correct, out_filename, comment)
    write_to_csv(df_excluded, out_excluded_filename, comment_excluded)
    print("Done! {} datapoints were saved in the cleaned file. {} datapoints were excluded and saved in another file.".format(len(df_correct), len(df_excluded)))
    
    
    
    
if __name__ == '__main__':
    
    in_filename = projectpath('data', 'source', 'ICSD', 'cleaned', '1_all_data_ICSD_cifs_normalized.csv')
    out_filename = projectpath('data', 'source', 'ICSD', 'cleaned', '2.1_all_data_ICSD_cleaned.csv')
    out_excluded_filename = projectpath('data', 'source', 'ICSD', 'cleaned', 'excluded_2.1_all_data_ICSD_cleaned.csv')
    in_type_filename = projectpath('data', 'source', 'ICSD', 'raw', 'ICSD_content_type.csv')    
    comment = f'All the data from {in_filename}, but the important columns are cleaned.'
    comment_excluded = f'All the data that was not included in {out_filename} because it was filtered out.'
        
    clean_ICSD(in_filename, in_type_filename, out_filename, out_excluded_filename, comment, comment_excluded)
    
    
    
    
    
    
    
    