#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:09:29 2021

@author: timo
This script cleans the Materials Project database that was downloaded. 
"""
from superconductors_3D.utils.projectpaths import projectpath
import pandas as pd
from superconductors_3D.dataset_preparation.utils.check_dataset import if_valid_formula, set_column_type, standardise_chem_formula, get_chemical_composition, prepare_df_2, filter_entries
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, movecol
from superconductors_3D.dataset_preparation.utils.normalize_dataset_columns import rename_to_Sc, rename_MP


def clean_MP(in_filename, out_filename, out_excluded_filename, comment, comment_excluded):
    
    important_cols = ['database_id', 'full_formula', 'formula_pymatgen', 'spacegroup_pymatgen', 'crystal_system_pymatgen',
       'lata_pymatgen', 'latb_pymatgen', 'latc_pymatgen',
       'cif_pymatgen_path', 'cif', 'material_id', 'band_gap', 'band_structure',
       'created_at', 'density', 'doi', 'doi_bibtex', 'dos', 'e_above_hull', 'efermi', 'encut', 'energy', 'energy_per_atom', 'exp', 'final_energy',
       'final_energy_per_atom', 'formation_energy_per_atom',
        'has', 'has_bandstructure',
       'icsd_ids', 'is_ordered', 'last_updated',
       'magnetic_type', 'nsites', 'ntask_ids',
       'original_task_id', 'oxide_type', 'pretty_formula',
       'pseudo_potential', 'reduced_cell_formula', 'run_type', 'spacegroup',
       'task_id', 'task_ids', 'total_magnetization', 'unit_cell_formula',
       'volume', 'warnings', 'ordering', 'is_magnetic', 'exchange_symmetry',
       'num_unique_magnetic_sites', 'magmoms',
       'total_magnetization_normalized_vol',
       'total_magnetization_normalized_formula_units', 'num_magnetic_sites',
       'true_total_magnetization']
    
    print('Read in file.')
    df = pd.read_csv(in_filename, header=1)
    df = df[important_cols]
    df['origin'] = 'MP'
    
    # Filter invalid chemical formulae.
    print('Checking chemical formula.')
    indices_correct = df['full_formula'].map(if_valid_formula)
    df_correct = df[ indices_correct ]
    df_excluded = df[ ~ indices_correct ]
    reason = 'Reason for exclusion'
    df_excluded[reason] = 'invalid chemical formula'
    # Standardise formula and check if it is still valid.
    print('Standardise chemical formula.')
    df_correct['formula_pymatgen'] = df_correct['formula_pymatgen'].apply(standardise_chem_formula)
    # All formulae should still be valid
    assert all(df_correct['formula_pymatgen'].apply(if_valid_formula)), 'Standardising broke something.'
    
    # Check if all values in a column have the same dtype
    df_correct = df_correct.apply(set_column_type, axis=0)
    
    # Get combination of elements.
    df_correct['chemical_composition'] = df_correct['formula_pymatgen'].apply(get_chemical_composition)
    
    # Get column with normalised quantities.
    df_correct['norm_formula'] = df_correct['formula_pymatgen'].apply(standardise_chem_formula, args=(True,))    
        
    # Rename columns to make it more standardised like the Supercon and COD dataset.
    df_correct = df_correct.drop(columns='spacegroup')
    df_correct = df_correct.rename(columns=rename_MP)
    df_correct = prepare_df_2(df_correct, rename_to_Sc)
    df_correct = movecol(df_correct, cols=['chemical_composition_2', 'norm_formula_2'], to='formula_2')
    
    write_to_csv(df_correct, out_filename, comment)
    write_to_csv(df_excluded, out_excluded_filename, comment_excluded)
    print(f'Done! {len(df_correct)} datapoints were saved in the cleaned file. {len(df_excluded)} datapoints were excluded and saved in another file.')


if __name__ == '__main__':
    
    in_filename = projectpath('data', 'source', 'MP', 'cleaned', '1_all_data_MP_cifs_normalized.csv')
    out_filename = projectpath('data', 'source', 'MP', 'cleaned', '2.1_all_data_MP_cleaned.csv')
    out_excluded_filename = projectpath('data', 'source', 'MP', 'cleaned', 'excluded_2.1_all_data_MP_cleaned.csv')
    comment = 'All the data from {in_filename}, but the important columns are cleaned.'
    comment_excluded = 'All the data that was not included in {out_filename} because it was filtered out.'

    clean_MP(in_filename, out_filename, out_excluded_filename, comment, comment_excluded)
    
    
    
    
    
    
    
