#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:30:59 2022

@author: Timo Sommer

This script reduces the whole downloaded MP data to the needed cifs for the 5773 3DSC_MP entries plus 1000 cifs more to demonstrate the cleaning and sorting algorithm.
"""
import pandas as pd
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
from superconductors_3D.utils.projectpaths import projectpath, Data_Paths
import os 
import shutil




if __name__ == '__main__':
    
    database = 'MP'
    n_random_additional_cifs = 1000

    paths = Data_Paths(database)
    
    raw_data_dir = projectpath('data', 'source', database, 'raw')
    raw_df_path = os.path.join(raw_data_dir, f'0_all_data_{database}.csv')
    raw_cif_dir = os.path.join(raw_data_dir, 'all_cifs')
    
    needed_raw_df_path = paths.crystal_db_csv
    needed_raw_cif_dir = paths.crystal_db_cifs_dir
    
    final_df_path = paths.final_3DSC
    
    df_raw = pd.read_csv(raw_df_path, header=1)
    df_final = pd.read_csv(final_df_path, header=1)
    
    needed_materials = df_final['material_id_2'].unique()
    is_needed = df_raw['material_id'].isin(needed_materials)
    df_raw_needed = df_raw[is_needed]
    
    df_random_more_materials = df_raw[~is_needed].sample(frac=1).head(n_random_additional_cifs)
    df_raw_paper = pd.concat((df_raw_needed, df_random_more_materials))
    
    # Save needed df
    comment = 'A subset of the MP database, reduced to the needed entries plus 1000 additional entries to demonstrate the algorithm.'
    write_to_csv(df_raw_paper, needed_raw_df_path, comment)
    # Save needed cifs
    for mat_id in df_raw_paper['material_id']:
        cif_name = f'{mat_id}.cif'
        origin_cif_path = os.path.join(raw_cif_dir, cif_name)
        needed_cif_path = os.path.join(needed_raw_cif_dir, cif_name)
        shutil.copyfile(src=origin_cif_path, dst=needed_cif_path)
        
    
    
