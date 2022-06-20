#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:56:06 2021

@author: Timo Sommer

This script extracts the cifs from the Materials Project dataframe and writes them to an own directory to seperate cif and dataframe. The path to the cif file is written into the same column `cif` in which the whole cif string was in before.
"""
from superconductors_3D.utils.projectpaths import projectpath
import pandas as pd
import os
from tqdm import tqdm
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv







def extract_cifs(in_filename, out_filename, out_cifs_dir, comment):
    df = pd.read_pickle(in_filename)
    cifs = df['cif']
    mp_ids = df['material_id']
    
    cif_paths = []
    for mp_id, cif in tqdm(zip(mp_ids, cifs)):
        
        # Define name and save_path of cif.
        cif_name = f'{mp_id}.cif'
        rel_cif_path = os.path.join(out_cifs_dir, cif_name)
        
        # Save cif.
        abs_cif_path = projectpath(rel_cif_path)
        assert not os.path.exists(abs_cif_path), 'You are trying to overwrite cif files. This is for safety not supported. If you want to do a new run, please manually delete the cif files beforehand.'
        assert os.path.exists(projectpath(out_cifs_dir)), 'The specified directory for saving cif files doesn\'t exist. Please make the directory.'
        with open(abs_cif_path, 'w') as f:
            f.write(cif)    
        
        cif_paths.append(rel_cif_path)
    
    # Overwrite cif column to include the path to the cif file instead of the cif file itself.
    df['cif'] = cif_paths
    # Add normalized column like all other crystal datasets.
    df['database_id'] = 'MP-' + df['material_id'].astype(str)
    
    write_to_csv(df, out_filename, comment)
    
    

if __name__ == '__main__':
    
    in_filename = projectpath('data', 'source', 'MP', 'raw', 'materials_project.pkl')
    out_filename = projectpath('data', 'source', 'MP', 'raw', '0_all_data_MP.csv')
    out_cifs_dir = os.path.join('data', 'source', 'MP', 'raw', 'cifs')
    
    comment = f'Same as in {in_filename} but the cif strings that were part of the df are now saved under the path that is now in the column cif.'
    
    extract_cifs(in_filename, out_filename, out_cifs_dir, comment)

    
    
            
            
        
        
