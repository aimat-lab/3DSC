#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:49:50 2021

@author: Timo Sommer

This script sets up the original SuperCon Data from Stanev with MAGPIE features and kmeans clusters. 
"""
from superconductors_3D.utils.projectpaths import projectpath

input_sc_data = projectpath('data', 'source', 'SuperCon', 'cleaned', '2.0_all_data_SuperCon_cleaned.csv')

output_sc_data = projectpath('data', 'final', 'SuperCon', 'SC_original_MAGPIE.csv')
comment = 'Dataset of original SuperCon data with MAGPIE features and CV columns.'

import pandas as pd
from superconductors_3D.dataset_preparation.utils.lib_generate_datasets import MAGPIE_features
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, movecol
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans


rename_columns = {
                    'tc_sc': 'tc',
                    'sc_class_sc': 'sc_class'
                    }

if __name__ == '__main__':
    
    df = pd.read_csv(input_sc_data, header=1)
    assert not any(df.duplicated('formula_sc'))
        
    # Get MAGPIE features.
    formulas = df['formula_sc']
    magpie_features = MAGPIE_features(formulas)
    magpie_names = magpie_features.columns
    magpie_names = [f'MAGPIE_{name}' for name in magpie_names]
    df[magpie_names] = magpie_features.to_numpy()
    
    # cluster by MAGPIE features with k=5 groups
    df['5-kmeans-magpie-cluster'] = KMeans(n_clusters=5).fit_predict(df[magpie_names])
    
    # Add pseudo sample weight column for easier algorithms.
    df['weight'] = 1
    
    assert df[magpie_names].notna().all().all()
    
    df = df.rename(columns=rename_columns)
    
    write_to_csv(df, output_sc_data, comment)
