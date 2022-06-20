#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:49:50 2021

@author: Timo Sommer

This script prepares the raw data downloaded from Stanev exactly as the "Regression model in 2017 Stanev, page 7. This means only data with Tc > 10K is used. Otherwise the data is left untouched and MAGPIE features are calculated."
"""
from superconductors_3D.utils.projectpaths import projectpath

input_sc_data = projectpath('data', 'source', 'SuperCon', 'raw', 'Supercon_data_by_2018_Stanev.csv')

output_sc_data = projectpath('data', 'final', 'SuperCon', 'raw_SC_MAGPIE.csv')
comment = 'Dataset of original SuperCon data without cleaning or excluding data points with MAGPIE features.'


import pandas as pd
from superconductors_3D.dataset_preparation.utils.lib_generate_datasets import MAGPIE_features
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
import numpy as np

rename_columns = {
                    'Tc': 'tc',
                    'name': 'formula_sc'
                    }

if __name__ == '__main__':
    
    df = pd.read_csv(input_sc_data)
    df = df.rename(columns=rename_columns)
    
    regression_model = df['tc'] > 10
    df = df[regression_model].reset_index(drop=True)
    print(f'For the regression model {sum(~regression_model)} data points where excluded because they have Tc < 10K.')

    # Get MAGPIE features.
    formulas = df['formula_sc']
    magpie_features = MAGPIE_features(formulas)
    magpie_names = magpie_features.columns
    magpie_names = [f'MAGPIE_{name}' for name in magpie_names]
    df[magpie_names] = magpie_features.to_numpy()
    
    assert df.notna().all().all()
    
    write_to_csv(df, output_sc_data, comment)
