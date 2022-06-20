#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:49:50 2021

@author: Timo Sommer

This script adds the CV columns from another run to the whole SuperCon dataset so that the test set is exactly the same for each CV run.
"""
from superconductors_3D.utils.projectpaths import projectpath

database = 'ICSD'

input_sc_data = projectpath('data', 'source', 'SuperCon', 'cleaned', '2.0_all_data_SuperCon_cleaned.csv')
final_matched_df = projectpath('analysis', 'results', '211208_dataset_hyperparameter_optimization_ICSD_25_reps_correct_comps', 'results', 'results_0_ICSD-5', 'All_values_and_predictions.csv')

output_sc_data = projectpath('data', 'final', 'SuperCon', 'mirror_CV_of_run', '211208_dataset_hyperparameter_optimization_ICSD_25_reps_correct_comps', f'SC_{database}_original_MAGPIE.csv')
comment = 'Dataset of original SuperCon data with MAGPIE features and CV columns.'

import pandas as pd
from superconductors_3D.dataset_preparation.utils.lib_generate_datasets import MAGPIE_features
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, movecol
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans

def same_test_data_and_keep_comps_separated(df, matched_df, CV):
    """Returns `df` with the CV col of `matched_df` appended so that the test superconductors are the same and the train superconductors are all others except if their chemical composition is in the test set. For entries where the CV is NaN the whole chemical composition will be excluded as well.
    """
    df = deepcopy(df)
    # Copy test set.
    is_test = matched_df[CV] == 'test'
    test_formulas = matched_df.loc[is_test]['formula_sc'].unique()
    is_test_data = df['formula_sc'].isin(test_formulas)
    df[CV] = np.where(is_test_data, 'test', 'train')
    
    # If a material is part of the train set but it's chemical composition is part of the test set exclude this material due to our grouped CV. The same with NaN entries because in the original dataset `matched_df` these chemical compositions are not in the train set either.
    is_nan = matched_df[CV].isna()
    test_comps = matched_df.loc[is_test | is_nan]['chemical_composition_sc'].unique()
    is_train = df[CV] == 'train'
    is_test_comp = df['chemical_composition_sc'].isin(test_comps)
    df.loc[is_train & is_test_comp, CV] = np.nan
    
    # Sanity check.
    train_comps = df.loc[df[CV] == 'train', 'chemical_composition_sc'].unique().tolist()
    test_comps = df.loc[df[CV] == 'test', 'chemical_composition_sc'].unique().tolist()
    nan_comps = df.loc[df[CV].isna(), 'chemical_composition_sc'].unique().tolist()
    assert len(set(train_comps + test_comps + nan_comps)) == len(train_comps) + len(set(test_comps + nan_comps))
    
    # For debugging.
    # df1 = pd.DataFrame([df['formula_sc'], df['chemical_composition_sc'], df[CV], is_test_data, is_test_comp], ['formula_sc', 'chemical_composition_sc', CV, 'is_test_data', 'is_test_comp']).T
    
    return df


rename_columns = {
                    'tc_sc': 'tc',
                    'sc_class_sc': 'sc_class'
                    }

if __name__ == '__main__':
    
    df = pd.read_csv(input_sc_data, header=1)
    assert not any(df.duplicated('formula_sc'))
    
    # Get CV columns with the same SuperCon entries as in the df where crystal structures and SuperCon entries are matched.
    is_necessary_column = lambda colname: colname.startswith('test_CV_') or colname.startswith('CV_') or colname == 'formula_sc' or colname == 'chemical_composition_sc'
    matched_df = pd.read_csv(final_matched_df, header=1, usecols=is_necessary_column)
    
    # Test CV columns.
    CV_cols = [col for col in matched_df.columns if col.startswith('test_CV_')]
    for CV in CV_cols:
        df = same_test_data_and_keep_comps_separated(df, matched_df, CV)
        
    # Validation CV columns.
    CV_cols = [col for col in matched_df.columns if col.startswith('CV_')]
    for CV in tqdm(CV_cols):
        df = same_test_data_and_keep_comps_separated(df, matched_df, CV)
        
        # Sanity checks that chemical compositions of train and test sets don't overlap and are correct for both dfs.
        df_train_comps = df.loc[df[CV] == 'train', 'chemical_composition_sc'].unique()
        df_test_formulas = df.loc[df[CV] == 'test', 'formula_sc'].unique()
        df_test_comps = df.loc[df[CV] == 'test', 'chemical_composition_sc'].unique()
        df_nan_comps = df.loc[df[CV].isna(), 'chemical_composition_sc'].unique()
        matched_train_comps = matched_df.loc[matched_df[CV] == 'train', 'chemical_composition_sc'].unique()
        matched_test_formulas = matched_df.loc[matched_df[CV] == 'test', 'formula_sc'].unique()
        matched_test_comps = matched_df.loc[matched_df[CV] == 'test', 'chemical_composition_sc'].unique()
        matched_nan_comps = matched_df.loc[matched_df[CV].isna(), 'chemical_composition_sc'].unique()
        assert len(matched_nan_comps) == 0
        assert set(df_test_formulas) == set(matched_test_formulas)
        assert np.isin(matched_train_comps, df_train_comps).all()
        assert np.isin(df_nan_comps, matched_test_comps).all()
        assert len(df_test_comps) + len(df_train_comps) == len(set.union(set(df_test_comps), set(df_train_comps)))
        assert len(matched_test_comps) + len(matched_train_comps) == len(set.union(set(matched_test_comps), set(matched_train_comps)))
        
        
    # Get MAGPIE features.
    formulas = df['formula_sc']
    magpie_features = MAGPIE_features(formulas)
    magpie_names = magpie_features.columns
    magpie_names = [f'MAGPIE_{name}' for name in magpie_names]
    df[magpie_names] = magpie_features.to_numpy()
    
    # Add pseudo sample weight column for easier algorithms.
    df['weight'] = 1
    
    assert df[magpie_names].notna().all().all()
    
    df = df.rename(columns=rename_columns)
    
    write_to_csv(df, output_sc_data, comment)
