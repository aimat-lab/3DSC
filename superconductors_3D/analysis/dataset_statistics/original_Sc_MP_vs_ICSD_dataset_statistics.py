#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:57:03 2021

@author: Timo Sommer

This script generates plots of the train and test columns of the MP and ICSD
"""
from superconductors_3D.utils.projectpaths import projectpath
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
from copy import deepcopy
import numpy as np
import superconductors_3D.machine_learning.own_libraries.data.All_scores as All_Scores
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Experiment import Experiment, results_barplot, results_catplot, av_and_error_df
from scipy.stats import sem
import warnings
sns.set_theme()
# TODO:
    # Add the final df of the best criteria and compare with the final original SuperCon df.


df_paths = {
            'MP':   {
                    'all': projectpath('data', 'final', 'SuperCon', 'mirror_CV_of_run', '211201_dataset_hyperparameter_optimization_MP_100_reps', 'SC_MP_original_MAGPIE.csv'),
                    'matched': #Insert df after run with CV cols
                    }
            }

data = []
for database, df_dict in df_paths.items():
    for sc_name, df_path in df_dict.items():
        df = pd.read_csv(df_path, header=1)
        CV_cols = [col for col in df.columns if col.startswith('CV_')]

    
        # Plot average number of superconductors per train/test set.
        n_unique_sc = df['formula_sc'].nunique()
        df = df.drop_duplicates(['formula_sc'] + CV_cols)
        assert len(df) == n_unique_sc, 'Some of the CV columns seem to be different for the same superconductor! I.e. the same superconductor is in train and test set.'
        mean_test = np.mean([df[cv] == 'test' for cv in CV_cols])
        mean_train = np.mean([sum(df[cv] == 'train') for cv in CV_cols])
        data.append({
                        'database': database,
                        'dataset_type': sc_name,
                        'mean_test': mean_test,
                        'mean_train': mean_train,
                    })
data = pd.DataFrame(data)