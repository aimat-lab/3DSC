#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:12:48 2021

@author: Timo Sommer

This script outputs statistics about the runs of the dataset hyperparameter optimization.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os
import numpy as np
import pandas as pd
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Experiment import Experiment

exp_dir = projectpath('analysis', 'results', '211201_dataset_hyperparameter_optimization_MP_100_reps')

def mean_or_unique(series):
    """Returns either the mean of a series if this makes sense, otherwise the unique value if there is only one.
    """
    try:
        value = np.mean(series)
    except TypeError:
        if len(np.unique(series)) == 1:
            value = np.unique(series)[0]
        else:
            value = None
    return value
          
models = ['XGB']
targets = ['tc']
scores = ['r2', 'MSLE', 'MAE']
exp = Experiment(exp_dir)
df = exp.get_av_and_error_scores_and_params(models, targets, scores)


usecols = ['model', 'target', 'score', 'run dir', 'r2 (test)', 'r2 (train)', 'MSLE (test)', 'MSLE (train)', 'MAE (test)', 'MAE (train)',
       'add_params__criteria', 'add_params__database', 'add_params__features',
       'add_params__n_data_points', 'add_params__n_exclude_if_more_structures',
           ]
df = df[usecols]
df_runs = df.groupby('run dir').apply(lambda subdf: pd.Series([mean_or_unique(subdf[col]) for col in subdf.columns], subdf.columns)).reset_index(drop=True)