#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:05:35 2021

@author: Timo Sommer

This script contains a class for dealing with feature importances.
"""


data_name = 'feature_importances.csv'

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML
from itertools import product

    
def plot_feature_importances(importances, features, outpath, plot_max=10):     
    """Plot feature importances.
    """    
    # Sort
    order = np.argsort(-importances)
    importances = importances[order]
    features = np.array(features)[order]
    
    # Plot only the first if there are too many features.
    if len(features) > plot_max:
        importances = importances[:plot_max]
        features = features[:plot_max]
    
    plt.figure()
    sns.barplot(x=importances*100, y=features)
    plt.xlabel("Relative importance (%)")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()
    return 

def feature_importances_from_models(rundir, features, modelnames, repetitions):
    """Calculates feature importances and returns a df with sorted features and importances.
    """
    data = {'features': features}
    
    for modelname in modelnames:
        model_importances = []
        for rep in repetitions:
            # Get feature importances for single model of one repetition.
            model = ML.get_saved_model(modelname, rep, rundir, regressor=True)
            importances = get_feature_importances(model)
            assert round(np.sum(importances), 4) == 1, f"Feature importances in {modelname} don't sum up to 1 but are {np.sum(importances)}."

            colname = feature_importance_colname(modelname, rep)
            
            data[colname] = importances
            model_importances.append(importances)
        
        # Calculate average for model over all repetitions.
        model_importances = np.mean(model_importances, axis=0)
        colname = feature_importance_colname(modelname, 'total')
        data[colname] = model_importances
   
    data = pd.DataFrame(data)
    # Sort so that the 'total' columns come first.
    mean_cols = [col for col in data.columns if col.endswith('total')]
    rep_cols = [col for col in data.columns if not (col.endswith('total') or col == 'features')]
    sort_cols = ['features'] + mean_cols + rep_cols
    data = data[sort_cols]
    return(data)

def get_feature_importances(model):
    """Returns the importances for a random forest like model. XGB has a different API.
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
            importances = model.get_score(importance_type='gain')   # XGB
            importances = np.array(list(importances.values()))        
    return(importances)

def feature_importance_colname(model, repetition):
    return f'FI_{model}_{repetition}'

def save_Feature_Importances(df, outpath, features, modelnames, repetitions, **kwargs):
    metadata = {
            'models': modelnames,
            'features': features,
            'repetitions': repetitions
            }
    metadata.update(kwargs)
        
    ML.save_df_and_metadata(df, metadata, outpath)
    print(f'Saved Feature_Importances with metadata.')

    

            
