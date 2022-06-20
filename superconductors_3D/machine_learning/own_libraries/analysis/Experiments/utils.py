#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:54:40 2021

@author: timo
This script contains utility functions for the analysis of experiments.
"""


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import itertools
from .plot_utils import bool2string


def correlations(df, cols1: list, cols2: list, colname1: str='columns1', colname2: str='columns2'):
    """Returns a df with the correlations of cols1 and cols2. Colname1 and Colname2 are used as names for cols1 and cols2 for the resulting df.
    """
    df = bool2string(df)
    combinations = itertools.product(cols1, cols2)
    all_corrs = {colname1: [], colname2: [], 'corr': []}
    for col1, col2 in combinations:
        if col1 == col2: 
            continue
        corr, _ = spearmanr(df[col1], df[col2])
        all_corrs[colname1].append(col1)
        all_corrs[colname2].append(col2)
        all_corrs['corr'].append(corr)
    df_corrs = pd.DataFrame(all_corrs)
    # Sort by absolute value of correlation.
    df_corrs = df_corrs.sort_values(by='corr', key=np.abs, ascending=False)
    df_corrs = df_corrs.reset_index(drop=True)
    return(df_corrs)
