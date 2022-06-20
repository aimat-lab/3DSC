#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:17:50 2021

@author: timo
Different functions for plotting for the analysis.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import gc
import seaborn as sns
from copy import deepcopy
sns.set_theme()


def bool2string(df):
    """Make any bool values to string values for plotting. This is useful because Seaborn doesn't like mixed bool and string columns and otherwise plots 0/1 for False/True."""
    mask = df.applymap(type) != bool
    d = {True: 'True', False: 'False'}
    df_result = df.where(mask, df.replace(d))
    return(df_result)

def plot_correlations(df_results, df_corrs, save_dir, colorbar={}, plot_log_feats=[], max_plot_corrs=np.int, ylimits=[None, None], scalarFormatter=[False, False], dpi=150):
    """Plot correlation plot of columns in df_corrs that are not 'corr'."""
    # Seaborn doesn't like mixed bool and string columns and otherwise plots 0/1 for False/True.
    df_results = deepcopy(bool2string(df_results))
    
    for i, row in df_corrs.iterrows():
        # Only plot max_plot_corrs plots.
        if i >= max_plot_corrs:
            break
        plt.figure()
        feat1, feat2, corr = row
        
        # Setup colors defined by colorbar['colname'].
        if colorbar:
            hue_norm = plt.Normalize(df_results[colorbar['colname']].min(), df_results[colorbar['colname']].max())
            cmap = colorbar['cmap']
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=hue_norm)
            sm.set_array([])
            hue = colorbar['colname']
            # For datapoints on top of each other make the hue the median score.
            df_plot = df_results.groupby(by=[feat1, feat2]).apply(lambda subdf: subdf[colorbar['colname']].median()).rename(colorbar['colname']).reset_index()
        else:
            hue = None
            hue_norm = None
            cmap = None
            df_plot = df_results
            
        ax = sns.scatterplot(data=df_plot, x=feat1, y=feat2, hue=hue, palette=cmap, hue_norm=hue_norm, legend=False)
        
        if feat1 in plot_log_feats:
            ax.set(xscale='log')
            if scalarFormatter[0]:
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        if feat2 in plot_log_feats:
            ax.set(yscale='log')
            if scalarFormatter[1]:
                ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        
        
        plt.ylim(*ylimits)
        plt.title(f'Correlation $r={corr:.3f}$')
        if colorbar:
            clb = ax.figure.colorbar(sm)
            clb.ax.set_title(colorbar['title'])
        # Save
        outpath = os.path.join(save_dir, f'{i}_correlation_{feat1}_{feat2}.png')
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.cla(), plt.clf(), plt.close(), gc.collect()
    return(ax, outpath)


        
    
    # Plot results for each combination of hyperparameters only with different random seeds.
# =============================================================================
# if combination_plots:
#     for modelname in modelnames:
#         all_scorenames = [exp.get_scorename(model, average_func, score, mode) for model in modelnames for mode in ['test', 'train']]
#         
#         savedir = os.path.join(exp.analysis_dir, 'Combination_plots')
#         if not os.path.exists(savedir):
#             os.mkdir(savedir)
#         
#         # Get average and std for all runs with different random_seeds.
#         hparams = [h for h in hparams if not h == 'random_seed']
#         df_grouped = df.groupby(by=hparams)
#         df_results = pd.DataFrame()
#         for scorename in all_scorenames:
#             if average_func == 'mean':
#                 df_results[scorename + '_mean'] = df_grouped[scorename].mean()
#             elif average_func == 'median':
#                 df_results[scorename + '_median'] = df_grouped[scorename].median()
#             df_results[scorename + '_std'] = df_grouped[scorename].std()            
#         df_results = df_results.reset_index()
#         df_results = df_results.reset_index()
#         
#         # Find hparams that actually vary to display them in plot.
#         show_hparams = []
#         for h in hparams:
#             if df_results[h].nunique() > 1:
#                 show_hparams.append(h)
#         
#         for i, results in df_results.iterrows():
#     
#             plt.figure()
#             for mode in ['test', 'train']:
#                 color = 'r' if mode == 'test' else 'b'
#                 averages = []
#                 stds = []
#                 models = []
#                 for model in modelnames:
#                     models.append(model)
#                     scorename = exp.get_scorename(model, average_func, score, mode)
#                     averages.append(results[scorename + '_' +  average_func])
#                     stds.append(results[scorename + '_std'])
#                 
#                 plt.errorbar(x=models, y=averages, yerr=stds, fmt='.', markersize=13, c=color, label=mode)
#             
#             plt.ylabel('r²')
#             plt.xlabel('model')
#             plt.legend()
#             
#             hparam_values = [f'{h}={results[h]}' for h in show_hparams]
#             outfile = f'Combination_plot_{i}_' + '_'.join(hparam_values) + '.png'
#             outpath = os.path.join(savedir, outfile)
#             plt.savefig(outpath, dpi=150, bbox_inches="tight")
#             plt.cla(), plt.clf(), plt.close(), gc.collect()
# =============================================================================
