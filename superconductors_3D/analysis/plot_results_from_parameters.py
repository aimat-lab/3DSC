#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:54:54 2021

@author: Timo Sommer

This script makes plots of the results of the comparison of SOAP vs MAGPIE features for different cross validations.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os


exp_dir = projectpath('..', 'analysis', 'results', '220210_phys_groups_comparison')


score = 'MSLE'
ylim = (1e-6, 1)
xlim = None#(100, 1.5e4)

sep_feature = ['add_params__database']
features = ['add_params__group', 'add_params__features']
baseline = None


max_or_min_of_feature = False#{'r2': 'max', 'MSLE': 'min'}    # False or dict
plot_only_top = False#{'top': 5}


plot_CVs = ['train']

average_type = 'mean'
error_type = 'sem'

yscale = 'log'

# # Build up dictionary with directories and their parameter values.
# result_dirs = {}
# for dir in all_result_dirs:
#     name = dir#os.path.basename(dir)
#     for param, param_values in plot_parameters.items():
#         for pattern, value in param_values.items():
#             if pattern in name:
#                 try:
#                     result_dirs[dir][param] = value
#                 except KeyError:
#                     result_dirs[dir] = {param: value}
#                 # If one pattern is found, skip the others. That way you can exclude patterns that are subpatterns of other patterns by specifiying them later.
#                 break
                




# analysis_dir = os.path.join(result_dir, 'plots')





import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
from copy import deepcopy
import numpy as np
import superconductors_3D.machine_learning.own_libraries.data.All_scores as All_Scores
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Experiment import Experiment, results_barplot, results_catplot, av_and_error_df
import scipy
import warnings
sns.set_theme()

# Set average and error function
error_functions = {'std': scipy.std, 'sem': scipy.stats.sem}
average_functions = {'mean': np.mean, 'median': np.median}
average = average_functions[average_type]
error = error_functions[error_type]



# def get_plot_data(result_dirs, model):
#     """Returns a dictionary with plotting data.
#     """
#     data = []
#     for result_dir, params in result_dirs.items():
        
#         score_file = os.path.join(result_dir, 'All_scores.csv')
#         scores = All_Scores.All_scores(score_file)
#         test_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='test')
#         train_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='train') 
        
#         for test, train in zip(test_values, train_values):
#             test_name = score + ' (test)'
#             train_name = score + ' (train)'
#             row = {'model': model, test_name: test, train_name: train}
#             row.update(params)
#             data.append(row)
#     return data


plot_xlabels = {
                'add_params__nblocks': 'model size',
                'add_params__prev_model': 'transfer from',
                'add_params__features': 'features',
                'add_params__mean_n_train_sc': 'n train superconductors',
                'add_params__only_abs_matches': 'only absolute matches'
                }
plot_ylabels = {
                'r2': '$r^2$',
                'logr2': '$r^2 _\log$'
                }
plot_categories = {
                    'None': 'reference',
                    'Heavy_fermion': 'Heavy\nfermion',
                    'KFold_chem_comp': '$\mathrm{KFold}_\mathrm{comp}$',
                    'KFold_no_grouping': 'KFold',
                    'LOCO_5_kmeans': '$\mathrm{LOCO}_\mathrm{KMeans}$',
                    'LOCO_phys': '$\mathrm{LOCO}_\mathrm{phys}$',
                    'LOCO_num_elements': '$\mathrm{LOCO}_\mathrm{num\ el}$',
                    'drop_duplicate_superconductors': 'randomly\ndrop crystals',
                    'without_lattice_feats': 'no symmetry\nfeatures',
                    True: 'true',
                    False: 'false',
                    'graph': 'MEGNet',
                    'MAGPIE': 'MAGPIE',
                    'MAGPIE (all)': 'MAGPIE\n(all)',
                    'SOAP': 'SOAP',
                    'SOAP+electro': 'SOAP\n+electro',
                    'MAGPIE+SOAP': 'MAGPIE\n+DSOAP',
                    'MAGPIE+SOAP+electro': 'MAGPIE\n+SOAP\n+electro',
                    'no_crystal_temp_given_2+totreldiff': '$T_{cry}$\n+$\Delta_\mathrm{totrel}$',
 'no_crystal_temp_given_2+correct_formula_frac': '$T_{cry}$\n+formula',
 'totreldiff+no_crystal_temp_given_2': '$\Delta_\mathrm{totrel}$\n+$T_{cry}$',
 'no_crystal_temp_given_2': '$T_{cry}$',
 'totreldiff': '$\Delta_\mathrm{totrel}$',
 'correct_formula_frac+totreldiff': 'formula\n+$\Delta_\mathrm{totrel}$',
 'no_crystal_temp_given_2+totreldiff+correct_formula_frac': '$T_{cry}$\n+$\Delta_\mathrm{totrel}$\n+formula',
 'no_crystal_temp_given_2+correct_formula_frac+totreldiff': '$T_{cry}$\n+formula\n+$\Delta_\mathrm{totrel}$',
 'totreldiff+no_crystal_temp_given_2+correct_formula_frac': '$\Delta_\mathrm{totrel}$\n+$T_{cry}$\n+formula',
 'correct_formula_frac': 'formula',
 'totreldiff+correct_formula_frac+no_crystal_temp_given_2': '$\Delta_\mathrm{totrel}$\n+formula\n+$T_{cry}$',
 'correct_formula_frac+no_crystal_temp_given_2+totreldiff': 'formula\n+$T_{cry}$\n+$\Delta_\mathrm{totrel}$',
 'correct_formula_frac+totreldiff+no_crystal_temp_given_2': 'formula\n+$\Delta_\mathrm{totrel}$\n+$T_{cry}$',
 '': 'no\ncriteria',
 'totreldiff+correct_formula_frac': '$\Delta_\mathrm{totrel}$\n+formula',
 'correct_formula_frac+no_crystal_temp_given_2': 'formula\n+$T_{cry}$',
 'e_above_hull_2': 'e ab. hull',
 'e_above_hull_2+correct_formula_frac': 'e ab. hull\n+formula',
 'correct_formula_frac+e_above_hull_2': 'formula\n+e ab. hull',
 'totreldiff+e_above_hull_2': '$\Delta_\mathrm{totrel}$\n+e ab. hull',
 'e_above_hull_2+totreldiff': 'e ab. hull\n+$\Delta_\mathrm{totrel}$',
 'totreldiff+correct_formula_frac+e_above_hull_2': '$\Delta_\mathrm{totrel}$\n+formula\n+e ab. hull',
 'totreldiff+e_above_hull_2+correct_formula_frac': '$\Delta_\mathrm{totrel}$\n+e ab. hull\n+formula',
 'correct_formula_frac+totreldiff+e_above_hull_2': 'formula\n+$\Delta_\mathrm{totrel}$\n+e ab. hull',
 'correct_formula_frac+e_above_hull_2+totreldiff': 'formula\n+e ab. hull\n+$\Delta_\mathrm{totrel}$',
 'e_above_hull_2+totreldiff+correct_formula_frac': 'e ab. hull\n+$\Delta_\mathrm{totrel}$\n+formula',
 'e_above_hull_2+correct_formula_frac+totreldiff': 'e ab. hull\n+formula\n+$\Delta_\mathrm{totrel}$',
                    }
# Huelabels should be like categorical labels but without linebreak
plot_huelabels = {key: val.replace('\n', '') for key, val in plot_categories.items()}

plot_order = list(plot_categories.keys())

def get_unique_permutations_of_column_values(df, columns):
    unique_combs = df[columns].drop_duplicates()
    all_permutations = [dict(row) for _, row  in unique_combs.iterrows()]
    return all_permutations

if __name__ == '__main__':
    
    analysis_dir = os.path.join(exp_dir, 'plots')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    exp = Experiment(exp_dir)
    run_dirs = exp.get_run_dirs_and_params()
    all_data = exp.get_long_df_with_scores_and_params(
                                    scores=[score],
                                    run_dirs=run_dirs
                                    )
    plot_models = all_data['model'].unique()
    
    # assert all(all_data.groupby(list(plot_parameters.keys())).size().values == all_data.groupby(['run dir']).size().values), 'You mixed up some directories!'
    
    if 'add_params__criteria' in all_data:
        all_data['add_params__criteria'] = all_data['add_params__criteria'].apply(lambda l: '+'.join(l) if isinstance(l,list) else l)
    
    groupby = ['run dir'] + features + sep_feature
    df_all_averages =  av_and_error_df(all_data, groupby=groupby, columns=[f'{score} (test)', f'{score} (train)'], average=average, error=error)
    assert len(df_all_averages) == len(run_dirs)
    
    # df_all_averages = df_all_averages[~df_all_averages['add_params__features'].str.contains('electro')]     # TODO
    

    
    permutations = get_unique_permutations_of_column_values(all_data, sep_feature)
    
    for model, subsets in product(plot_models, permutations):
        subset = '+'.join(subsets.values())
        
        data0 = deepcopy(all_data)
        df_averages = deepcopy(df_all_averages)
        for feat_name, value in subsets.items():
            data0 = data0.loc[data0[feat_name] == value]
            df_averages = df_averages.loc[df_averages[feat_name] == value]
        
        order = [c for c in plot_order if c in df_averages[features[0]].values]

            
        data = av_and_error_df(data0, groupby=features, columns=[f'{score} (test)', f'{score} (train)'], average=average, error=error)
        if len(data) != data0['run dir'].nunique():
            warnings.warn('Some of the plotted averages come from multiple directories!')
        
        if max_or_min_of_feature:
            mode = max_or_min_of_feature[score]
            print(f'Plotting only the {mode} of {features}.')
            data = deepcopy(df_averages)
            if mode == 'max':
                data = df_averages.groupby(features).apply(lambda subdf: subdf.loc[subdf[f'{score} (test)'] == subdf[f'{score} (test)'].max()])
            elif mode == 'min':
                data = df_averages.groupby(features).apply(lambda subdf: subdf.loc[subdf[f'{score} (test)'] == subdf[f'{score} (test)'].min()])
            else:
                raise ValueError()
        else:
            mode = 'mean'
        
        if plot_only_top:
            ascending = {'r2': False, 'MSLE': True, 'MAE': True}
            mode = ascending[score]
            data = data.sort_values(f'{score} (test)', ascending=mode)
            data = data.iloc[0:plot_only_top['top']]

        
        # TODO remove
        sortby = {'KFold_no_grouping': 0, 'KFold_chem_comp': 1, 'LOCO_num_elements': 2, 'LOCO_5_kmeans': 3, 'LOCO_phys': 4}
        data['sortby'] = data[features[0]].replace(sortby)
        data = data.sort_values(by='sortby').drop(columns='sortby')

        title = f'{subset} & {average_type} & {error_type}'
        ylabel = score if not score in plot_ylabels else plot_ylabels[score]
        xlabel = None#plot_xlabels[features[0]]

        filename = f'results_barplot_{score}_{subset}_{"+".join(features)}_{model}_{"+".join(plot_CVs)}.png'
        filename = os.path.join(analysis_dir, filename)
        results_barplot(data, features=features, score=score, savepath=filename, plot_CVs=plot_CVs, xlabel=xlabel, ylabel=ylabel, title=title, ylim=ylim, xticklabels=plot_categories, huelabels=plot_huelabels, baseline=baseline, yscale=yscale)
        
        
        # filename = f'results_stripplot_{score}_{subset}_{model}_{"+".join(plot_CVs)}.png'    #   TODO
        # filename = os.path.join(analysis_dir, filename)
        # results_catplot(df_averages, features=features, score=score, savepath=filename, plot_CVs=plot_CVs, xlabel=xlabel, ylabel=ylabel, title=title, ylim=ylim, mode='strip', plot_av=False, plot_unc=False, xticklabels=plot_categories, order=order)
        
        
        # filename = f'results_lineplot_{score}_{subset}_{model}_{"+".join(plot_CVs)}.png'
        # filename = os.path.join(analysis_dir, filename)
        # xlabel = plot_xlabels[features[0]] if features[0] in plot_xlabels else features[0]
        # results_catplot(data, features=features, score=score, savepath=filename, plot_CVs=plot_CVs, xlabel=xlabel, ylabel=ylabel, title=title, ylim=ylim, mode='line', plot_av=False, plot_unc=False, log_x=True, log_y=True, xlim=xlim)
            
            
        
        
        
        
