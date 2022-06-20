#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:54:54 2021

@author: Timo Sommer

This script makes plots of the results of the comparison of SOAP vs MAGPIE features for different cross validations.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os

database = 'ICSD'

result_dir = projectpath('analysis', 'results', f'211109_compare_performance_of_physical_groups', database, 'results')


# Directories of results of ML run that shall be plotted with a dictionary specifying the experiment parameters.
all_result_dirs = [dirname for dirname, _, _ in os.walk(result_dir) if os.path.exists(os.path.join(dirname, 'All_scores.csv'))]

# Recognize parameters from the names of the result directories. For each parameter you can specify multiple choices and their names as dictionary. If one pattern is a subpattern of another pattern, specify the smaller pattern last because after finding one pattern all other ones will be skipped.
recognize_parameters = {
                        'features':
                                    {
                                    '_MAGPIE-all': 'MAGPIE\n(all data)',
                                    '_MAGPIE+SOAP+electronic+crys_sys': 'MAGPIE\n+SOAP\n+electronic',
                                    '_MAGPIE+SOAP': 'MAGPIE\n+SOAP',
                                    '_MAGPIE+PCA_SOAP': 'MAGPIE\n+PCA SOAP',
                                    '_PCA_SOAP': 'PCA\nSOAP',
                                    '_MAGPIE': 'MAGPIE',
                                    '_SOAP': 'SOAP'
                                    },
                        'group':
                                {
                                '/all': 'All',
                                'Carbon': 'Carbon',
                                'Chevrel': 'Chevrel',
                                'Cuprate': 'Cuprate',
                                'Ferrite': 'Ferrite',
                                'Heavy_fermion': 'Heavy fermion',
                                'Other': 'Other',
                                'Oxide': 'Oxide'
                                }
                        }
plot_train = False

# Build up dictionary with directories and their parameter values.
result_dirs = {}
for dir in all_result_dirs:
    name = dir#os.path.basename(dir)
    for param, param_values in recognize_parameters.items():
        for pattern, value in param_values.items():
            if pattern in name:
                try:
                    result_dirs[dir][param] = value
                except KeyError:
                    result_dirs[dir] = {param: value}
                # If one pattern is found, skip the others. That way you can exclude patterns that are subpatterns of other patterns by specifiying them later.
                break
                




analysis_dir = os.path.join(result_dir, 'plots')

score = 'MAE'
plot_models = ['XGB']
ylim = (0, None)


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import superconductors_3D.machine_learning.own_libraries.data.All_scores as All_Scores
from scipy.stats import sem
sns.set_theme()


average = np.mean
error = sem



def get_plot_data(result_dirs, model):
    data = []
    for result_dir, params in result_dirs.items():
        
        score_file = os.path.join(result_dir, 'All_scores.csv')
        scores = All_Scores.All_scores(score_file)
        test_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='test')
        train_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='train') 
        
        for test, train in zip(test_values, train_values):
            test_name = score + ' (test)'
            train_name = score + ' (train)'
            row = {'model': model, test_name: test, train_name: train}
            row.update(params)
            data.append(row)
    return data



if __name__ == '__main__':
    
    
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    
    for model in plot_models:
        
        all_data = get_plot_data(result_dirs, model)        
        all_data = pd.DataFrame(all_data)
        assert all_data.groupby(['features', 'group']).size().nunique() == 1, 'You probably mixed up some directories.'
        
        # nicer label
        if score == 'r2':
            nice_r2 = '$r^2$'
            all_data = all_data.rename(columns={'r2': nice_r2})
            score_name = nice_r2
        else:
            score_name = score
        
        for subset in all_data['group'].unique():
            
            data0 = all_data.loc[all_data['group'] == subset]
            
            score_type = f'{score_name} (test)'
            av_and_error = lambda subdf: pd.Series([average(subdf[score_type]), error(subdf[score_type])], [f'{score_type}', f'error {score_type}'])
            data = data0.groupby('features').apply(av_and_error)
            
            score_type = f'{score_name} (train)'
            av_and_error = lambda subdf: pd.Series([average(subdf[score_type]), error(subdf[score_type])], [f'{score_type}', f'error {score_type}'])
            data = pd.concat((data, data0.groupby('features').apply(av_and_error)), axis=1)
            data = data.reset_index()

            
            # Make plot
            plt.figure()
            if not plot_train:
                colors = sns.color_palette()[:len(data)]
                plt.bar(data['features'], data[f'{score_name} (test)'], yerr=data[f'error {score_name} (test)'], color=colors, capsize=10)
            else:
                x = np.arange(len(data['features']))
                width = 0.35
                colors = sns.color_palette()[:2] 
                plt.bar(x - width/2, data[f'{score_name} (train)'], width, yerr=data[f'error {score_name} (train)'], color=colors[0], capsize=10, label='train')
                plt.bar(x + width/2, data[f'{score_name} (test)'], width, yerr=data[f'error {score_name} (test)'], color=colors[1], capsize=10, label='test')
                
                plt.gca().set_xticks(x)
                plt.gca().set_xticklabels(data['features'])
                plt.legend()
                
            
            plt.ylabel(score_name)
            plt.xlabel('features')
            
            plt.title(subset)
            plt.tight_layout()
            
            plt.ylim(ylim)
            
            filename = f'results_{score}_{subset}_{model}.png'
            filename = os.path.join(analysis_dir, filename)
            plt.savefig(filename, dpi=300)
            plt.show()
    
        
        
    
    
    
    
