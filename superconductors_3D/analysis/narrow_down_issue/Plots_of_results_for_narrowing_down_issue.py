#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:54:54 2021

@author: Timo Sommer

This script makes plots of the results of the comparison of SOAP vs MAGPIE features for different cross validations.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os

database = 'MP'

result_dir = projectpath('analysis', 'results', f'211102_narrow_down_issue_fixed_SOAP', database)


# Directories of results of ML run that shall be plotted with a dictionary specifying the experiment parameters.
all_result_dirs = [os.path.join(result_dir, dirname) for dirname in os.listdir(result_dir)]
result_dirs = {}

# Recognize parameters from the names of the result directories. For each parameter you can specify multiple choices and their names as dictionary. If one pattern is a subpattern of another pattern, specify the smaller pattern last because after finding one pattern all other ones will be skipped.
recognize_parameters = {'subset': 
                                    {
                                    '_all-': 'all data',
                                    '_only_one_crystal_structure_per_SuperCon_entry-': 'SuperCon entries with unique crystal structure',
                                    '_totreldiff=0-': 'perfect matches',
                                    '_both_conditions-': 'both conditions'
                                    },
                        'features':
                                    {
                                    '-MAGPIE+SOAP': 'MAGPIE + SOAP',
                                    '-MAGPIE': 'MAGPIE',
                                    '-SOAP': 'SOAP'
                                    }
                        }

# Build up dictionary with directories and their parameter values.
result_dirs = {}
for dir in all_result_dirs:
    name = os.path.basename(dir)
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

score = 'r2'
plot_models = ['XGB']
ylim = (0, 1)


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
        score_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='test')      
        
        for score_value in score_values:
            row = {'model': model, score: score_value}
            row.update(params)
            data.append(row)
    return data



if __name__ == '__main__':
    
    
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    
    for model in plot_models:
        
        all_data = get_plot_data(result_dirs, model)        
        all_data = pd.DataFrame(all_data)
        
        # nicer label
        if score == 'r2':
            nice_r2 = '$r^2$'
            all_data = all_data.rename(columns={'r2': nice_r2})
            score_name = nice_r2
        else:
            score_name = score
        
        for subset in all_data['subset'].unique():
            data = all_data.loc[all_data['subset'] == subset]
            av_and_error = lambda subdf: pd.Series([average(subdf[score_name]), error(subdf[score_name])], [f'{score_name}', 'error'])
            data = data.groupby('features').apply(av_and_error).reset_index()
            
            # Make plot
            plt.figure()
            colors = sns.color_palette()[:len(data)]
            plt.bar(data['features'], data[score_name], yerr=data['error'], color=colors, capsize=10)
            
            plt.ylabel(score_name)
            plt.xlabel('features')
            
            plt.title(subset)
            plt.tight_layout()
            
            plt.ylim(ylim)
            
            filename = f'results_{score}_{subset}_{model}.png'
            filename = os.path.join(analysis_dir, filename)
            plt.savefig(filename, dpi=300)
            plt.show()
    
        
        
    
    
    
    
