#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:54:54 2021

@author: Timo Sommer

This script makes plots of the results of the comparison of SOAP vs MAGPIE features for different cross validations.
"""
from superconductors_3D.utils.projectpaths import projectpath

database = 'MP'


analysis_dir = projectpath('analysis', 'results', f'SC_{database}_dataset', 'plots')

# Directories of results of ML run that shall be plottet with a dictionary specifying the experiment parameters.
result_dirs = {
                # Reference (MAGPIE of all SuperCon entries)
                projectpath('analysis', 'results', 'original_SC_dataset', 'results_2_ground_truth'): {
                                    'cross validation': 'ground truth',
                                    'features': 'MAGPIE (all)'
                                    },
                projectpath('analysis', 'results', 'original_SC_dataset', 'results_1_5-KFold-grouped'): {
                                    'cross validation': '5-fold grouped',
                                    'features': 'MAGPIE (all)'
                                    },
                projectpath('analysis', 'results', 'original_SC_dataset', 'results_0_5-KFold-ungrouped'): {
                                    'cross validation': '5-fold ungrouped',
                                    'features': 'MAGPIE (all)'
                                    },
                
                # New experiments
                projectpath('analysis', 'results', f'SC_{database}_dataset', 'results_0_MAGPIE_ground_truth'): {
                                    'cross validation': 'ground truth',
                                    'features': 'MAGPIE (matched)'
                                    },
                projectpath('analysis', 'results', f'SC_{database}_dataset', 'results_1_SOAP_ground_truth'): {
                                    'cross validation': 'ground truth',
                                    'features': 'SOAP (matched)'
                                    },
                projectpath('analysis', 'results', f'SC_{database}_dataset', 'results_4_MAGPIE_5-KFold-grouped'): {
                                    'cross validation': '5-fold grouped',
                                    'features': 'MAGPIE (matched)'
                                    },
                projectpath('analysis', 'results', f'SC_{database}_dataset', 'results_5_SOAP_5-KFold-grouped'): {
                                    'cross validation': '5-fold grouped',
                                    'features': 'SOAP (matched)'
                                    },
                projectpath('analysis', 'results', f'SC_{database}_dataset', 'results_3_MAGPIE_5-KFold-ungrouped'): {
                                    'cross validation': '5-fold ungrouped',
                                    'features': 'MAGPIE (matched)'
                                    },
                projectpath('analysis', 'results', f'SC_{database}_dataset', 'results_2_SOAP_5-KFold-ungrouped'): {
                                    'cross validation': '5-fold ungrouped',
                                    'features': 'SOAP (matched)'
                                    }
               }

score = 'r2'
plot_models = ['XGB', '1NN']

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import superconductors_3D.machine_learning.own_libraries.data.All_scores as All_Scores

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
    
    sns.set_theme()
    
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    
    for model in plot_models:
        
        data = get_plot_data(result_dirs, model)        
        data = pd.DataFrame(data)
        
        # nicer label
        if score == 'r2':
            nice_r2 = '$r^2$'
            data = data.rename(columns={'r2': nice_r2})
            score_name = nice_r2
        else:
            score_name = score
        
        # Make plot
        plt.figure()
        
        sns.catplot(data=data, x='cross validation', y=score_name, hue='features', kind='bar', legend=False)
        
        plt.legend(loc='best')
        plt.title(model)
        plt.tight_layout()
        
        if score_name == '$r^2$':
            plt.ylim(0, 1)
        
        filename = f'{model}_results_SOAP_vs_MAGPIE_for_all_CVs.png'
        filename = os.path.join(analysis_dir, filename)
        plt.savefig(filename, dpi=300)
        plt.show()
    
        
        
    
    
    
    
