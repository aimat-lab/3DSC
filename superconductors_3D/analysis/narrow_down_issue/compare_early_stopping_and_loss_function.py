#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:54:54 2021

@author: Timo Sommer

"""
from superconductors_3D.utils.projectpaths import projectpath
import os

result_dir = '/home/timo/superconductors_3D/analysis/results/211116_MEGNet_early_stopping_and_loss_comparisons_with_crys_sys'

result_dirs = {
    # '/home/timo/superconductors_3D/analysis/results/211116_MEGNet_early_stopping_and_loss_comparisons_with_crys_sys/results/results_0_normal': {'loss': 'MSE', 'early stopping': False},
    '/home/timo/superconductors_3D/analysis/results/211116_MEGNet_early_stopping_and_loss_comparisons_with_crys_sys/results/results_0_huber-early_stopping': {'loss': 'Huber', 'early stopping': True},
    '/home/timo/superconductors_3D/analysis/results/211116_MEGNet_early_stopping_and_loss_comparisons_with_crys_sys/results/results_0_mse-early_stopping': {'loss': 'MSE', 'early stopping': True},
    '/home/timo/superconductors_3D/analysis/results/211116_MEGNet_early_stopping_and_loss_comparisons_with_crys_sys/results/results_0_sc_huber-early_stopping': {'loss': 'Huber, positive', 'early stopping': True},
    }


# plot_in_separate_figures = []
plot_in_same_figure = ['loss']

analysis_dir = os.path.join(result_dir, 'plots')

score = 'MSLE'
plot_models = ['MEGNet']
ylim = (0, None)
plot_train = True

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import superconductors_3D.machine_learning.own_libraries.data.All_scores as All_Scores
from scipy.stats import sem
sns.set_theme()


average = np.mean
uncertainty = sem



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

def get_average_and_uncertainty(data0, score_name, average, uncertainty):
    """Reduces the data to average and uncertainty.
    """                
    score_type = f'{score_name} (test)'
    av_and_uncertainty = lambda subdf: pd.Series([average(subdf[score_type]), uncertainty(subdf[score_type])], [f'{score_type}', f'uncertainty {score_type}'])
    data = data0.groupby(plot_in_same_figure).apply(av_and_uncertainty)
    
    score_type = f'{score_name} (train)'
    av_and_uncertainty = lambda subdf: pd.Series([average(subdf[score_type]), uncertainty(subdf[score_type])], [f'{score_type}', f'uncertainty {score_type}'])
    data = pd.concat((data, data0.groupby(plot_in_same_figure).apply(av_and_uncertainty)), axis=1)
    data = data.reset_index()
    return data

def plot_results(data, feature, plot_train, score_name, sep_feature, ylim, analysis_dir):
    """Makes a bar plot of the results.
    """
    # Make plot
    plt.figure()
    if not plot_train:
        colors = sns.color_palette()[:len(data)]
        plt.bar(data[plot_in_same_figure], data[f'{score_name} (test)'], yerr=data[f'uncertainty {score_name} (test)'], color=colors, capsize=10)
    else:
        x = np.arange(len(data[plot_in_same_figure]))
        width = 0.35
        colors = sns.color_palette()[:2] 
        plt.bar(x - width/2, data[f'{score_name} (train)'], width, yerr=data[f'uncertainty {score_name} (train)'], color=colors[0], capsize=10, label='train')
        plt.bar(x + width/2, data[f'{score_name} (test)'], width, yerr=data[f'uncertainty {score_name} (test)'], color=colors[1], capsize=10, label='test')
        
        
        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(data[feature])
        plt.legend()
        
    if score_name == 'r2':
        plt.ylabel('$r^2$')
    else:
        plt.ylabel(score_name)
    plt.xlabel(feature)
    
    plt.title(sep_feature)
    plt.tight_layout()
    
    plt.ylim(ylim)
    
    filename = f'results_{score}_{sep_feature}_{model}.png'
    filename = os.path.join(analysis_dir, filename)
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == '__main__':
    
    
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    all_features = plot_in_same_figure #+ plot_in_separate_figures
    
    for model in plot_models:
        
        all_data = get_plot_data(result_dirs, model)        
        all_data = pd.DataFrame(all_data)
        assert all_data.groupby(all_features).size().nunique() == 1, 'You probably mixed up some directories.'
        
        # for sep_feature_name in plot_in_separate_figures:
        #     for sep_feature in all_data[sep_feature_name].unique():
        # data0 = all_data.loc[all_data['group'] == sep_feature]
        data = get_average_and_uncertainty(all_data, score, average, uncertainty)
        data = data.iloc[[2, 0, 1]]
        
        sep_feature = ''
        if len(plot_in_same_figure) > 1:
            data['run'] = data.apply(lambda row: f'loss: {row["loss"]}\nearly stopping: {row["early stopping"]}', axis=1)
            feature = 'run'
        else:
            feature = plot_in_same_figure[0]
        plot_results(data, feature, plot_train, score, sep_feature, ylim, analysis_dir)
    
                
    
        
        
    
    
    
    
