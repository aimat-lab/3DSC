#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:30:48 2022

@author: Timo Sommer

Train XGB models to compare structural features (DSOAP) versus only the chemical formula (MAGPIE features) and make a learning curve.
"""
from superconductors_3D.machine_learning.Apply_ML_Models_v1_3 import train_with_args, parse_arguments, make_output_directory
from superconductors_3D.machine_learning.own_libraries.data.All_Data import load_All_Data
import os
import argparse
import pandas as pd
from joblib import Parallel, delayed, cpu_count
import datetime
from superconductors_3D.utils.projectpaths import projectpath, Data_Paths
import sys
import numpy as np
from itertools import product
from superconductors_3D.machine_learning.own_libraries.data.All_scores import All_scores, all_scores_filename
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import time
sns.set_theme()

def parse_input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', dest='database', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    args = parser.parse_args()
    return args

def expspace(start, end, steps):
    exp_values = np.exp(np.linspace(np.log(start), np.log(end), steps))
    return exp_values

def plot_learning_curve(runs, xticks, save_dir):
    """Plots a learning curve of all runs.
    """
    all_score_names = ['MSLE', 'MAE', 'r2']
    CVs = ['test', 'train']
    for CV, score_name in product(CVs, all_score_names):
        
        all_scores = {'train_frac': [], 'features': [], score_name: [], 'mean': [], 'sem': []}
        for i in range(len(runs)):
            score_path = os.path.join(runs[i]['outdir'], all_scores_filename)
            scores = All_scores(score_path).get_score_stats()['XGB']['tc'][score_name]
            all_scores['train_frac'].append(runs[i]['train_frac'])
            all_scores['features'].append(runs[i]['features'])
            all_scores[score_name].append(score_name)
            all_scores['mean'].append(scores['mean'][CV])
            all_scores['sem'].append(scores['sem'][CV])
        all_scores = pd.DataFrame(all_scores)
        
        plt.figure()
        sns.lineplot(data=all_scores, x='train_frac', y='mean', hue='features')
        features = all_scores['features'].unique()
        for i, feat in enumerate(features):
            all_scores_feat = all_scores[all_scores['features'] == feat]
            color = sns.color_palette()[i]
            plt.errorbar(x=all_scores_feat['train_frac'], y=all_scores_feat['mean'], yerr=all_scores_feat['sem'], color=color, marker='o', linestyle='None')
        ax = plt.gca()
        plt.title(CV)
        plt.ylabel(score_name)
        plt.xlabel('train fraction')
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(xticks)
        plt.yticks(expspace(min(all_scores['mean']), max(all_scores['mean']), 5))
        plt.minorticks_off()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        plt.tight_layout()
        
        # Save plot and data
        os.makedirs(save_dir, exist_ok=True)
        df_save_path = os.path.join(save_dir, f'learning_curve_{score_name}_{CV}.csv')
        all_scores.to_csv(df_save_path)
        img_save_path = os.path.join(save_dir, f'learning_curve_{score_name}_{CV}.png')
        plt.savefig(img_save_path, dpi=400)
        # plt.show()
        
    return                
    
def train_model(features, args_from_fn, use_models, experiment, output_note, outdirname, calcdir, save_models, CV, n_folds, n_repeats, CV_keep_groups, n_reps, train_frac, domain_colname, is_optimizing_NN, dataset, hparams_file, n_jobs, sample_weights, metrics_sample_weights, use_data_frac):
    sleep_time = np.random.uniform(1, 10)
    print(f'Sleeping for {sleep_time}s.')
    time.sleep(sleep_time)
    
    print(f'Start run with {features} features and a train fraction of {train_frac}.')
    add_params =  {
            'features': features,
            'train_frac': train_frac,
            'CV_keep_groups': CV_keep_groups,
            }
    
    print('Input:\n', sys.argv)
    args = parse_arguments(args_from_fn, use_models, experiment, output_note, outdirname, calcdir, save_models, CV, n_folds, n_repeats, CV_keep_groups, n_reps, train_frac, domain_colname, is_optimizing_NN, dataset, hparams_file, n_jobs, sample_weights, metrics_sample_weights, use_data_frac, add_params)
    print('args.add_params:\n', args.add_params)
    
    # Run ML and measure time of run.
    starttime = datetime.datetime.now()
    ml, outdir = train_with_args(args)
    duration = datetime.datetime.now() - starttime
    print(f'Duration:  {duration}')
    
    run = {
            'features': features,
            'train_frac': train_frac,
            'outdir': outdir
            }
    
    return run

def main(args_from_fn, database, n_cpus, n_reps=3, start_train_frac=0.1, end_train_frac=0.8, n_train_fracs=2):

    use_models = ['XGB']
    output_note = ''
    outdirpath = projectpath('..', 'results', 'machine_learning')
    calcdir = projectpath('machine_learning')
    # Cross validation
    CV = 'Random'    # `KFold`, `LOGO`, `Random` or None
    n_folds = 5     # for KFold
    n_repeats = 1   # for KFold
    CV_keep_groups = 'chemical_composition_sc'     # for KFold, Random
    domain_colname = None  # for LOGO
    # Weights
    sample_weights = 'weight'
    metrics_sample_weights = 'weight'
    # Dataset
    dataset = projectpath('data', 'final', database, f'3DSC_{database}.csv')
    # Hyperparameters
    hparams_file = 'hparams.yml'
    n_jobs = 1
    is_optimizing_NN = False # Only run NN and don't plot anything when optimizing NN
    save_models = True
    # Debugging
    use_data_frac = None    # None for using all data.
    
    
    # =========================================================================
    #             Importance of structural information  
    # =========================================================================
    experiment = f'3DSC_{database}_Importance_of_structural_information'
    outdirname = make_output_directory(outdirpath, label=experiment)
    
    # invert so that the specified random seed is for train_frac=0.8 for repeatibility
    try_train_fracs = expspace(start_train_frac, end_train_frac, n_train_fracs)[::-1]
    try_features = ['MAGPIE', 'MAGPIE+DSOAP']
    
    with Parallel(n_jobs=n_cpus, verbose=1, pre_dispatch='all') as parallel:
        runs = parallel(delayed(train_model)(features, args_from_fn, use_models, f'3DSC_{database}_{features}_{train_frac}', output_note, outdirname, calcdir, save_models, CV, n_folds, n_repeats, CV_keep_groups, n_reps, train_frac, domain_colname, is_optimizing_NN, dataset, hparams_file, n_jobs, sample_weights, metrics_sample_weights, use_data_frac) for features, train_frac in product(try_features, try_train_fracs))
    
    save_dir = os.path.join(outdirname, 'plots')
    plot_learning_curve(
                        runs,
                        xticks=try_train_fracs[::-1],
                        save_dir=save_dir
                        )
                
    
if __name__ == '__main__':
    
    database = 'MP'
    n_cpus = 2
    
    args = parse_input_parameters()
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    
    n_reps = 100 if database == 'MP' else 25
    start_train_frac = 0.1
    end_train_frac = 0.8
    n_train_fracs = 10
    main(args_from_fn={}, database=database, n_cpus=n_cpus, n_reps=n_reps, start_train_frac=start_train_frac, end_train_frac=end_train_frac, n_train_fracs=n_train_fracs)








