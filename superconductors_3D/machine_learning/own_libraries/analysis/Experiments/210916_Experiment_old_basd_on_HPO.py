#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:46:54 2021

@author: timo
This is a script that contains utilities for analysing the results of several ML runs (experiments).
"""

import os
import numpy as np
import pandas as pd
import itertools
import yaml
from .utils import correlations
from .plot_utils import bool2string, plot_correlations
from Own_functions import write_to_csv
from copy import deepcopy


default_log_hparams = ['RGM_erm_e', 'RGM_holdout_e', 'RGM_rgm_e', 'coeff_lr_classifier', 'learning_rate', 'nn_l2', 'NN_clip_grad']

default_ignore_hparams = ['GB_n_estimators', 'GP_alpha', 'RF_n_estimators', 'lr_power_t', 'lr_scheduler_factor', 'nn_patience', 'random_seed']
default_results_path = 'results/results_0_LOGO'


class Experiment():
    """A class for analysing an experiment which consists of several ML runs with different hyperparameters.
    """
    def __init__(self, exp_dir, modelnames, target, score='MAE', average_func='mean', average_fn=np.mean, ignore_hparams=default_ignore_hparams, log_hparams=default_log_hparams, results_path=default_results_path, scores_filename='All_scores.csv', hparams_filename='hparams.yml', sort_ascending=True):
        self.exp_dir = exp_dir
        self.analysis_dir = os.path.join(exp_dir, 'analysis')
        self.results_dir = os.path.join(exp_dir, 'results')
        
        self.modelnames = modelnames
        self.average_func = average_func
        self.average_fn = average_fn
        self.ignore_hparams = ignore_hparams
        self.log_hparams = log_hparams
        self.results_path = results_path
        self.scores_filename = scores_filename
        self.score = score
        self.target = target
        self.hparams_filename = hparams_filename
        self.sort_ascending = sort_ascending
        
        self.get_all_results()
        
    def get_scorename(self, model, stat_func, score, mode):
        name = f'{score}_{mode}_{model}_{stat_func}'
        return(name)
    
    def get_all_results(self):
        """Get df with overview of all results.
        """
        all_results = {}
        # Initiate column names of scores.
        modes = ['test', 'train']
        self.all_scores = []
        for model, mode in itertools.product(self.modelnames, modes):
            scorename = self.get_scorename(model, self.average_func, self.score, mode)
            self.all_scores.append(scorename)
            all_results[scorename] = []
        
        dirlist = [d for d in os.listdir(self.results_dir) if not (d == os.path.basename(self.analysis_dir) or d == 'Used_scripts' or d == 'Skripte')]
        for d in dirlist:
            d = os.path.join(self.results_dir, d)
            
            # Read in hyperparameters of run
            hparams_file = os.path.join(d, self.hparams_filename)
            print(f'Read hyperparameters from {hparams_file}.')
            hparams = yaml.load(open(hparams_file,"r"), Loader=yaml.FullLoader)
            hparams = {key: val for key, val in hparams.items() if not key in self.ignore_hparams}
            
            # Read in scores of run.
            d = os.path.join(d, self.results_path)
            scores_file = os.path.join(d, self.scores_filename)
            try:
                df_scores = pd.read_csv(scores_file)
            except FileNotFoundError:
                print(f'No results found in {d}.')
                continue
            df_scores = df_scores[df_scores['Target'] == self.target]
            assert len(df_scores) != 0
            
            # Get average score of run between all domains.
            all_av_score_test = []
            all_av_score_train = []
            for model in self.modelnames:
                is_model = df_scores['Model'] == model
                if self.average_func == 'mean':
                    all_av_score_test.append(df_scores.loc[is_model, f'{self.score}_test'].mean())
                    all_av_score_train.append(df_scores.loc[is_model, f'{self.score}_train'].mean())
                elif self.average_func == 'median':
                    all_av_score_test.append(df_scores.loc[is_model, f'{self.score}_test'].median())
                    all_av_score_train.append(df_scores.loc[is_model, f'{self.score}_train'].median())
                else:
                    raise ValueError
            
            # Append scores and hyperparameters.
            for i, model in enumerate(self.modelnames):
                test_scorename = self.get_scorename(model, self.average_func, self.score, 'test')
                train_scorename = self.get_scorename(model, self.average_func, self.score, 'train')
                all_results[test_scorename].append(all_av_score_test[i])
                all_results[train_scorename].append(all_av_score_train[i])
                
            score_dir = os.path.relpath(d, self.results_dir)
            try:
                all_results['directory'].append(score_dir)
                for key, val in hparams.items():
                    all_results[key].append(val)
            except KeyError:
                all_results['directory'] = [score_dir]
                for key, val in hparams.items():
                    all_results[key] = [val]
        
        self.df_all_results = pd.DataFrame(all_results)
        self.df_all_results = self.df_all_results.sort_values(by=self.df_all_results.columns[0], ascending=self.sort_ascending)
        
        # Get variable and constant hyperparameters in different lists.
        self.all_hparams = list(hparams.keys())
        self.constant_hparams = [h for h in self.df_all_results.columns if self.df_all_results[h].nunique() == 1]
        self.var_hparams = [h for h in self.all_hparams if not h in self.constant_hparams]        
        
        self.df = self.get_av_and_std_results(self.df_all_results)
        
        
    def get_av_and_std_results(self, df):
        """Get average and std of results with same hyperparameters and name columns well."""
        score_cols_without_av = [col.rstrip(self.average_func) for col in self.all_scores]
        df = df.rename(columns={col: newcol for col, newcol in zip(self.all_scores, score_cols_without_av)})
        result_dirs = df.groupby(by=self.all_hparams)['directory'].apply(list)
        df_results = df.groupby(by=self.all_hparams)[score_cols_without_av].apply(lambda subdf: self.get_average_and_std(subdf))
        self.all_scores_and_stds = df_results.columns.tolist()
        df_results = df_results.join(result_dirs)
        df_results = df_results.reset_index()
        df_results = self.sort_df(df_results)
        return(df_results)
    
    def get_average_and_std(self, df):
        """Returns average and std of all columns in df."""
        results = pd.Series()
        for col in df.columns:
            av_and_std = pd.Series(data=[self.average_fn(df[col]), df[col].std()], index=[col+self.average_func, col+'std'])
            results = results.append(av_and_std)
        return(results)
    
    def sort_df(self, df):
        """Nicely sort df. For the columns the scores come first, then variable hyperparameters, then constant hyperparameters, then other stuff. The rows are sorted by the first entry of the scores."""
        sort_cols = self.all_scores_and_stds + self.var_hparams + self.constant_hparams
        sort_cols = sort_cols + [col for col in df.columns if not col in sort_cols]
        df_sorted = df[sort_cols]
        df_sorted = df_sorted.sort_values(by=df_sorted.columns[0], ascending=self.sort_ascending)
        return(df_sorted)
                
    def save_df(self, df, filename, comment):
        """Saves a df in a specified filename with a comment."""
        if not os.path.exists(self.analysis_dir):
            os.mkdir(self.analysis_dir)
        outpath = os.path.join(self.analysis_dir, filename)
        write_to_csv(df, outpath, comment)
        
    def save_all_results(self):
        """" Save all results in csv."""
        filename = 'all_results_averaged.csv'
        comment = 'All results averaged for each different combination of hyperparameters.'
        self.save_df(self.df, filename, comment)
        
        filename = 'all_results.csv'
        comment = 'All results raw.'
        self.save_df(self.df_all_results, filename, comment)
    
    def score_correlations(self, df, hparams):
        """Calculate correlation between scores and hyperparameters and sort df nicely.
        """
        df_score_corrs = correlations(df, cols1=self.all_scores, cols2=hparams, colname1='score', colname2='hparams')
        # Sort first by score, then by the abs value of the corr.
        df_score_corrs['score_idx'] = df_score_corrs['score'].apply(lambda score: self.all_scores.index(score))
        df_score_corrs['corr_abs'] = df_score_corrs['corr'].abs()
        df_score_corrs = df_score_corrs.sort_values(by=['score_idx', 'corr_abs'], ascending=[True, False])
        df_score_corrs = df_score_corrs.drop(columns=['score_idx', 'corr_abs'])
        df_score_corrs = df_score_corrs.reset_index(drop=True)
        return(df_score_corrs)
    
    def hparams_correlations(self, df, hparams):
        """Calculate correlation between hyperparameters and sort dataframe by absolute value of correlation.
        """
        df_hparam_corrs = correlations(df, cols1=hparams, cols2=hparams, colname1='hparams1', colname2='hparams2')
        return(df_hparam_corrs)
    
    def plot_hparams_correlations(self, df, df_hparam_corrs, modelnames=[], max_plot_corrs=np.inf):
        """Plots lots of figures for the correlations between the hyperparameters.
        """
        print('Start plotting hyperparameter correlations.')
        if len(modelnames) == 0:            # Use all models.
            modelnames = self.modelnames
            
        for modelname in modelnames:
            save_dir = os.path.join(self.analysis_dir, 'hyperparameter_correlations', modelname)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            score_name = self.get_scorename(modelname, self.average_func, self.score, "test")
            cmap = 'plasma_r' if self.sort_ascending else 'plasma'
            colorbar = {'colname': score_name, 'title': self.score, 'cmap': cmap}
            plot_correlations(
                                df,
                                df_hparam_corrs,
                                save_dir,
                                colorbar=colorbar,
                                plot_log_feats=self.log_hparams,
                                max_plot_corrs=max_plot_corrs
                                )
        print('Finished plotting hyperparameter correlations.')
    
    def plot_score_correlations(self, df, df_score_corrs, modelnames=[], max_plot_corrs=np.inf):
        """Plots lots of figures for the correlations between the scores and hyperparameters.
        """
        print('Start plotting score correlations.')
        if len(modelnames) == 0:            # Use all models.
            modelnames = self.modelnames
            
        for modelname in modelnames:
            save_dir = os.path.join(self.analysis_dir, 'score_correlations', modelname)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_scores = self.get_scores_of_models(self.all_scores, [modelname])
            df_model_score_corrs = df_score_corrs[df_score_corrs['score'].isin(model_scores)]
            plot_correlations(
                                df,
                                df_model_score_corrs,
                                save_dir,
                                plot_log_feats=self.log_hparams,
                                max_plot_corrs=max_plot_corrs,
                                scalarFormatter=[True, False]
                                )
        print('Finished plotting score correlations.')
        
    def get_scores_of_models(self, all_scores, models):
        """Returns all scores that belong to one model in models."""
        model_scores = []
        stats = [self.average_func, 'std']
        modes = ['test', 'train']
        for model, stat, mode in itertools.product(models, stats, modes):
            scorename = self.get_scorename(model, stat, self.score, mode)
            if scorename in all_scores:
                model_scores.append(scorename)
        return(model_scores)



