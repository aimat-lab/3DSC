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
from scipy import stats
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns; sns.set_theme()
from .utils import correlations
from .plot_utils import bool2string, plot_correlations
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
from copy import deepcopy
from superconductors_3D.machine_learning.own_libraries.data.All_scores import All_scores
from superconductors_3D.machine_learning.own_libraries.data.All_Data import All_Data
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Run import is_run_dir, MLRun



default_log_hparams = ['RGM_erm_e', 'RGM_holdout_e', 'RGM_rgm_e', 'coeff_lr_classifier', 'learning_rate', 'nn_l2', 'NN_clip_grad']

default_ignore_hparams = ['GB_n_estimators', 'GP_alpha', 'RF_n_estimators', 'lr_power_t', 'lr_scheduler_factor', 'nn_patience', 'random_seed']


def update_carefully(dict1, dict2):
    dict1 = deepcopy(dict1)
    dict2 = deepcopy(dict2)
    both = dict1.update(dict2)
    assert len(both) == len(dict1) + len(dict2), 'There is an overlap of keys in the dictionaries!'

def get_all_run_dirs_in_tree(source_dir, is_run_dir=is_run_dir):
    """Returns all directories containing an ML run in the directory tree of `source_dir`.
    """
    run_dirs = []
    for abs_dirname, _, _ in os.walk(source_dir):
        if is_run_dir(abs_dirname):
            run_dirs.append(abs_dirname)
    return run_dirs

def av_and_error_df(df, groupby, columns, average, error):
    """Returns a df with average and error columns. When doing the average, the data is grouped by `groupby` and the average and error is taken over all columns `columns` in the df.
    """
    df_av = pd.DataFrame()
    for col in columns:
        av_and_error = lambda subdf: pd.Series([average(subdf[col]), error(subdf[col])], [f'{col}', f'error {col}'])
        data = df.groupby(groupby).apply(av_and_error)
        df_av = pd.concat((df_av, data), axis=1)
    
    return df_av.reset_index()

def change_xticklabels(label_dict):
    """Changes the labels of the x axis based on the replacemant specified in label_dict.
    """
    ax = plt.gca()
    ticks = ax.get_xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [label_dict[l] if l in label_dict else l for l in labels]
    plt.xticks(ticks, labels)

def results_barplot(data, features, score, savepath, plot_CVs: list=['test'], xlabel=None, ylabel=None, title=None, ylim=None, xticklabels={}, huelabels={}, baseline=None, yscale='linear'):
    """Makes a bar plot of the results of an experiment.
    """
    data = wide_score_df_to_long(data, score)
    errname = f'error {score}'

    assert not isinstance(features, str), '`features` must be a list.'
    if len(features) == 1:
        feature = features[0]
        sub_feature = None
    else:
        feature = features[0]
        sub_feature = features[1]
        # raise NotImplementedError('Multiple features should be plotted in groups, but this is not yet implemented.')
    
    # If both train and test score should be plotted, make this the sub_feature.
    if 'train' in plot_CVs and 'test' in plot_CVs:
        if sub_feature is None:
            sub_feature = 'CV'
        else:
            raise ValueError(f'Can only plot one sub_feature, either test/train or {sub_feature}.')
    elif plot_CVs == ['test']:
        data = data.loc[data['CV'] == 'test']
    elif plot_CVs == ['train']:
        data = data.loc[data['CV'] == 'train']
    else:
        raise ValueError(f'Unknown value for plot_CVs {plot_CVs}.')
    
    # Get baseline to plot as horizontal line
    if not baseline is None:
        baseline_results = data.loc[data[feature] == baseline].iloc[0]
        # Remove baseline from plotted data
        data = data.drop(index=baseline_results.name)
        
        if not len(plot_CVs) == 1:
            raise NotImplementedError('More than 1 baseline not implemented.')
            
        baseline_value = baseline_results[score]
        
        
    plt.figure()
    if sub_feature is None:
        cats = data[feature].replace(xticklabels)
        colors = sns.color_palette()[:len(data)]
        ax = plt.bar(cats, data[score], yerr=data[errname], color=colors, capsize=10, label=feature)
    else:
        # Plot bars in groups with distance between them. Each group has the same `feature` and the individual bars in each group are the `sub_features`.
        cats = data[feature].replace(xticklabels).unique()
        x = np.arange(len(cats))
        total_width = 0.7
        unique_sub_features = data[sub_feature].unique()
        n_subs = len(unique_sub_features)
        width = total_width / n_subs
        colors = sns.color_palette()[:n_subs]
        outer_edge_width = width * (n_subs-1)/2
        sub_locs = np.linspace(-outer_edge_width, outer_edge_width, n_subs)
        locations = [x + sub_loc for sub_loc in sub_locs]
        # Plot for each group the left, then the second left (etc) bar.
        for val, xloc, color in zip(unique_sub_features, locations, colors):
            y = data.loc[data[sub_feature] == val, score]
            yerr =  data.loc[data[sub_feature] == val, errname]
            label = huelabels[val] if val in huelabels else val # adapt label of legend
            plt.bar(xloc, y, width, yerr=yerr, color=color, capsize=10, label=label , align='center')
        
        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(cats)
        plt.legend()
    
    if not baseline is None:
        plt.axhline(y=baseline_value, linewidth=1, linestyle='--', color='k')
    

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.yscale(yscale)
    plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()

def wide_score_df_to_long(data, score):
    """Makes a df with test and train score as two columns into a dataframe that is twice as long with one score column and one column indicating if it is test or train.
    """
    train_name = f'{score} (train)'
    test_name = f'{score} (test)'
    # Rename train score column, add CV=train column and drop test score column.
    train_data = deepcopy(data)
    train_data['CV'] = 'train'
    train_data = train_data.rename(columns={
                                    f'{score} (train)': score,
                                    f'error {score} (train)': f'error {score}'
                                            })
    train_data = train_data.drop(columns=[f'{score} (test)', f'error {score} (test)'])
    # Same for test.
    test_data = deepcopy(data)
    test_data['CV'] = 'test'
    test_data = test_data.rename(columns={
                                    f'{score} (test)': score,
                                    f'error {score} (test)': f'error {score}'
                                          })
    test_data = test_data.drop(columns=[f'{score} (train)', f'error {score} (train)'])
    
    wide_df = test_data.append(train_data, ignore_index=True, verify_integrity=True)
    
    # Sanity check.
    diff_cols = ['CV', f'{score} (train)', f'error {score} (train)', f'{score} (test)', f'error {score} (test)']
    should_be_same_columns = [col for col in data.columns if not col in diff_cols]
    pd.testing.assert_frame_equal(train_data[should_be_same_columns], test_data[should_be_same_columns])
    
    return wide_df
    
    
    
def results_catplot(data, features, score, savepath, plot_CVs: list=['test'], xlabel='default', ylabel='default', title=None, ylim=None, mode='strip', plot_av=False, plot_unc=False, order=None, xticklabels={}, legend=False, err_bars=False, log_x=False, log_y=False, xlim=None):
    """Makes a scatter plot of the results of an experiment. 
    
    `features`: list of column names where the first entry indicates the  categorical variable to plot. If it also has a second entry, this is used as color. 
    `mode` can be strip or swarm.
    `mean_and_sem`: plot the mean and the standard error of the mean for each categorical variable.
    `xticklabels`: Dictionary to change the labels of the xticks.
    """
    # Expand features in categorical variable and hue.
    assert not isinstance(features, str), '`features` must be a list.'
    if len(features) == 1:
        cat = features[0]
        hue = None
    elif len(features) == 2:
        cat =  features[0]
        hue = features[1]
    else:
        raise ValueError('The length of `features` can at most be 2 in a swarmplot.')
        
    data = wide_score_df_to_long(data, score)
    data = data[data['CV'].isin(plot_CVs)]
    
    if order is None:
        order = data[cat].unique().tolist()
    else:
        assert all([c in order for c in data[cat]]), 'order must have all categorical variables.'
    
    plt.figure()

    
    if mode == 'strip':
        sns.stripplot(data=data, x=cat, y=score, hue=hue, order=order)
    elif mode == 'swarm':
        sns.swarmplot(data=data, x=cat, y=score, hue=hue, order=order)
    elif mode == 'line':
        sns.relplot(data=data, x=cat, y=score, hue=hue, kind='line')
        
    else:
        raise ValueError(f'Unknown mode {mode}')
    
    # Plot averages and errorbars of each categorical variable.
    if plot_av:
        locations = range(len(order))
        means = data.groupby(cat)[score].apply(plot_av)
        # Sort correctly.
        means = [means.loc[c] for c in order]
        if plot_unc:
            unc = data.groupby(cat)[score].apply(plot_unc)
            # Sort correctly
            unc = [unc.loc[c] for c in order]
        else:
            unc = None
        # zorder so that errorbars are plotted on top of the data points.
        plt.errorbar(x=locations, y=means, yerr=unc, capsize=4, zorder=100)
    
    # Change xticklabels using the dictionary `xticklabels`.
    if xticklabels:
        change_xticklabels(xticklabels)

    plt.title(title)
    if not xlabel == 'default':
        plt.xlabel(xlabel)
    if not ylabel == 'default':
        plt.ylabel(ylabel)
    if legend == False:
        plt.legend([],[], frameon=False)
    
    if log_x == True:
        plt.xscale('log')
    if log_y == True:
        plt.yscale('log')
    
    plt.ylim(ylim)
    plt.xlim(xlim)
        
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()
    
    return



class Experiment():
    """A class for analysing an experiment which consists of several ML runs with different parameters.
    """
    def __init__(self, exp_dir=None, results_dir=None, plot_dir=None, run_dirs=[], check_same_train_test_splits=False):
        self.exp_dir = exp_dir
        self.plot_dir = plot_dir or os.path.join(self.exp_dir, 'plots')
        self.results_dir = results_dir or os.path.join(self.exp_dir, 'results')
        self.run_dirs = run_dirs or get_all_run_dirs_in_tree(self.results_dir)
        if not check_same_train_test_splits == False:
            self.check_for_same_train_test_splits(check_same_train_test_splits)
        
    def get_run_dirs_and_params(self):
        """Returns a dictionary of dictionaries with the first dict specifying the run_dirs and the second specifying the params of each run_dir.
        """
        self.run_dirs_and_params = {run_dir: MLRun(run_dir).get_params(unroll=True) for run_dir in self.run_dirs}
        return self.run_dirs_and_params
    
    def check_for_same_train_test_splits(self, col=None):
        """Checks all run directories if the given train and test splits are equal. If col is True or NaN, this is checked exactly. If col is the name of a columns of the final dataset, this is checked only for unique values of this column.
        """
        if col == True:
            col = None
        
        print('Getting all All_Data dfs of all run directories.')
        for i, run_dir in enumerate(tqdm(self.run_dirs)):
            run = MLRun(run_dir)
            print(f'df path: {run.all_Data_path}')
            usecols = lambda colname: colname.startswith('CV_') or colname == col
            df = pd.read_csv(run.all_Data_path, header=1, usecols=usecols)
            try:
                assert CV_cols == [col for col in df.columns if col.startswith('CV_')]
            except NameError:
                CV_cols = [col for col in df.columns if col.startswith('CV_')]
            
            # Drop duplicates if the column col is given.
            n_unique_cols = df[col].nunique() if not col is None else len(df)
            df = df.drop_duplicates()
            assert len(df) == n_unique_cols, f'Some of the CV columns are  not unique with respect to the column {col}!'
            
            # Sort df so that we can conpare the pure numpy arrays.
            columns = [col] + CV_cols if col is not None else CV_cols
            df = df.sort_values(by=columns).reset_index(drop=True)
            
            # Checking equality of current and first df. This is enough, we don't need to test all combinations because equality is transitive.
            if i == 0:
                first_df = deepcopy(df)
            else:    
                print('Checking current df and first df for equality.')
                pd.testing.assert_frame_equal(df, first_df)
                print(f'dfs {i} and {0} are equal!')
                    
    
    def get_long_df_with_scores_and_params(self, scores, run_dirs=None):
        """Returns a df with parameters and scores of all runs of the experiment.
        """        
        data = []
        run_dirs = run_dirs or self.get_run_dirs_and_params()
        for run_dir, params in run_dirs.items():
            score_file = os.path.join(run_dir, 'All_scores.csv')
            all_scores = All_scores(MLRun(run_dir).all_scores_file)
            models = params['use_models']
            targets = params['Targets']
            for model, target, score in itertools.product(models, targets, scores):
                test_values = all_scores.get_scores(targets=target, scores=score, models=model, CVs='test')
                train_values = all_scores.get_scores(targets=target, scores=score, models=model, CVs='train') 
                
                for test, train in zip(test_values, train_values):                        
                    test_name = score + ' (test)'
                    train_name = score + ' (train)'
                    row = {
                            'model': model, 
                            'target': target,
                            'score': score,
                            'run dir': run_dir,
                            test_name: test,
                            train_name: train
                            }
                    
                    row.update(params)
                    data.append(row)
        
        data = pd.DataFrame(data)
        return data
    
    # def get_av_and_error_scores_and_params(self, models, targets, scores, run_dirs=None, av={'mean': np.mean}, error={'error': stats.sem}, groupby='run dir', mean_cols='scores'):
    #     """Returns the average scores and their error over all columns specified by mean_cols and grouped by all columns specified by groupby. If mean_cols == 'scores' then automatically the mean columns are all scores. If a column is not specified as mean_col and there are more than one unique value these values are put into a list.
    #     """
        
    #     long_df = self.get_long_df_with_scores_and_params(models=models, targets=targets, scores=scores, run_dirs=run_dirs)
    #     av_df = long_df.groupby(groupby)[mean_col]
        
        
    # def get_long_df_of_scores_and_params(self, return_params='all'):
    #     """Returns a dictionary containing the scores for all repetitions of each combination of parameters.
    #     """
    #     for run_dir in self.run_dirs:
    #         run = MLRun(run_dir)
    #         params = run.get_all_params()
    #         scores = All_scores(run.all_scores_file)

    #         return_params = params if return_params == 'all' else return_params

    #         test_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='test')
    #     train_values = scores.get_scores(targets='tc', scores=score, models=model, CVs='train') 
            
            

    
    # init_function of when the Experiment class was mainly for hyperparameters.
    # def __init__(self, exp_dir, modelnames, target, run_dirs=None, score='MAE', average_func='mean', average_fn=np.mean, ignore_hparams=default_ignore_hparams, log_hparams=default_log_hparams, scores_filename='All_scores.csv', hparams_filename='hparams.yml', sort_ascending=True):
    #     self.exp_dir = exp_dir
    #     self.analysis_dir = os.path.join(exp_dir, 'analysis')
    #     self.results_dir = os.path.join(exp_dir, 'results')
    #     # Get all run directories.
    #     if run_dirs == None:
    #         self.run_dirs = [os.path.join(self.results_dir, d) for d in os.listdir(self.results_dir) if not (d == os.path.basename(self.analysis_dir) or d == 'Used_scripts' or d == 'Skripte')]
    #     else:
    #         self.run_dirs = run_dirs
        
    #     self.modelnames = modelnames
    #     self.average_func = average_func
    #     self.average_fn = average_fn
    #     self.ignore_hparams = ignore_hparams
    #     self.log_hparams = log_hparams
    #     self.scores_filename = scores_filename
    #     self.score = score
    #     self.target = target
    #     self.hparams_filename = hparams_filename
    #     self.sort_ascending = sort_ascending
        
    #     self.get_all_results()
        
    def get_scorename(self, model, stat_func, score, mode):
        name = f'{score}_{mode}_{model}_{stat_func}'
        return(name)
    
    def get_hparams(self, path):
        """Read in hyperparameters of run.
        """
        print(f'Read hyperparameters from {path}.')
        hparams = yaml.load(open(path,"r"), Loader=yaml.FullLoader)
        hparams = {key: val for key, val in hparams.items() if not key in self.ignore_hparams}
        return hparams
    


    # def get_all_results(self):
    #     """Get df with overview of all results averaged over the repetitions.
    #     """
    #     all_results = {}
    #     # Initiate column names of scores.
    #     modes = ['test', 'train']
    #     self.all_scores = []
    #     for model, mode in itertools.product(self.modelnames, modes):
    #         scorename = self.get_scorename(model, self.average_func, self.score, mode)
    #         self.all_scores.append(scorename)
    #         all_results[scorename] = []
        
    #     for d in self.run_dirs:            
    #         hparams_file = os.path.join(d, self.hparams_filename)
    #         hparams = self.get_hparams(hparams_file)
            
    #         # Read in scores of run.
    #         scores_file = os.path.join(d, self.scores_filename)
    #         if os.path.exist(scores_file):
    #             all_scores = All_scores(scores_file)
    #         else:
    #             print(f'No results found in {d}.')
    #             continue
            
    #         # Get average score of run between all domains.
    #         all_av_score_test = []
    #         all_av_score_train = []
    #         for model in self.modelnames:
    #             is_model = df_scores['Model'] == model
    #             if self.average_func == 'mean':
    #                 scores
    #                 all_av_score_test.append(df_scores.loc[is_model, f'{self.score}_test'].mean())
    #                 all_av_score_train.append(df_scores.loc[is_model, f'{self.score}_train'].mean())
    #             elif self.average_func == 'median':
    #                 all_av_score_test.append(df_scores.loc[is_model, f'{self.score}_test'].median())
    #                 all_av_score_train.append(df_scores.loc[is_model, f'{self.score}_train'].median())
    #             else:
    #                 raise ValueError
            
    #         # Append scores and hyperparameters.
    #         for i, model in enumerate(self.modelnames):
    #             test_scorename = self.get_scorename(model, self.average_func, self.score, 'test')
    #             train_scorename = self.get_scorename(model, self.average_func, self.score, 'train')
    #             all_results[test_scorename].append(all_av_score_test[i])
    #             all_results[train_scorename].append(all_av_score_train[i])
                
    #         score_dir = os.path.relpath(d, self.results_dir)
    #         try:
    #             all_results['directory'].append(score_dir)
    #             for key, val in hparams.items():
    #                 all_results[key].append(val)
    #         except KeyError:
    #             all_results['directory'] = [score_dir]
    #             for key, val in hparams.items():
    #                 all_results[key] = [val]
        
    #     self.df_all_results = pd.DataFrame(all_results)
    #     self.df_all_results = self.df_all_results.sort_values(by=self.df_all_results.columns[0], ascending=self.sort_ascending)
        
    #     # Get variable and constant hyperparameters in different lists.
    #     self.all_hparams = list(hparams.keys())
    #     self.constant_hparams = [h for h in self.df_all_results.columns if self.df_all_results[h].nunique() == 1]
    #     self.var_hparams = [h for h in self.all_hparams if not h in self.constant_hparams]        
        
    #     self.df = self.get_av_and_std_results(self.df_all_results)
        
        
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



