#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:23:49 2021

@author: Timo Sommer

This script contains a class to plot stuff for ML runs.
"""

import os
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from superconductors_3D.machine_learning.own_libraries.utils.Scalers import Arcsinh_Scaler
import matplotlib.colors
import sklearn
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter, FormatStrFormatter
import seaborn as sns
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML
from superconductors_3D.machine_learning.own_libraries.data import All_Data
from itertools import product
import warnings
import yaml
import copy
from superconductors_3D.machine_learning.own_libraries.data.All_scores import All_scores
from superconductors_3D.machine_learning.own_libraries.data.Domain_statistics import Domain_statistics
from typing import Union
from superconductors_3D.machine_learning.own_libraries.data import Feature_Importances
from superconductors_3D.dataset_preparation.utils.check_dataset import get_chem_dict
sns.set_theme()
from superconductors_3D.machine_learning.own_libraries.utils.Scores import SMAPE


def unroll_dictionaries(dictionary, sep='__'):
    """Unrolls all dictionaries in the given dictionary.
    """
    unrolled_dictionary = copy.deepcopy(dictionary)
    for key, val in dictionary.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                unrolled_key = key + sep + subkey
                assert not unrolled_key in dictionary
                unrolled_dictionary[unrolled_key] = subval
            del unrolled_dictionary[key]
    return unrolled_dictionary

def disassemble_dict(value):
    if isinstance(value, dict):
        val = list(value.values())[0]
        key = list(value.keys())[0]
    else:
        val = value
        key = value
    return key, val

def color_for_gaussian_datapoints(y_std_train, y_std_test):
    """Set colors for datapoints so that datapoints with higher std have a higher transparency. Only useful for the Gaussian Process.
    """
    color_train=matplotlib.colors.to_rgba("C1")
    colors_train=np.outer(np.ones(len(y_std_train)),np.array(color_train))
    alpha_train=1.0-y_std_train/(np.max(y_std_train)+0.1)
    if np.min(alpha_train)<0.01:
        alpha_train-=np.min(alpha_train)
    if np.max(alpha_train)>=1.0:
        alpha_train/=(np.max(alpha_train)+0.01)
    colors_train[:,-1]=alpha_train.reshape(-1)

    color_test=matplotlib.colors.to_rgba("C2")
    colors_test=np.outer(np.ones(len(y_std_test)),np.array(color_test))
    alpha_test=1.0-y_std_test/(np.max(y_std_test)+0.1)
    if np.min(alpha_test)<0.01:
        alpha_test-=np.min(alpha_test)
    if np.max(alpha_test)>=1.0:
        alpha_test/=(np.max(alpha_test)+0.01)
    colors_test[:,-1]=alpha_test.reshape(-1)
    
    return(colors_train, colors_test)

def is_run_dir(absdir, accept_deprecated=False):
    """Checks if a given directory is an ML run directory.
    """
    has_scores = os.path.exists(os.path.join(absdir, 'All_scores.csv'))
    has_arguments = os.path.exists(os.path.join(absdir, 'arguments'))
    has_hparams = os.path.exists(os.path.join(absdir, 'hparams.yml'))
    
    if has_arguments and has_hparams:
        if has_scores:
            return True
        else:
            warnings.warn(f'Failed ML run in directory {absdir}!')
            
    # old versions
    if accept_deprecated:
        if has_scores and not (has_arguments or has_hparams):
            warnings.warn(f'Old run with deprecated structure found! Accepting old run.')
            return True
        
    return False

def get_hparams(hparams_file):
    """Get hyperparameter from yaml file."""
    if os.path.exists(hparams_file):
        hparams = yaml.load(open(hparams_file,"r"), Loader=yaml.FullLoader)
    else:
        raise ValueError(f'No hyperparameter file "{hparams_file}" found.')
    return(hparams)

def hparams_file(run_dir):
    return os.path.join(run_dir, 'hparams.yml')

def analysis_dir(run_dir):
    return os.path.join(run_dir, 'plots')

def all_scores_file(run_dir):
    return os.path.join(run_dir, 'All_scores.csv')

def params_file(run_dir):
    return os.path.join(run_dir, 'arguments')

class MLRun():
    """This class contains functions to plot ML runs.
    """
    def __init__(self, run_dir, get_arguments=True):
        self.run_dir = run_dir
        self.analysis_dir = analysis_dir(self.run_dir)
        self.all_scores_file = all_scores_file(self.run_dir)
        self.hparams_file = hparams_file(self.run_dir)
        self.params_file = params_file(self.run_dir)
        # self.all_scores = All_scores(self.all_scores_file)
        self.all_Data_path = os.path.join(self.run_dir, 'All_values_and_predictions.csv')
        if get_arguments:
            self.arguments = self.get_params()
    
    def get_hparams(self):
        """Returns a dictionary of the hyperparameters of this run.
        """
        return get_hparams(self.hparams_file)
    
    def get_params(self, unroll=False):
        """Returns a dictionary of the parameters of this run (saved under `arguments`).
        """
        if os.path.exists(self.params_file):
            params = yaml.load(open(self.params_file,"r"), Loader=yaml.Loader)
        else:
            raise ValueError(f'No hyperparameter file "{self.params_file}" found.')
        if unroll:
            params = unroll_dictionaries(params)
        return params
    
    def get_all_params(self, unroll=False):
        """Returns all parameters and hyperparameters as a dictionary.
        """
        hparams = self.get_hparams()
        params = self.get_params()
        all_params = params.update(hparams)
        assert len(all_params) == len(hparams) + len(params), 'There is overlap between the names of parameters and hyperparameters!'
        if unroll:
            all_params = unroll_dictionaries(all_params)
        return all_params
    
    def get_scores(self):
        """Returns the df of all scores.
        """
        all_scores = All_scores(self.all_scores_file)
        return all_scores
        
    def savefig(self, filename, **kwargs):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.mkdir(directory)
        plt.savefig(filename, **kwargs)
        return
    
    def plot_score_of_models(self, models: list, score_name: Union[str,dict], target: str, CVs: list=['test'], save_file=None):
        """Makes bar plots of all given scores of all models. `score_name` can be either a string or a dictionary where the key is the score_name in the csv file and the value is the name in the plot.
        """
        # Allow to override plotted name of score by passing a dict.
        score_name, plot_score_name = disassemble_dict(score_name)
        all_scores = All_scores(self.all_scores_file)
        
        data = []
        for model, CV in product(models, CVs):
            scores = all_scores.get_scores(target, score_name, model, CV)
            for score in scores:
                data.append(
                    {'model': model, 
                     score_name: score,
                     'CV': CV,
                     'target': target
                     })
        data = pd.DataFrame(data)
        
        plt.figure()
        hue = 'CV' if len(CVs) > 1 else None
        sns.catplot(data=data, x='model', y=score_name, hue=hue, kind='bar', legend=False)
        plt.ylabel(plot_score_name)
        plt.legend(loc='best')
        plt.tight_layout()
        if save_file == None:
            save_file = f'scores_{target}_{score_name}_{"+".join(models)}_{"+".join(CVs)}.png'
        filename = os.path.join(self.analysis_dir, 'scores', save_file)
        self.savefig(filename, dpi=300)
        plt.show()
        plt.clf()
        return    
    
    def plot_domain_score_of_model(self, domain_col: str, models: list, score_name: Union[str,dict], target: str, CVs: list=['test'], save_file=None, domain_order=None, rotate_xlabels=None, yscale: str='linear', ylim: tuple=None):
        """Makes a barplot of the domain scores of some models.
        """
        # Allow to override plotted name of score by passing a dict.
        score_name, plot_score_name = disassemble_dict(score_name)
        # Get domain data.
        domain_statistics_file = os.path.join(self.run_dir, 'Domain_statistics.csv')
        domain_stats = Domain_statistics(domain_statistics_file, domain_col)
        domains = domain_stats.all_domains
        # Build up df.
        data = []
        for model, CV, domain in product(models, CVs, domains):
            model, plot_model_name = disassemble_dict(model)
            score = domain_stats.get_score(model, domain, score_name, target, CV)
            data.append({
                        'model': plot_model_name,
                        'group': domain,
                        'target': target,
                        'CV': CV,
                        plot_score_name: score
                })
        data = pd.DataFrame(data)
        plot_models = list(data['model'].unique())
        # Plot.
        plt.figure()
        hue = 'group' if len(domains) > 1 else None
        if domain_order == None:
            domain_order = domains
        sns.catplot(data=data, x='model', y=plot_score_name, hue=hue, hue_order=domain_order, kind='bar', legend=False)
        # Scale
        plt.yscale(yscale)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.ylim(ylim)
        plt.legend(loc='best')
        if rotate_xlabels != None:
            ax.set_xticklabels(rotation=rotate_xlabels, ha="right")
        plt.tight_layout()
        if save_file == None:
            save_file = f'domain_scores_{target}_{score_name}_{"+".join(plot_models)}_{"+".join(CVs)}.png'
        filename = os.path.join(self.analysis_dir, 'domain_scores', save_file)
        self.savefig(filename, dpi=300)
        plt.show()
        plt.clf()
        return  
            

        
    
    def plot_2D_preds(self, model, feature_dict, outpath, x_true=[], y_true=[], target_true=[], scatter_kwargs={}, target_name='target', target_idx=0, res=30):
        """Plots the prediction surface of the target of a model dependent on two features in given ranges. `feature_dict` must have feature names as keys and as value an iterable of [feature_min, feature_max].
        """
        features = list(feature_dict.keys())
        x = features[0]
        y = features[1]
        xmin = feature_dict[x][0]
        xmax = feature_dict[x][1]
        ymin = feature_dict[y][0]
        ymax = feature_dict[y][1]
        
        x_grid = np.outer(np.linspace(xmin, xmax, res), np.ones(res))
        y_grid = np.outer(np.linspace(ymin, ymax, res), np.ones(res)).T 
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        
        feats = np.array([x_flat, y_flat]).T
        # feats = model.x_scaler.transform(feats)
        z_flat = model.predict(feats)
        # z_flat = model.y_scaler.inverse_transform(z_flat)
        if len(z_flat.shape) > 1:
            z_flat = z_flat[:, target_idx]
        z_grid = z_flat.reshape(res, res)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, cmap='plasma', edgecolor='none', alpha=0.3)
        
        if len(x_true) > 0 and len(y_true) > 0 and len(target_true) > 0:
            ax.scatter(x_true, y_true, target_true, **scatter_kwargs)
        ax.set_title('Prediction plot')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(target_name)
        
        plt.savefig(outpath, dpi=300)
        plt.close()


    def plot_1D_preds(self, model_dict, data_dict, outpath, scatter_kwargs={}, y_idx=0, res=300, xlabel='x', ylabel='target', x_limits=[], y_limits=[], add_fn={}):
        """Plots the prediction surface of the target of some models dependent on one feature in given range. With `add_fn` you can add other functions to the plot, it needs the name of the function as key and the function as value.
        """
        plt.figure()
        sns.set_theme()
        
        # Get x for the prediction.
        if x_limits == []:
            xmax = -np.inf
            xmin = np.inf
            for _, (x, _) in data_dict.items():
                if max(x) > xmax:
                    xmax = max(x)
                if min(x) < xmin:
                    xmin = min(x)
        else:
            xmin = x_limits[0]
            xmax = x_limits[1]
        x = np.linspace(xmin, xmax, res)
        x = x.reshape((-1, 1))
            
        # Plot true data points.
        for label, (x_, y_) in data_dict.items():
            plt.plot(x_, y_, label=label, marker='.', linestyle='None', **scatter_kwargs)
        
        # Plot prediction of model.
        for modelname, model in model_dict.items():
            x_pred = x
            y_pred = model.predict(x_pred)
            if len(y_pred.shape) > 1:
                y_pred = y_pred[:, y_idx]
            plt.plot(x_pred, y_pred, label=modelname)
        
        # Plot additional functions.
        for fn_name, f in add_fn.items():
            y = f(x)
            plt.plot(x, y, ':', label=fn_name)
        
        plt.legend()        
        # plt.title('Predictions')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if not y_limits == []:
            plt.ylim(y_limits)
        
        plt.savefig(outpath, dpi=300, transparent=False)
        plt.close()
        
    def reduce_duplicates(self, df, duplicate_col: str, mean_cols: list):
        """Reduces the df so that of duplicates in `duplicate_col` will be taken the mean if all the values are numeric and otherwise it is asserted that it is the same for all the duplicates. Useful for making a df with crystals to a df based on individual superconductors.
        """
        all_columns = df.columns
        n_unique = sum(~ df.duplicated([duplicate_col, 'repetition']))
        groupby_cols = [col for col in df.columns if not col in mean_cols]
        assert duplicate_col in groupby_cols
        df = df.groupby(groupby_cols).mean().reset_index()
        assert len(df) == n_unique
        assert sorted(df.columns) == sorted(all_columns)
        return df
    
    def parity_plot_all_test_data(self, target, model, repetitions, domain_col, duplicate_col=None, log_log=False):
        """This makes a parity plot with all test data from all CV runs of all repetitions.
        """
        true_target = f'true {target}'
        pred_target = f'pred {target}'
        pred_lower_bound = f'${ML.SIGMA} \sigma$ lower bound'
        pred_upper_bound = f'${ML.SIGMA} \sigma$ upper bound'
        rel_pred_lower_bound = f'${ML.SIGMA} \sigma$ lower bound (relative)'
        pred_scaled_unc = f'scaled ${ML.SIGMA} \sigma$'
        uncertainty = f'relative uncertainty'
        
        # When All_Data is finished this should be written to use it.
        data_path = os.path.join(self.run_dir, 'All_values_and_predictions.csv')
        df, _ = ML.load_df_and_metadata(data_path)
        
        other_cols = [duplicate_col] if not duplicate_col == None else []
        data = All_Data.All_Data.get_test_data(df,
                                               target,
                                               model,
                                               repetitions,
                                               domain_col,
                                               other_cols=other_cols,
                                               true_target=true_target,
                                               pred_target=pred_target,
                                               pred_lower_bound=pred_lower_bound,
                                               pred_upper_bound=pred_upper_bound,
                                               pred_scaled_unc=pred_scaled_unc
                                               )
            
        # Reduce data so that we have only one entry per superconductor instead of one entry per crystal and take mean of true/ pred/ unc tc columns.
        if duplicate_col != None:
            data = self.reduce_duplicates(data, duplicate_col, mean_cols=[true_target, pred_target, pred_lower_bound, pred_upper_bound, pred_scaled_unc])
        
        # Sort so that domains with few data points are plotted on top.
        if domain_col != None:
            data['count'] = data.groupby('group')['group'].transform(pd.Series.count)
            data.sort_values('count', inplace=True, ascending=False)
            data = data.drop(columns='count')
            
        # Start plotting.
        if domain_col != None:
            plt.figure(figsize=(8, 4.8))
        else:
            plt.figure()
        
        # To remove title of legend.
        if domain_col != None:
            data = data.rename(columns={'group': ''})
            hue = ''
        else:
            hue = None
        
        # If uncertainty given, use color for uncertainty, not for group anymore.
        has_unc = pred_lower_bound in data.columns
        if has_unc:
            # rel_unc = data[pred_lower_bound] / data[pred_target]
            # data[rel_pred_lower_bound] = rel_unc
            unc = data[pred_scaled_unc] - min(data[pred_scaled_unc])
            unc /= max(unc)
            data[uncertainty] = unc
            norm = plt.Normalize(min(unc), max(unc))
            cmap = sns.cubehelix_palette(as_cmap=True, reverse=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            hue = uncertainty
        else:
            cmap = None
        
        # Plot scatter plot.
        marker_size = 15
        ax = sns.scatterplot(data=data, x=true_target, y=pred_target, hue=hue, alpha=1, palette=cmap, s=marker_size)
        
        # Plot dotted line of perfect fit.
        x_min = min(data[true_target])
        x_max = 200#max(data[true_target])
        line = np.linspace(x_min, x_max, 300)
        ax.plot(line, line, '--k', label='perfect fit')
        
        # Plot title.
        plt.title(model)
        
        # Add legend or colorbar.
        if has_unc:
            ax.get_legend().remove()
            cbar = ax.figure.colorbar(sm)
            cbar.set_label(hue, labelpad=10)
        elif domain_col != None:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        # Ticks.
        if log_log:
            ax.set(xscale='symlog', yscale='symlog')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.12g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.12g'))
            
        # Make labels nicer for tc.
        if target == 'tc':
            plt.ylabel('pred $T_c$ (K)')
            plt.xlabel('true $T_c$ (K)')
            ticks = [0, 0.3, 1, 3, 10, 30, 100, 200]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ticklabels = ['0', '0.3', '1', '3', '10', '30', '100', '200']
            ax.set_xticklabels(ticklabels)
            ax.set_yticklabels(ticklabels)
            
        # Save plot
        plt.tight_layout()
        save_dir = os.path.join(self.run_dir, 'plots/parity_plots')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = os.path.join(save_dir, f'all_together_parity_plot_{target}_{model}.png')
        plt.savefig(save_name, dpi=300)
        plt.show()
        
        return()
    
    def score_over_elemental_prevalence(self, model, target, score, repetitions, domains, chem_formula, duplicate_col, log=False):
        """This makes a parity plot of the uncertainty and the error with all test data from all CV runs of all repetitions.
        """
        true_target = f'true {target}'
        pred_target = f'pred {target}'
        pred_lower_bound = f'${ML.SIGMA} \sigma$ lower bound'
        pred_upper_bound = f'${ML.SIGMA} \sigma$ upper bound'
        pred_scaled_unc = f'${ML.SIGMA} \sigma$ (scaled)'
        
        # Rename x and y axis to make them nicer for Tc.
        if target == 'tc':
            error = f'{score} of $T_c$'
        else:
            error = f'{score} of {target}'
        
        # When All_Data is finished this should be written to use it.
        data_path = os.path.join(self.run_dir, 'All_values_and_predictions.csv')
        df, _ = ML.load_df_and_metadata(data_path)
        
        other_cols = [duplicate_col] if duplicate_col != None else []
        data = All_Data.All_Data.get_test_data(df,
                                               target,
                                               model,
                                               repetitions=repetitions,
                                               domain=domains,
                                               other_cols=other_cols,
                                               true_target=true_target,
                                               pred_target=pred_target,
                                               pred_lower_bound=pred_lower_bound,
                                               pred_upper_bound=pred_upper_bound,
                                               pred_scaled_unc=pred_scaled_unc
                                               )
        

        # Reduce data so that we have only one entry per superconductor instead of one entry per crystal and take mean of true/ pred/ unc tc columns.
        if duplicate_col != None:
            mean_cols=[true_target, pred_target, pred_lower_bound, pred_upper_bound, pred_scaled_unc]
            data = self.reduce_duplicates(data, duplicate_col, mean_cols=mean_cols)
        
        assert score == 'SMAPE'
        data[score] = SMAPE(data[true_target], data[pred_target])
        
        elemental_data = {'element': [], score: []}
        for formula, value in zip(data[chem_formula], data[score]):
            elements = list(get_chem_dict(formula).keys())
            for el in elements:
                elemental_data['element'].append(el)
                elemental_data[score].append(value)
        elemental_data = pd.DataFrame(elemental_data)
        # Take mean per element.
        els = elemental_data.groupby('element')
        df = els.size().reset_index().rename(columns={0: 'occurrences of element'})
        df[score] = list(els[score].mean())
        df['std'] = list(els[score].std())
        
        # Start plotting.
        plt.figure()
       
        # Plot scatter plot.
        ax = plt.errorbar(x=df['occurrences of element'], y=df[score], yerr=df['std'], fmt='.')
        
        # Add axis labels.
        plt.xlabel('occurrences of element')
        plt.ylabel(score)
        
        # Add title.
        plt.title(model)
        
        if log:
            plt.xscale('log')
        
        # # Add legend for quantile bars
        # label = '25%/ 75% quantiles'
        # handles, labels = plt.gca().get_legend_handles_labels()
        # line = Line2D([0], [0], label=label, color='k')
        # handles.extend([line])
        # plt.legend(handles=handles)
        

        # Save plot
        plt.tight_layout()
        save_dir = os.path.join(self.run_dir, 'plots')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = os.path.join(save_dir, f'score_elemental_prevalence_{target}_{model}_{score}.png')
        plt.savefig(save_name, dpi=300)
        plt.show()
        
        return()
        
        return()
    
    def hist_error_over_x(self, x, target, model, repetitions, domain_col, duplicate_col=None, ylim=None, errortype='MdAE', uncertainty='quantiles', log_bars=False):
        """This makes a parity plot of the uncertainty and the error with all test data from all CV runs of all repetitions.
        `uncertainty´: 'quantiles' or 'sem'
        """
        true_target = f'true {target}'
        pred_target = f'pred {target}'
        pred_lower_bound = f'${ML.SIGMA} \sigma$ lower bound'
        pred_upper_bound = f'${ML.SIGMA} \sigma$ upper bound'
        pred_scaled_unc = f'${ML.SIGMA} \sigma$ (scaled)'
        
        # Rename x and y axis to make them nicer for Tc.
        if target == 'tc':
            error = f'{errortype} of $T_c$'
        else:
            error = f'{errortype} of {target}'
        
        # When All_Data is finished this should be written to use it.
        data_path = os.path.join(self.run_dir, 'All_values_and_predictions.csv')
        df, _ = ML.load_df_and_metadata(data_path)
                
        other_cols = [duplicate_col, x] if duplicate_col != None else [x]
        data = All_Data.All_Data.get_test_data(df,
                                               target,
                                               model,
                                               repetitions,
                                               domain_col,
                                               other_cols=other_cols,
                                               true_target=true_target,
                                               pred_target=pred_target,
                                               pred_lower_bound=pred_lower_bound,
                                               pred_upper_bound=pred_upper_bound,
                                               pred_scaled_unc=pred_scaled_unc
                                               )
        
        data = data[(data[true_target] > 0) & (data[pred_target] > 0)]

        # Reduce data so that we have only one entry per superconductor instead of one entry per crystal and take mean of true/ pred/ unc tc columns.
        if duplicate_col != None:
            mean_cols=[true_target, pred_target, pred_lower_bound, pred_upper_bound, pred_scaled_unc, x]
            data = self.reduce_duplicates(data, duplicate_col, mean_cols=mean_cols)
        
        # Sort so that domains with few data points are plotted on top.
        if domain_col != None:
            data['count'] = data.groupby('group')['group'].transform(pd.Series.count)
            data.sort_values('count', inplace=True, ascending=False)
            data = data.drop(columns='count')
        
        # Start plotting.
        if domain_col != None:
            plt.figure(figsize=(8, 4.8))
        else:
            plt.figure()
        
        if log_bars:
            scaler = np.arcsinh
            inv_scaler = np.sinh
        else:
            scaler = lambda x: x
            inv_scaler = lambda x: x
            
        # Get errors to plot.
        if errortype == 'MdAE':
            data[error] = np.abs(data[pred_target] - data[true_target])
            reduce_fn = 'median'
        elif errortype == 'SMAPE':
            data[error] = SMAPE(data[true_target], data[pred_target], multioutput='raw_values')
            reduce_fn = 'mean'
        
        # Bin data because there are too many points for a scatterplot.
        n_bins = 17
        bin_space = scaler(data[x])
        bin_width = (bin_space.max() - bin_space.min()) / (n_bins - 1)
        bins = np.linspace(bin_space.min(), bin_space.max(), n_bins)
        bin_width = inv_scaler(bins + bin_width) - inv_scaler(bins)
        bins = inv_scaler(bins)
        # if target == 'tc':
        #     # Bin to the left for tc==0.
        #     bins = np.insert(bins, 0, - bins[1], axis=0)
        #     bin_width = np.insert(bin_width, 0, bin_width[0], axis=0)
        
        data['binned'] = pd.cut(data[x], bins=bins, include_lowest=True)
        if reduce_fn == 'median':
            data_binned = data.groupby('binned').median()
        elif reduce_fn == 'mean':
            data_binned = data.groupby('binned').mean()
        bars = bins[:-1]
        bin_width = bin_width[:-1]
        
        # Get uncertainties.
        if uncertainty == 'quantiles':
            data_binned['25_quantile'] = data.groupby('binned')[error].quantile(0.25)
            data_binned['75_quantile'] = data.groupby('binned')[error].quantile(0.75)
            yerr = (data_binned[error] - data_binned['25_quantile'], data_binned['75_quantile'] - data_binned[error])
        elif uncertainty == 'sem':
            yerr = data.groupby('binned')[error].apply(stats.sem)
            
        # Plot bar plot.    
        plt.bar(bars, data_binned[error], yerr=yerr, width=bin_width, align='edge', capsize=5)
        
        # Add axis labels.
        convert_label = {'totreldiff': '$\Delta_{totrel}$', 'tc': '$T_c$ (K)'}
        xlabel = convert_label[x] if x in convert_label else x
        if log_bars:
            # plt.xscale('symlog')
            plt.xscale(value='function', functions=(scaler, inv_scaler))
        # Change log-scaled x-axis back into interpretable numbers.
        if target == 'tc' and log_bars:
            ax = plt.gca()
            ticks = np.array([0, 1, 3, 10, 30, 100])
            ax.set_xticks(ticks)
            ticklabels = ['0', '1', '3', '10', '30', '100']
            ax.set_xticklabels(ticklabels)
        plt.xlabel(xlabel)
        plt.ylabel(error)
        
        # Add title.
        plt.title(model)
        
        # Add legend for quantile bars
        labeldict = {'quantiles': '25-75% quantiles', 'sem': 'SEM'}
        label = labeldict[uncertainty]
        handles, labels = plt.gca().get_legend_handles_labels()
        line = Line2D([0], [0], label=label, color='k')
        handles.extend([line])
        plt.legend(handles=handles)
        
        # Add plot limits.
        plt.ylim(ylim)

        # Save plot
        plt.tight_layout()
        save_dir = os.path.join(self.run_dir, 'plots')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = os.path.join(save_dir, f'quality_plot_all_test_data_{target}_{model}_{x}.png')
        plt.savefig(save_name, dpi=300)
        plt.show()
        
        return()
    
    def parity_plot_uncertainty_all_test_data(self, target, model, repetitions, domain_col, duplicate_col=None, log_log=False):
        """This makes a parity plot of the uncertainty and the error with all test data from all CV runs of all repetitions.
        """
        true_target = f'true {target}'
        pred_target = f'pred {target}'
        pred_lower_bound = f'${ML.SIGMA} \sigma$ lower bound'
        pred_upper_bound = f'${ML.SIGMA} \sigma$ upper bound'
        pred_scaled_unc = f'${ML.SIGMA} \sigma$ (scaled)'
        uncertainty = f'relative uncertainty'
        
        # Rename x and y axis to make them nicer for Tc.
        if target == 'tc':
            error = 'error of $T_c$'
            uncertainty = f'${ML.SIGMA} \sigma$ bound of $T_c$'
        else:
            error = f'error of {target}'
            uncertainty = f'${ML.SIGMA} \sigma$ bound of {target}'
        
        # When All_Data is finished this should be written to use it.
        data_path = os.path.join(self.run_dir, 'All_values_and_predictions.csv')
        df, _ = ML.load_df_and_metadata(data_path)
        
        other_cols = [duplicate_col] if duplicate_col != None else []
        data = All_Data.All_Data.get_test_data(df,
                                               target,
                                               model,
                                               repetitions,
                                               domain_col,
                                               other_cols=other_cols,
                                               true_target=true_target,
                                               pred_target=pred_target,
                                               pred_lower_bound=pred_lower_bound,
                                               pred_upper_bound=pred_upper_bound,
                                               pred_scaled_unc=pred_scaled_unc
                                               )
        
        # If this model has no uncertainty given leave.
        has_uncertainty = pred_lower_bound in data.columns and pred_upper_bound in data.columns
        if not has_uncertainty:
            return

        # Reduce data so that we have only one entry per superconductor instead of one entry per crystal and take mean of true/ pred/ unc tc columns.
        if duplicate_col != None:
            data = self.reduce_duplicates(data, duplicate_col, mean_cols=[true_target, pred_target, pred_lower_bound, pred_upper_bound, pred_scaled_unc])
        
        # Sort so that domains with few data points are plotted on top.
        data['count'] = data.groupby('group')['group'].transform(pd.Series.count)
        data.sort_values('count', inplace=True, ascending=False)
        data = data.drop(columns='count')
        
        # Start plotting.
        _ = plt.figure(figsize=(8, 4.8))
        
        # To remove title of legend.
        data = data.rename(columns={'group': ''})
        hue = ''
        
        # Get errors to plot.
        data[error] = data[pred_target] - data[true_target]
        
        # Get uncertainties to plot.
        overestimating = data[error] > 0
        underestimating = data[error] <= 0
        upper_uncertainties = data.loc[underestimating, pred_upper_bound] - data.loc[underestimating, pred_target]
        lower_uncertainties = data.loc[overestimating, pred_target] - data.loc[overestimating, pred_lower_bound]
        data.loc[underestimating, uncertainty] = upper_uncertainties
        data.loc[overestimating, uncertainty] = lower_uncertainties
        
        # Get percentage of data points out of sigma bound.
        out_of_sigma = ML.out_of_sigma(data[true_target], data[pred_target], data[pred_lower_bound], data[pred_upper_bound])
        own_oos = data[error].abs() > data[uncertainty]
        assert sum(own_oos) / len(own_oos) == out_of_sigma, 'Implementations of out_of_sigma don\'t match.'
        
        # For debugging, can be deleted.
        # unc = data[pred_scaled_unc] - min(data[pred_scaled_unc])
        # unc /= max(unc)
        # data['uncertainty'] = unc
        
        # Plot scatter plot.
        ax = sns.scatterplot(data=data, x=uncertainty, y=error, hue=hue, alpha=1)
        
        # Plot dotted line where the error is exactly at the boundary of the uncertainty.
        x_min = 0
        x_max = max(data[uncertainty])
        line = np.linspace(x_min, x_max, 300)
        ax.plot(line, line, '--k', label=f'${ML.SIGMA} \sigma$ boundary')
        ax.plot(line, -line, '--k')
        
        # Plot title.
        out_of_sigma_str = f'Out of ${ML.SIGMA} \sigma$: {out_of_sigma:.2f}'
        title = f'{model} ({out_of_sigma_str})'
        plt.title(title)
        
        # Add legend.
        ax.legend(loc='best')#, bbox_to_anchor=(1, 0.5))
            
        # Ticks.
        if log_log:
            ax.set(xscale='symlog', yscale='symlog')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.12g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.12g'))
            
        # Make labels nicer for tc.
        if target == 'tc' and log_log:
            yticks = [-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100]
            xticks = [0, 1, 3, 10, 30, 100, 200]
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            
            yticklabels = ['-100', '-30', '-10', '-3', '-1', '0', '1', '3', '10', '30', '100']
            xticklabels = ['0', '1', '3', '10', '30', '100', '200']
            ax.set_xticklabels(xticklabels)
            ax.set_yticklabels(yticklabels)
            
        # Save plot
        plt.tight_layout()
        save_dir = os.path.join(self.run_dir, 'plots/parity_plots')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = os.path.join(save_dir, f'all_test_data_uncertainty_parity_plot_{target}_{model}.png')
        plt.savefig(save_name, dpi=300)
        plt.show()
        
        return()
    
    def raw_parity_plot(self, df, true_target, pred_target, hue, style, ax=None, log_log=False):
        """Plots a parity plot of predicted vs true target.
        """
        true_target = df[true_target] if not isinstance(true_target, pd.Series) else true_target
        pred_target = df[pred_target] if not isinstance(pred_target, pd.Series) else pred_target
        if hue != None:
            hue = df[hue] if not isinstance(hue, pd.Series) else hue
            # Setup hue colors to be categorical.
            hue = hue.convert_dtypes().astype(str)
            palette = sns.color_palette(n_colors=hue.nunique())
            hue_order = sorted(hue.unique())
            # Remove legend title because it's annoying.
            hue = hue.rename('')
        else:
            palette = None
            hue_order = None
        style = df[style] if not isinstance(style, pd.Series) else style
        style_order = ['train', 'test']
        # Plot
        ax = sns.scatterplot(data=df, x=true_target, y=pred_target, hue=hue, style=style, style_order=style_order, palette=palette, hue_order=hue_order, ax=ax, alpha=0.8)
        if log_log:
            ax.set(xscale='symlog', yscale='symlog')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.12g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.12g'))
        # Make labels nicer for tc.
        if true_target.name == 'true tc':
            plt.ylabel('pred $T_c$ (K)')
            plt.xlabel('true $T_c$ (K)')
            ticks = [0.1, 0.5, 1, 5, 10, 50, 100]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ticklabels = ['0.1', '0.5', '1', '5', '10', '50', '100']
            ax.set_xticklabels(ticklabels)
            ax.set_yticklabels(ticklabels)
        # Add vertical line of perfect fit.
        x_min = min(true_target)
        x_max = max(true_target)
        line = np.linspace(x_min, x_max, 300)
        ax.plot(line, line, '--k', label='perfect fit')
        ax.legend(loc='upper right')
        return ax
        
    def parity_plot(self, target, model, repetition, hue, duplicate_col=None, log_log=False):
        """Plot parity plots for all given targets, models and repetitions in run_dir.
        """
        # When All_Data is finished this should be written to use it.
        data_path = os.path.join(self.run_dir, 'All_values_and_predictions.csv')
        df, _ = ML.load_df_and_metadata(data_path)
        pred_target_name = All_Data.All_Data.name_preds_col(model, repetition, target)
        CV = All_Data.name_CV_col(repetition)
        # Make df of crystals to df of superconductors.
        used_cols = [target, pred_target_name, CV, duplicate_col, hue]
        df = df[used_cols]
        if duplicate_col != None:
            df = self.reduce_duplicates(df, duplicate_col, mean_cols=[target, pred_target_name])
        # Make that test data points are plotted on top of train data points.
        df = df.sort_values(by=CV, ascending=False)
        plt.figure()
        style = df[CV].rename('')
        true_target = df[target].rename(f'true {target}')
        pred_target = df[pred_target_name].rename(f'pred {target}')
        self.raw_parity_plot(df, true_target, pred_target, hue, style=style, log_log=log_log)
        # Save plot
        save_dir = os.path.join(self.run_dir, 'plots/parity_plots')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = os.path.join(save_dir, f'parity_plot_{target}_{model}_{repetition}.png')
        plt.savefig(save_name, dpi=300)
        plt.clf()
        return

    def get_loss_dfs(self, plot_models, repetitions, run_dir, smoothen=False):
        """Returns dataframes in long and wide format with the losses of the models and repetitions in run_dir. Smoothen means summing over one epoch for the curves with higher resolution than that.
        """
        # Loop through all saved models and get their loss curves.
        loss_curves = {}
        for modelname in plot_models:
            for i in repetitions:
                regr = ML.get_saved_model(modelname, i, run_dir)
                model = regr.regressor_['model']
                # Standardize loss curve to dict.
                if isinstance(model.loss_curve_, list):
                    loss_curves[f'train_{i}'] = model.loss_curve_
                elif isinstance(model.loss_curve_, dict):
                    for key, vals in model.loss_curve_.items():
                        loss_curves[f'{key}_{i}'] = vals
                else:
                    raise ValueError('model.loss_curve_ is neither list nor dict.')
        
        # Get number of epochs for each repetition.
        num_epochs = {}
        for i in repetitions:
            num_epochs[i] = len(loss_curves[f'train_{i}'])
            
        df = pd.DataFrame(columns=['epoch'])
        for curve_name, loss_curve in loss_curves.items():
            df_curve = pd.DataFrame(data=loss_curve, columns=[curve_name])
            df_curve = df_curve.reset_index().rename(columns={'index': 'epoch'})
            df_curve['epoch'] = df_curve['epoch'] + 1
            rep = int(curve_name.split('_')[-1])
            norm = num_epochs[rep] / len(df_curve)
            df_curve['epoch'] = norm * df_curve['epoch']
            df = df.merge(df_curve, on='epoch', how='outer')
        df = df.sort_values(by='epoch')
        
        if smoothen:
            # Sum curves over one epoch if resolution is higher than that.
            df['epoch'] = df['epoch'].apply(np.ceil)
            df = df.groupby(by=['epoch']).sum().reset_index()
        
        df_save = copy.deepcopy(df)
        
        # Prepare df in long format for sns.lineplot.
        df = pd.melt(df, ['epoch'], value_name='loss', var_name='curve')
        df = df[df['loss'].notna()]
        pattern = r'^(.*)_(\d+)$'
        df[['mode', 'repetition']] = df['curve'].str.extract(pattern)
        
        # Extract different metrics
        df.loc[df['mode'] == 'train', 'mode'] = 'loss (train)'
        df.loc[df['mode'] == 'valid', 'mode'] = 'loss (valid)'
        metrics_pattern = r'^(.*) \((.*)\)$'
        df[['metric', 'mode']] = df['mode'].str.extract(metrics_pattern)
        
        df = df.rename(columns={'loss': 'value'})
        
        
        return(df, df_save)
    
    def plot_loss_curves(self, plot_models, repetitions, run_dir, outpath, losses, ax=None, save=True, smoothen=False, scale=False, mean=False):
        """Plot and save loss curves of MLPregressor and torch models. Smoothen means summing over one epoch for the curves with higher resolution than that.
        """
        df, df_save = self.get_loss_dfs(plot_models, repetitions, run_dir)
        
        wanted_metrics = df['metric'].isin(losses)
        assert len(wanted_metrics) > 0, f'Attribute {losses[0]} not found.'
        df = df[wanted_metrics]
       
        # Define style of plot.
        
        if ax == None:
            ax = plt.figure().gca()
        
        modes = df['mode'].unique().tolist()
        dashes = {mode: (2, 2) if mode.startswith('train') else '' for mode in modes}
        
        if mean or len(repetitions) == 1:
            
            if len(wanted_metrics) > 1:
                hue = 'metric'
                style = 'mode'
            else:            
                hue = 'mode'
                style = None
                ax.set_ylabel(wanted_metrics[0])
                
        else:
            assert len(wanted_metrics) == 1
            hue = 'repetition'
            style = 'mode'
            
        # Plot and save loss curves.    
        sns.lineplot(x='epoch', y='value', hue=hue, style=style, dashes=dashes, data=df, ax=ax)
        
        ax.set_title('+'.join(plot_models))
        
        plt.yscale('log')
        if scale:
            max_losses = df.groupby(by=['mode', 'repetition'])['value'].max()
            max_plot_loss = 2*max_losses.median()
            plt.ylim(0, max_plot_loss)
        
        if save:
            plt.savefig(outpath + '.png', dpi=300)
            df_save.to_csv(outpath + '.csv')
        
        plt.show()
        plt.close()
        return()
        

    def plot_grid_loss_curves(self, plot_models, repetitions, run_dir, outpath, losses):
        """Plots a grid plot of all loss curves.
        """
        # Get number of images in rows and columns (height and width).
        num_models = len(plot_models)
        if num_models <= 2:
            height = num_models
            width = 1
        else: 
            height = int(np.ceil(num_models / 2))
            width = 2        
        fig, axes = plt.subplots(height, width, gridspec_kw=dict(hspace=0.3), figsize=(12,9), sharex=True, squeeze=False)
        
        # Plot figures in a grid.
        idx = -1
        for w in range(width):
            for h in range(height):
               idx += 1
               if idx >= num_models:
                   continue
               plot_model = plot_models[idx]
               ax = axes[h][w]
               self.plot_loss_curves([plot_model], repetitions, run_dir, outpath='', losses=losses, ax=ax, save=False)
               # Add ticks to shared axes.
               ax.xaxis.set_tick_params(labelbottom=True)
               
        # Save plot.
        plt.savefig(outpath, dpi=300)
        plt.close('all')
        return()
        

    def final_plots(self, plot_dir, plot_models, df_data, domain_colname, features, targets, use_models, outdir):
        """Do some final plots of your models.
        """
        self.plot_dir = plot_dir
        n_repetitions = len([col for col in df_data.columns if col.startswith('CV_')])
        repetitions = list(range(n_repetitions))
        # repetitions = list(range(len(train_indices)))
        print(f'Plot some stuff in {self.plot_dir}...')
        
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
            
        # # Plot grid figure for loss curve.
        # outpath = os.path.join(self.plot_dir, 'loss_all_models.png')
        # losses = ['train', 'valid', 'train2', 'valid2']
        # if plot_models:
        #     try:
        #         self.plot_grid_loss_curves(plot_models, repetitions, self.run_dir, outpath, losses)
        #     except AttributeError:
        #         warnings.warn('Could not make a grid plot of loss curves because one of the specified models did not have the attribute `loss_curve_`!')
        
        # losses = ['loss', 'mse', 'mae']
        # for plot_model in plot_models:
        #     # Plot individual figures for curve.
        #     outpath = os.path.join(self.plot_dir, f'loss_{"+".join(plot_models)}')
        #     self.plot_loss_curves([plot_model], repetitions, self.run_dir, outpath, losses)
            
        # Plot individual loss curves with mean and standard deviation.
        losses = ['loss', 'mse']
        for plot_model in plot_models:
            outpath = os.path.join(self.plot_dir, f'mean_{"+".join(losses)}_{"+".join(plot_models)}')
            self.plot_loss_curves([plot_model], repetitions, self.run_dir, outpath, losses, mean=True)
        
        
        # # Plot figures for each of the minor loss curves.
        # losses = ['extrapol', 'oracle', 'erm', 'holdout', 'eff_regret', 'regret', 'total', 'eff_loss', 'rep_loss']
        # for plot_model in ['RGM']:
        #     if not plot_model in plot_models:
        #         continue    # if model doesn't exist
        #     for i in repetitions:
        #         outpath = os.path.join(self.plot_dir, f'minor_loss_RGM_{i}')
        #         self.plot_loss_curves([plot_model], [i], self.run_dir, outpath, losses)
        
        # # Plot figures for the norm of the gradient.
        # losses = ['grad_norm_before', 'grad_norm_clipped']
        # for plot_model in ['RGM']:
        #     if not plot_model in plot_models:
        #         continue    # if model doesn't exist
        #     for i in repetitions:
        #         outpath = os.path.join(self.plot_dir, f'norm_grad_RGM_{i}')
        #         self.plot_loss_curves([plot_model], [i], self.run_dir, outpath, losses)
        
        # Plot backwards graphs.
        for modelname in use_models.keys():
            repetition = 0
            model = ML.get_saved_model(modelname, repetition, self.run_dir)
            try:
                for i, graph in enumerate(model.backward_graphs):
                    filename = f'Backward_{modelname}_{repetition}_{i}'
                    graph.render(filename, self.plot_dir, cleanup=True)
            except AttributeError:
                pass
        
        # Plot prediction surface if features are 2D.
        try:
            for modelname in use_models.keys():
                for repetition in repetitions:
                    model = ML.get_saved_model(modelname, repetition, self.run_dir)    
                    feature_dict = {'x': (-1.5, 1.5), 'y': (-1.5, 1.5)}
                    x_true = df_data['x_0'].to_numpy()
                    y_true = df_data['x_1'].to_numpy()
                    target_true = df_data['target'].to_numpy()
                    outpath = os.path.join(self.plot_dir, f'Preds_surface_{modelname}_{repetition}.png')
                    
                    # Get color per domain.
                    domains = df_data[domain_colname].to_numpy()
                    if len(domains) > 0:
                        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        color_dict = {d: cycle[i] for i, d in enumerate(np.unique(domains))}
                        colors = [color_dict[d] for d in domains]
                    else:
                        colors = None
                    scatter_kwargs = {'c': colors}
                    
                    self.plot_2D_preds(model, feature_dict, outpath, x_true, y_true, target_true, scatter_kwargs)
        except KeyError:
            pass
        
        # Plot prediction line if features are 1D.
        models = [model for model in use_models.keys() if not (model == 'LR')]
        plot_1D_features = len(features) == 1 and is_numeric_dtype(df_data[features])
        if plot_1D_features:
            try:
                x_true = df_data[features].to_numpy()
                target_true = df_data[targets[0]].to_numpy()            
                for repetition in repetitions:
                    model_dict = {}
                    for modelname in models:
                        model_dict[modelname] = ML.get_saved_model(modelname, repetition, self.run_dir)
                        
                    # Get test and train feature and target for this repetition.
                    CV_col = All_Data.All_Data.name_CV_col(repetition)
                    train_indices = df_data[CV_col] == 'train'
                    test_indices = df_data[CV_col] == 'test'
                    x_train = x_true[train_indices]
                    target_train = target_true[train_indices]
                    x_test = x_true[test_indices]
                    target_test = target_true[test_indices]
                    
                    data_dict = {
                                'train data': (x_train, target_train),
                                'test data': (x_test, target_test),
                                }
                    
                    scatter_kwargs = {'markersize': 5}
                    outpath = os.path.join(self.plot_dir, f'Preds_1D_line_{modelname}_{repetition}.png')
                    
                    # x_limits = [-2, 2]
                    # y_limits = [-1.5, 1.5]
                    # add_fn={'Cbrt': np.cbrt}
                    x_limits = []
                    y_limits = []
                    add_fn = {}
                    
                    self.plot_1D_preds(model_dict, data_dict, outpath, scatter_kwargs, x_limits=x_limits, y_limits=y_limits, add_fn=add_fn)
            except KeyError:
                pass
        
        # Parity plot for all test data.
        plot_models = use_models.keys()
        duplicate_col = 'formula_sc' if 'formula_sc' in df_data.columns else None
        log_log = True
        for target, model in product(targets, plot_models):
            self.parity_plot_all_test_data(target, model, repetitions, domain_colname, duplicate_col=duplicate_col, log_log=log_log)
            
        # Plot uncertainty parity plot.
        plot_models = use_models.keys()
        duplicate_col = 'formula_sc' if 'formula_sc' in df_data.columns else None
        log_log = True
        # The function itself checks whether the model has uncertainty and if not it returns.
        for target, model in product(targets, plot_models):
            self.parity_plot_uncertainty_all_test_data(target, model, repetitions, domain_colname, duplicate_col=duplicate_col, log_log=log_log)
        
        # Plot error over target distribution.
        ylim = (0, 1)
        duplicate_col = 'formula_sc' if 'formula_sc' in df_data.columns else None
        for target, model in product(targets, plot_models):
            x = target
            log_bars = True if target == 'tc' else False
            self.hist_error_over_x(x, target, model, repetitions, domain_colname, duplicate_col=duplicate_col, log_bars=log_bars, ylim=ylim, errortype='SMAPE', uncertainty='sem')
        
        # Plot quality plot with totreldiff.
        x = 'totreldiff'
        ylim = (0, None)
        duplicate_col = 'formula_sc' if 'formula_sc' in df_data.columns else None
        if x in df_data: 
            varying_quality = sum(df_data[x].unique()) > 1
            if varying_quality:
                for target, model in product(targets, plot_models):
                    self.hist_error_over_x(x, target, model, repetitions, domain_colname, duplicate_col=duplicate_col, ylim=ylim, errortype='SMAPE', uncertainty='sem')
        else:
            print('Can\'t plot dataset quality plot.')
        
        # Plot prediction error over elemental prevalence.
        score = 'SMAPE'
        chem_formula = 'formula_sc'
        duplicate_col = 'formula_sc' if 'formula_sc' in df_data.columns else None
        log = True
        if chem_formula in df_data.columns:
            for target, modelname in product(targets, plot_models):
                self.score_over_elemental_prevalence(model, target, score, repetitions, domain_colname, chem_formula, duplicate_col, log)
        
        # Plot feature importances.
        for target, modelname in product(targets, plot_models):
            all_importances = []
            for repetition in repetitions:
                model = ML.get_saved_model(modelname, repetition, self.run_dir).regressor_['model']
                
                try:
                    importances = Feature_Importances.get_feature_importances(model)
                except AttributeError:
                    continue
                all_importances.append(importances)
            
            if len(all_importances) > 0:
                importances = np.mean(all_importances, axis=0)
            
                feat_dir = os.path.join(self.run_dir, 'plots', 'feature_importances')
                if not os.path.exists(feat_dir):
                    os.mkdir(feat_dir)
                outpath = os.path.join(feat_dir, f'{modelname}_{target}_feature_importances')
                
                Feature_Importances.plot_feature_importances(importances, features, outpath)
                
        print('Finished with plotting.')
        return()
