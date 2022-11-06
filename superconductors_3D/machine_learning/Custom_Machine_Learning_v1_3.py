#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:14:31 2020

@author: timo
This module is for the class MachineLearning that automatically executes a lot of different models and prints and saves all the output.
"""

import warnings
# warnings.filterwarnings("ignore")
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
import sklearn.metrics
import sklearn.linear_model
import random
import copy
import yaml
import datetime
from shutil import copyfile
from superconductors_3D.machine_learning.own_libraries.own_functions import movecol, isfloat
# import torch.optim
import itertools
import superconductors_3D.machine_learning.own_libraries.data.All_Data as All_Data
import superconductors_3D.machine_learning.own_libraries.data.All_scores as All_Scores
from superconductors_3D.machine_learning.own_libraries.data.Domain_statistics import save_Domain_statistics
from superconductors_3D.machine_learning.own_libraries.data import Feature_Importances as FI
from contextlib import redirect_stdout
import json
from superconductors_3D.machine_learning.own_libraries.utils.Models import Models, get_modelpath



def load_df_and_metadata(path):
    """Loads the dataframe and if possible the metadata.
    """
    try:
        with open(path, 'r') as f:
            first_line = f.readline()
            metadata = json.loads(first_line)
        df = pd.read_csv(path, header=1)
        # print(f'Loaded df and metadata from {path}.')
    except json.JSONDecodeError:
        df = pd.read_csv(path)
        metadata = {}
        print(f'Metadata not found. Loaded df from {path}.')
    return(df, metadata)

def save_df_and_metadata(df, metadata, outpath):
    """Save a df as csv with all the important metadata saved as json comment in the first line of the file.
    """
    if os.path.exists(outpath):
        os.remove(outpath)
    with open(outpath, 'a') as f:
        json.dump(metadata, f)
        f.write('\n')
        df.to_csv(f, index=False)
    return

def regressor_from_pipeline(pipe):
    """Returns the ML model from a given sklearn Pipeline or TransformedTargetRegressor.
    """
    return pipe.regressor_['model']

def get_saved_model(modelname: str, repetition: int, run_dir: str, regressor=False):
    """Finds a single saved model with given name and repetition in run_dir/models and returns it.
    """
    model = Models().load(modelname=modelname, repetition=repetition, rundir=run_dir, regressor=regressor)
        
    return(model)
    
def assert_allclose(x1, x2, atol=1e-5, rtol=1e-3):
    """Asserts that arrays x1 and x2 are either equal or at least close.
    """
    try:
        assert np.allclose(x1, x2, rtol, atol)
    except TypeError:
        # If x1 and x2 are e.g. string arrays.
        assert all(x1 == x2)
    return

def net_pattern(n_layers, base_size, end_size):
    """Calculates layer sizes for each layer from first and last layer so that the layer size continously increases/decreases.
    """
    if n_layers != 1:
        factor = (end_size / base_size)**(1/(n_layers - 1))
    else:
        factor = 1
    
    layer_sizes = [int(round(base_size*factor**n)) for n in range(n_layers)]
    return(layer_sizes)
    
    
def unique_sorted(groupcol):
    """Returns a list with the unique group label sorted by frequency. groupcol must be a pandas series."""
    group_occ = groupcol.value_counts().reset_index()
    groups = group_occ['index'].tolist()
    return(groups)

def print_row(values, name, uncertainties=None, width=7, num_width=3, dec=2, delim=' | '):
    """Prints a nicely formatted row. Name is the left-most entry. Values and uncertainties need to be iterables.
    """
    values = np.asanyarray(values)
    if uncertainties != None:
        uncertainties = np.asanyarray(uncertainties)
        assert isinstance(values[0], float) or isinstance(values[0], int)
        num_width = 2
        print_vals = [f'{val:^.{num_width}g}±{unc:^.{num_width}g}' for val, unc in zip(values, uncertainties)]
        print_vals = [f'{string:^{width}.{width}}' for string in print_vals]
    else:
        if isinstance(values[0], str):
            print_vals = [f'{val:^{width}.{width}}' for val in values]
        elif isfloat(values[0]):
            print_vals = [f'{val:^{width}.{num_width}g}' for val in values]
    
    cells = [f'{name:5.5}']
    cells = cells + print_vals
    print(delim.join(cells))
    print('-'*75)
    return()




def Sc_classification(*Tc_arrays):
    """Returns the classification in 0 (non-sc) or 1 (sc) based on the continous value of Tc.
    """
    all_sc_class_arrays = []
    for Tc in Tc_arrays:
        Tc = np.asarray(Tc)
        assert all(Tc >= 0), 'We found a negative Tc!'
        # Reduce prediction of Tc to sc classes, i.e 0 or 1.
        Sc_class = np.where(Tc > 0, 1, 0)
        all_sc_class_arrays.append(Sc_class)
    return all_sc_class_arrays

warnings.simplefilter("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def logr2(y_true, y_pred, sample_weight):
    """Calculates the r2 score after taking the arcsinh (like the log)."""
    y_true = np.arcsinh(y_true)
    y_pred = np.arcsinh(y_pred)
    logr2 = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    return logr2

def Sc_F1(Tc_true, Tc_pred, sample_weight):
    """Calculates the F1 score for a superconductor (with regression of Tc), i.e. before calculating the usual F1 score it sets class=0 if Tc==0 and class=1 if Tc>0.
    """
    sc_true_class, sc_pred_class = Sc_classification(Tc_true, Tc_pred)
    score = sklearn.metrics.f1_score(sc_true_class, sc_pred_class, sample_weight=sample_weight)
    return score

def Sc_accuracy(Tc_true, Tc_pred, sample_weight):
    """Calculates the accuracy score for a superconductor (with regression of Tc), i.e. before calculating the usual accuracy score it sets class=0 if Tc==0 and class=1 if Tc>0.
    """
    sc_true_class, sc_pred_class = Sc_classification(Tc_true, Tc_pred)
    score = sklearn.metrics.accuracy_score(sc_true_class, sc_pred_class, sample_weight=sample_weight)
    return score

def Sc_precision(Tc_true, Tc_pred, sample_weight):
    """Calculates the precision score for a superconductor (with regression of Tc), i.e. before calculating the usual precision score it sets class=0 if Tc==0 and class=1 if Tc>0.
    """
    sc_true_class, sc_pred_class = Sc_classification(Tc_true, Tc_pred)
    score = sklearn.metrics.precision_score(sc_true_class, sc_pred_class, sample_weight=sample_weight)
    return score

def Sc_recall(Tc_true, Tc_pred, sample_weight):
    """Calculates the recall score for a superconductor (with regression of Tc), i.e. before calculating the usual recall score it sets class=0 if Tc==0 and class=1 if Tc>0.
    """
    sc_true_class, sc_pred_class = Sc_classification(Tc_true, Tc_pred)
    score = sklearn.metrics.recall_score(sc_true_class, sc_pred_class, sample_weight=sample_weight)
    return score

def specificity(y_true, y_pred, sample_weight):
    """Calculate the specificity (the recall of the negative class).
    """
    # With sklearn.
    score_with_sklearn = sklearn.metrics.recall_score(y_true, y_pred, pos_label=0, sample_weight=sample_weight)
    # Own implementation.
    # Invert classes.
    y_true = np.abs(y_true - 1)
    y_pred = np.abs(y_pred - 1)
    # Calculate recall of inverted classes.
    score = sklearn.metrics.recall_score(y_true, y_pred, sample_weight=sample_weight)
    # Double checking both implementations.
    assert score == score_with_sklearn
    return score
    
def Sc_specificity(Tc_true, Tc_pred, sample_weight):
    """Calculates the specificity score for a superconductor (with regression of Tc), i.e. before calculating the usual specificity score it sets class=0 if Tc==0 and class=1 if Tc>0.
    """
    sc_true_class, sc_pred_class = Sc_classification(Tc_true, Tc_pred)
    score = specificity(sc_true_class, sc_pred_class, sample_weight=sample_weight)
    return score

def Sc_G_means(Tc_true, Tc_pred, sample_weight):
    """Calculates the G-means score (geometric mean of Recall and specificity) for a superconductor (with regression of Tc), i.e. before calculating the usual specificity score it sets class=0 if Tc==0 and class=1 if Tc>0.
    """
    recall = Sc_recall(Tc_true, Tc_pred, sample_weight=sample_weight)
    spec = Sc_specificity(Tc_true, Tc_pred, sample_weight=sample_weight)
    score = np.sqrt(recall*spec)
    return score

def Sc_OoB(Tc_true, Tc_pred, sample_weight, bound_max=200):
    """Calculates what fraction of data points is Out of Bounds (OoB), i.e. how many are outliers. Doesn't need Tc_true, this is just for the sake of consistency.
    """
    is_OoB = np.isclose(Tc_pred, bound_max, atol=1).astype(int)
    if not (sample_weight is None):
        is_OoB = sample_weight * is_OoB
    score = sum(is_OoB) / len(is_OoB)
    return score

def Sc_MARE(Tc_true, Tc_pred, sample_weight, min_Tc=0):
    """Calculates the mean absolute relative error. Note, this definition is slightly different from the usual MAPE to accomodate for superconductors, e.g. it only looks at data points with a Tc higher than min_Tc.
    """
    Tc_true, Tc_pred = np.asarray(Tc_true), np.asarray(Tc_pred)
    is_sc = (Tc_true > min_Tc) & (Tc_pred > min_Tc)
    Tc_true = Tc_true[is_sc]
    Tc_pred = Tc_pred[is_sc]
    if not (sample_weight is None):
        sample_weight = sample_weight[is_sc]
    
    norm = np.maximum(Tc_true, Tc_pred)
    diff = np.abs(Tc_true - Tc_pred)
    score = diff / norm
    if len(score) > 0:
        score = np.average(score, weights=sample_weight)
    else:
        warnings.warn('The MARE can not be calculated, setting to 0.')
        score = 0
    return score

def Sc_SMAPE(Tc_true, Tc_pred, sample_weight, min_Tc=0):
    """Calculates the mean absolute relative error. Note, this definition is slightly different from the usual MAPE to accomodate for superconductors, e.g. it only looks at data points with a Tc higher than min_Tc.
    """
    Tc_true, Tc_pred = np.asarray(Tc_true), np.asarray(Tc_pred)
    is_sc = (Tc_true > min_Tc) & (Tc_pred > min_Tc)
    Tc_true = Tc_true[is_sc]
    Tc_pred = Tc_pred[is_sc]
    if not (sample_weight is None):
        sample_weight = sample_weight[is_sc]
    
    norm = (Tc_true + Tc_pred)
    diff = np.abs(Tc_true - Tc_pred)
    score = diff / norm
    if len(score) > 0:
        score = np.average(score, weights=sample_weight)
    else:
        warnings.warn('The SMAPE can not be calculated, setting to 0.')
        score = 0
    return score

def out_of_sigma(y_true, y_pred, sigma_lower_bound, sigma_upper_bound):
    """Calculates which fraction of data points is out of the lower and upper uncertainty bounds.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma_lower_bound = np.asarray(sigma_lower_bound)
    sigma_upper_bound = np.asarray(sigma_upper_bound)
    assert all(y_pred >= sigma_lower_bound) and all(y_pred <= sigma_upper_bound), 'Predicions don\'t lie between sigma bounds.'
    
    # Calculate uncertainty for each data point. Use upper bound when overestimating and lower bound when underestimating.
    error = y_pred - y_true
    overestimated = error > 0
    uncertainty_bounds = np.where(overestimated, sigma_lower_bound, sigma_upper_bound)
    uncertainty = np.abs(y_pred - uncertainty_bounds)
    
    error = np.abs(error)
    out_of_sigma = error > uncertainty
    
    # Fraction.
    result = sum(out_of_sigma) / len(out_of_sigma)
    
    return result
    
    
    
    

def name_score_column(target, scorename, CV):
    """Returns the name of the columns of `All_scores.csv`.
    """
    return f'{target}_{scorename}_{CV}'

def get_scores(score, model, target, CV, all_scores_path):
    """Returns scores of specified model and target from the file `all_scores_path` which is an instance of `All_scores.csv`.
    """
    df, _ = load_df_and_metadata(all_scores_path)
    # df = pd.read_csv(all_scores_path)
    df = df[df['Model'] == model]
    colname = name_score_column(target, score, CV)
    scores = df[colname]
    return(scores)


def tolist(*variables):
    results = []
    for var in variables:
        if isinstance(var, str) or isinstance(var, int) or isinstance(var, float):
            var = [var]
        results.append(var)
    return(results)

def inverse_transform_std(mu, std, scaler):
    """Makes the inverse transform of the std by transforming upper and lower bound. Returns upper and lower bound after the inverse transform.
    """
    lower_conf = mu - std
    upper_conf = mu + std
    lower_conf_trans = scaler.inverse_transform(lower_conf)
    upper_conf_trans = scaler.inverse_transform(upper_conf)
    
    return lower_conf_trans, upper_conf_trans





# For file Domain_statistics.csv
N_REPETITION = 'rand_instance'
MODEL = 'Model'
TEST_OR_TRAIN = 'test_or_train'
SIZE = 'Size'

# How many sigma uncertainty should be used for the uncertainty of the targets of data points returned from models (e.g. a Gaussian Process). This does NOT apply to the displayed uncertainty of metrics of models. Throughout all scripts one should refer to this variable.
SIGMA = 2

class Machine_Learning():
    """ This class contains functions to train with data on several models and automatically write the output to several files.
    """
    def __init__(self, data, features, targets, domain=None, sample_weights=None, metrics_sample_weights=None, Column_Transformer=None, save_models=False, is_optimizing_NN=False, save_value_of_variables=[], NN_valfrac=0.2, print_features=True, print_targets=True, print_domain_score=False, random_seed=None, average_func=np.mean, save_all_values_and_predictions=True, save_torch_models=False, n_jobs=1, copy_files=[]):
        # For debugging, to be able to always get the same output.
        self.random_seed = random_seed   # "None" for randomization.
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        df_data = copy.deepcopy(data.reset_index(drop=True))  # Important for make_result_df()
        self.targets = targets
        self.df_data = df_data
        self.n_datapoints = len(self.df_data)
        self.features = features
        self.n_features = len(features)
        self.domain = domain
        self.sample_weights = sample_weights
        self.metrics_sample_weights = metrics_sample_weights
        assert all([feat in self.df_data.columns for feat in self.features]), "One or more of the given features does not exists in the data."
        assert all([target in self.df_data.columns for target in self.targets]), "One or more of the targets does not exist in the data."
        self.n_targets = len(targets)
        
        self.Column_Transformer = Column_Transformer if Column_Transformer != None else {col: StandardScaler() for col in features + targets}
        
        self.CV_cols = [col for col in self.df_data.columns if col.startswith('CV_')]
        self.n_repetitions = len(self.CV_cols)
        self.all_n_repetitions = list(range(self.n_repetitions))
        assert self.CV_cols == [All_Data.All_Data.name_CV_col(i) for i in self.all_n_repetitions], 'Could not recognize CV columns with train and test indices in df_data.'
        self.train_lengths = [sum(self.df_data[col] == 'train') for col in self.CV_cols]
        if len(set(self.train_lengths)) == 1:
            self.train_lengths = self.train_lengths[0]
        self.test_lengths = [sum(self.df_data[col] == 'test') for col in self.CV_cols]
        if len(set(self.test_lengths)) == 1:
            self.test_lengths = self.test_lengths[0]
        
        # Check for NaN features
        nan_feature_cols = [col for col in self.features if self.df_data[col].isna().any()]
        if nan_feature_cols:
            raise ValueError(f'Error: Some of the values of the following features are NaN: {",".join(nan_feature_cols)}')
                            
        self.save_models = save_models
        self.average_func = average_func        
        self.NN_n_reps = 2
        self.NN_valfrac = NN_valfrac
        self.is_optimizing_NN = is_optimizing_NN
        self.save_all_values_and_predictions = save_all_values_and_predictions
        self.save_torch_models = save_torch_models
        self.experiment = save_value_of_variables['experiment']
        self.n_jobs = n_jobs
        self.copy_files = copy_files
                  
        self.all_scores = {"r2": sklearn.metrics.r2_score,
                           "logr2": logr2,
                           "MAE": sklearn.metrics.mean_absolute_error,
                           # "MdAE": sklearn.metrics.median_absolute_error,
                            "MSLE": sklearn.metrics.mean_squared_log_error,
                            # "MARE": Sc_MARE,
                            "SMAPE": Sc_SMAPE,
                            # "OoB": Sc_OoB,
                            # "ScSpec": Sc_specificity,
                            # "ScRecall": Sc_recall,
                            # "ScG": Sc_G_means,
                            # "ScAcc": Sc_accuracy,
                            # "ScPrec": Sc_precision
                           }
        self.all_scorenames = list(self.all_scores.keys())
        
        
        if self.is_optimizing_NN == True:
            self.save_models = False
                
        # Write these variables to the final output file for later convenience.        
        self.store_variables = {}
        self.store_variables["Length of train/ test set"] = (self.train_lengths, self.test_lengths)
        self.store_variables["Features"] = self.features
        self.store_variables["Targets"] = self.targets
        self.store_variables["Save models"] = self.save_models
        self.store_variables["Random seed"] = self.random_seed
        self.store_variables["Average function"] = self.average_func
        self.store_variables["Repetitions per model"] = self.n_repetitions
        self.store_variables["Repetitions for NN"] = self.NN_n_reps   
        self.store_variables["Sigma (Uncertainty)"] = SIGMA
        for varname, value in save_value_of_variables.items():
            self.store_variables[varname] = value
        
        
        # Initiate variables for later data storage.
        self.df_all_score_results = pd.DataFrame()
        self.all_values_and_predictions = self.df_data
        self.all_domain_stats = pd.DataFrame()
        self.all_loss_curves = {}
        
        # Set internal variables.
        self.num_features = len(self.features)
        self.print_features = print_features
        self.print_targets = print_targets
        self.print_domain_score = print_domain_score
        
    
    def get_train_and_test_data(self, CV_col, domain_col=None, sample_weights_col=None, metrics_sample_weights_col=None):
        """Return all train and test data."""
        df = self.df_data
        x = df[self.features].to_numpy()
        y = df[self.targets].to_numpy()
        test = df[CV_col] == 'test'
        train = df[CV_col] == 'train'
        
        x_train, x_test = x[train], x[test]        
        y_train, y_test = y[train], y[test]
        
        if domain_col != None:
            d_train = df[train][domain_col].to_numpy()
            d_test = df[test][domain_col].to_numpy()
        else:
            d_train, d_test = np.zeros(sum(train)), np.zeros(sum(test))
            
        if sample_weights_col != None:
            w_train = df[train][sample_weights_col].to_numpy()
            w_test = df[test][sample_weights_col].to_numpy()
        else:
            w_train, w_test = None, None
        
        if metrics_sample_weights_col != None:
            mw_train = df[train][metrics_sample_weights_col].to_numpy()
            mw_test = df[test][metrics_sample_weights_col].to_numpy()
        else:
            mw_train, mw_test = None, None
            
        return(x, y, x_train, x_test, y_train, y_test, d_train, d_test, w_train, w_test, mw_train, mw_test)
    
    def all_score_results(self, all_train_scores, all_test_scores, modelname):
        """ Write all scores to dataframe.
        """
        df_scores = pd.DataFrame()
        df_scores['Repetition'] = list(range(self.n_repetitions))
        for target_idx, target_name in enumerate(self.targets):
            for score_idx, score_name in enumerate(self.all_scores.keys()):
                train_scorename = name_score_column(target_name, score_name, 'train')
                test_scorename = name_score_column(target_name, score_name, 'test')
                df_scores[train_scorename] = all_train_scores[:,score_idx, target_idx]
                df_scores[test_scorename] = all_test_scores[:,score_idx, target_idx]
        df_scores["Model"] = modelname
        return(df_scores)
    
    def unique_transformer(self):
        """Returns a unique transformer for all targets because sklearn doesn\'t suppourt multiple target scalers.
        """
        all_target_scalers = [self.Column_Transformer[target] for target in self.targets]
        # Check for uniqueness
        for t1, t2 in itertools.product(all_target_scalers, all_target_scalers):
            if not type(t1) == type(t2):
                raise Warning('Sklearn doesn\'t support different scaling of differet targets.')
        # Get unique target scaler.
        target_scaler = all_target_scalers[0]
        return target_scaler
    
    def train_regressor(self, model, x_train, y_train, d_train, w_train):
        """Train a regressor of a model with data and return the regressor.
        """
        if self.sample_weights != None:
            x_train, y_train, d_train, w_train = sklearn.utils.shuffle(x_train, y_train, d_train, w_train)
        else:
            x_train, y_train, d_train = sklearn.utils.shuffle(x_train, y_train, d_train)
        
        # Transform and scale features and targets in a pipeline.
        transform_columns = []
        for idx, colname in enumerate(self.features):
            transformer = self.Column_Transformer[colname]
            entry = (colname, transformer, [idx])
            transform_columns.append(entry)
        feature_transformer = ColumnTransformer(transform_columns)
        pipe = Pipeline([
                            ('ColumnTransformer', feature_transformer),
                            ('model', model)
                        ])

        target_transformer = self.unique_transformer()
        regr = CustomTransformedTargetRegressor(regressor=pipe, transformer=target_transformer)
        
        # Fit with domain and sample weights if possible.
        train_kwargs = {}
        if hasattr(model, 'domain_col'):
            print('Fitting model with specified domains.')
            train_kwargs['model__d_train'] = d_train
        try:
            regr.fit(x_train, y_train, model__sample_weight=w_train, **train_kwargs)
            if self.sample_weights != None:
                print('Fitted model with sample weights.')
        except TypeError:
            if self.sample_weights != None:
                print('Model doesn\'t support sample weights.')
            regr.fit(x_train, y_train, **train_kwargs)
            
        return(regr)

    def apply_model(self, modelname, init_model, outdir):
        """Applies the model to the data to get the r² and MAE score and writes them into the numerical output file.
        """
        # For debugging, to be able to always get the same output.
        # TODO: Remove this shit.
        # np.random.seed(self.random_seed)
        # random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        
        print(f"\n   ###   {modelname}:")
        n_scores = len(self.all_scores)
        all_train_scores = np.zeros((self.n_repetitions, n_scores, self.n_targets))
        all_test_scores  = np.zeros((self.n_repetitions, n_scores, self.n_targets))        
        self.all_loss_curves[modelname] = []
        
        for i in range(self.n_repetitions):
            # Very important to have a new model for each run! Otherwise they might influence each other. 
            # try:
            model = copy.deepcopy(init_model)
            # except TypeError:
            #     print('CAN\'T COPY MODEL!!!')
            #     model = init_model
            
            # Setup SummaryWriter for Tensorboard. # TODO
# =============================================================================
#             if hasattr(model, 'use_tensorboard') and model.use_tensorboard:
#                 tb_dir = os.path.join(outdir, '../0tensorboard', os.path.basename(outdir) + f'_{modelname}_{i}')
#                 model.writer = SummaryWriter(tb_dir)
#                 model.outpath = get_modelpath(outdir, modelname, i) + '.pt'
# =============================================================================
            # Add NN_path to GP to  train on transformed features of model `NN_path`.
            if hasattr(model, 'NN_path') and model.NN_path is not None:
                NN_path = model.NN_path
                if not os.path.exists(NN_path) and NN_path in self.all_models:
                    NN_path = get_modelpath(outdir, NN_path, i) + '.pkl'
                    model.NN_path = NN_path

            # Get train and test data.
            try:
                domain_col = model.domain_col
                assert domain_col == self.domain
            except AttributeError:
                domain_col = None    
            CV_col = All_Data.All_Data.name_CV_col(i)
            x, y, x_train, x_test, y_train, y_test, d_train, d_test, w_train, w_test, mw_train, mw_test = self.get_train_and_test_data(CV_col=CV_col, domain_col=domain_col, sample_weights_col=self.sample_weights, metrics_sample_weights_col=self.metrics_sample_weights)
            
            # Train regressor in pipeline.
            regr = self.train_regressor(model, x_train, y_train, d_train, w_train)
                  
            # Get predictions for scoring.
            y_pred_train = regr.predict(x_train)
            y_pred_test = regr.predict(x_test)
            assert y_train.shape == y_pred_train.shape
            assert y_test.shape == y_pred_test.shape
            
            # train/test_scores have shape (n_scores, n_targets) where n_scores is the number of score functions.
            train_scores = self.scores(y_train, y_pred_train, weights=mw_train)
            test_scores  = self.scores(y_test, y_pred_test, weights=mw_test)
            all_train_scores[i,:,:] = train_scores
            all_test_scores[i,:,:]  = test_scores
            
            # TODO: Make logging with tensorboard or something else.
# =============================================================================
#             if hasattr(model, 'use_tensorboard'):
#                 r2_idx = list(self.all_scores.keys()).index('r2')
#                 # add_hparams only takes these data types.
#                 hparams = {key: val if type(val) in [int, float, str, bool, torch.tensor]  else str(val) for key, val in model.input_args.items()}
#                 model.writer.add_hparams(
#                                         hparams, 
#                                         {'r2_train': train_scores[r2_idx],
#                                           'r2_test': test_scores[r2_idx]
#                                           }
#                                         )
#                 model.writer.flush()
#                 # Pickle can't deal with tensorboard logger.
#                 delattr(model, 'writer')
# =============================================================================
            
            if self.save_models == True:
                Models().save(regr=regr, rundir=outdir, modelname=modelname, repetition=i)
                
                        
            # Construct dataframe to save all true and predicted data together.
            if not self.is_optimizing_NN:
                
                # Get predictions and uncertainty if possible. 
                # The std is a tuple of upper and lower bound because it will usually not be symmetrical due to the scaling.
                # The here called output `y_pred_std` is actually SIGMA * std (SIGMA is a global variable). This is so that one can dynamically change which degreee of uncertainty one wants. This is implemented in CustomTransformedTargetRegressor.
                try:
                    y_pred, y_pred_std = regr.predict(x, return_std=True)
                    scaled_unc, y_pred_std_lower, y_pred_std_upper = y_pred_std
                    assert (y_pred >= y_pred_std_lower).all() and (y_pred <= y_pred_std_upper).all(), 'Prediction not between uncertainty bounds.'
                      
                except TypeError:
                    y_pred = regr.predict(x)
                    
                for idx in range(self.n_targets):
                    target = self.targets[idx]
                    preds = y_pred[:,idx]
                    colname = All_Data.All_Data.name_preds_col(modelname, i, target)
                    self.all_values_and_predictions[colname] = preds
                    
                    # Add uncertainty to df if it exists.
                    try:
                        scaled_unc = scaled_unc[:,idx]
                        std_lower = y_pred_std_lower[:,idx]
                        std_upper = y_pred_std_upper[:,idx]
                        std_lower_colname = All_Data.All_Data.name_unc_col(modelname, i, target, kind='lower')
                        std_upper_colname = All_Data.All_Data.name_unc_col(modelname, i, target, kind='upper')
                        scaled_unc_colname = All_Data.All_Data.name_unc_col(modelname, i, target, kind='scaled_unc')
                        self.all_values_and_predictions[std_lower_colname] = std_lower
                        self.all_values_and_predictions[std_upper_colname] = std_upper
                        self.all_values_and_predictions[scaled_unc_colname] = scaled_unc
                    except UnboundLocalError:
                        pass
            # Some sanity tests due to paranoia.
            assert_allclose(x, self.all_values_and_predictions[self.features].to_numpy())
            assert_allclose(x_train, x[self.all_values_and_predictions[CV_col] == 'train'])
            assert_allclose(y_pred_test, y_pred[self.all_values_and_predictions[CV_col] == 'test'])
                        

        if not self.is_optimizing_NN:
            df_all_scores = self.all_score_results(all_train_scores, all_test_scores, modelname)
            self.df_all_score_results = self.df_all_score_results.append(df_all_scores)
            
        # Calculate averages and standard deviations over all repetitions.
        train_score = self.average_func(all_train_scores, axis=0)
        test_score  = self.average_func(all_test_scores, axis=0)
        train_score_std = np.std(all_train_scores, axis=0)
        test_score_std  = np.std(all_test_scores, axis=0)
        
        self.print_overall_performance_scores(train_score, train_score_std, test_score, test_score_std)
        
        if self.domain != None and self.print_domain_score:
            # Get scores and statistics for each domain
            df_domain_stats = self.get_domain_stats(modelname, self.domain)
            self.all_domain_stats = self.all_domain_stats.append(df_domain_stats)           
            self.print_domain_stats(df_domain_stats, self.domain)
        
        # Write score output to file
        self.save_numerical_output(outdir, modelname, train_score, test_score, train_score_std, test_score_std)          

        return


    def train(self, all_models, outdir):
        print("   ###   Start training")
        self.all_models = all_models
        
        # Make directories for the output.
        outdir = self.prepare_directories(outdir)
        
        # Copy specified files to outdir.
        for file in self.copy_files:
            file_in_new_dir = os.path.join(outdir, file)
            copyfile(file, file_in_new_dir)
        
        # Store all arguments.
        with open(outdir + "/arguments", 'w') as f:            
            yaml.dump(self.store_variables, f)
    
        # Print input information.
        print("   ---   Data:")
        if self.print_features:
            print("   Features: {}".format(self.features))
        if self.print_targets:
            print("   Targets: {}".format(self.targets))
        print("   Num features: {}".format(len(self.features)))
        print("   Train data size: {}".format(self.train_lengths))
        print("   Test data size: {}".format(self.test_lengths))
        print("   Num repetitions: {}".format(self.n_repetitions))
                

# =============================================================================
#               Train and evaluate all given models.
# =============================================================================
        for modelname, model in all_models.items():
            starttime = datetime.datetime.now()
            self.apply_model(modelname, model, outdir)
            duration = datetime.datetime.now() - starttime
            print(f'Training time of {self.n_repetitions} instances of {modelname}:   {duration}')
        
        
        # Save data
        if not self.is_optimizing_NN:
            all_models_names = list(all_models.keys())
            if self.save_all_values_and_predictions:
                All_Data_file = os.path.join(outdir, "All_values_and_predictions.csv")
                All_Data.save_All_Data(df=self.all_values_and_predictions,
                              outpath=All_Data_file,
                              targets=self.targets,
                              n_repetitions=self.n_repetitions,
                              features=self.features,
                              domains=self.domain,
                              models=all_models_names,
                              sample_weights=self.sample_weights,
                              SIGMA=SIGMA,
                              )
            All_scores_file = os.path.join(outdir, "All_scores.csv")
            All_Scores.save_All_scores(df=self.df_all_score_results, 
                            outpath=All_scores_file,
                            targets=self.targets,
                            scorenames=self.all_scorenames,
                            CVs=['train', 'test'],
                            models=all_models_names
                            )
            if len(self.all_domain_stats) > 0:
                Domain_statistics_file = os.path.join(outdir, "Domain_statistics.csv")
                save_Domain_statistics(df=self.all_domain_stats,
                                       outpath=Domain_statistics_file,
                                       domains=self.domain,
                                       targets=self.targets,
                                       scorenames=self.all_scorenames,
                                       models=all_models_names
                                       )
            try:
                # Try to get all feature importances
                if self.save_models:
                    importances = FI.feature_importances_from_models(
                                        rundir=outdir,
                                        features=self.features,
                                        modelnames=all_models_names,
                                        repetitions=self.all_n_repetitions
                                        )
                    # Save feature importances
                    feat_imps_file = os.path.join(outdir, FI.data_name)
                    FI.save_Feature_Importances(
                                                df=importances,
                                                outpath=feat_imps_file,
                                                features=self.features,
                                                modelnames=all_models_names,
                                                repetitions=self.all_n_repetitions
                                                )
                else:
                    print('Could not try to get feature importances because no model saved.')
            except AttributeError:
                pass            # No models with feature importances found.
                
            print("Successfully saved all data.")

        return()
    
    
    def prepare_directories(self, outdir):
        """Make directories for the output."""
        if not os.path.exists(outdir):
            raise ValueError('outdir doesn\'t exist!')
        
        if (not os.path.exists("%s/models"%(outdir))) and self.save_models:
            os.makedirs("%s/models"%(outdir))
  
        return(outdir)
    
    
    def print_overall_performance_scores(self, train_scores, train_scores_std, test_scores, test_scores_std):
        """Prints overall_scores in a nice format. The uncertainty is noted so that it is ± half of the std.
        """
        for target_idx, target_name in enumerate(self.targets):
            print(f'   ---     Target: {target_name}')
            for score_idx, score_name in enumerate(self.all_scores.keys()):
                train_score = train_scores[score_idx][target_idx]
                train_score_std = train_scores_std[score_idx][target_idx]
                test_score = test_scores[score_idx][target_idx]
                test_score_std = test_scores_std[score_idx][target_idx]
                print(f'   ---       {score_name}:\tTraining: {train_score:.3f} ± {train_score_std:.3f}\t\tTesting: {test_score:.3f} ± {test_score_std:.3f}')
        return
    
    def scores(self, all_y_true, all_y_pred, weights=None):
        """Returns an array with shape (num_scores, num_targets) with the scores calculated with the functions in all_scores for every target in the second dimension of all_y_true.
        """
        assert all_y_true.ndim == 2
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        num_targets = all_y_true.shape[1]
        num_data = all_y_true.shape[0]
        if not (weights is None):
            assert len(weights) == num_data
        num_scores = len(self.all_scores)
          
        # Calculate scores
        scores = np.zeros((num_scores, num_targets))
        for score_idx, name in enumerate(self.all_scores.keys()):
            func = self.all_scores[name]
            for target_idx, (y_true, y_pred) in enumerate(zip(all_y_true.T, all_y_pred.T)):
                score = func(y_true, y_pred, sample_weight=weights)
                scores[score_idx, target_idx] = score
        return np.array(scores)
    
    def result_statistics(self, y_true, y_pred, targetname, weights=None):
        """Calculates several statistical properties of y_true and y_pred.
        """
        y_true, y_pred = np.asanyarray(y_true), np.asanyarray(y_pred)
        if not (weights is None):
            assert len(y_true) == len(y_pred) and len(y_pred) == len(weights)
        stats = {}
        for score_name, score_fn in self.all_scores.items():
            score = score_fn(y_true, y_pred, sample_weight=weights)
            stats[f'{score_name}_{targetname}'] = score
        stats[f'mean_{targetname}_true'] = y_true.mean()
        stats[f'std_{targetname}_true'] = y_true.std()
        stats[f'mean_{targetname}_pred'] = y_pred.mean()
        stats[f'std_{targetname}_pred'] = y_pred.std()
        return(stats)
      
    def get_domain_stats(self, modelname, domain_col):
        """Returns a df with the scores for each domain.
        """
        domains = pd.unique(self.all_values_and_predictions[domain_col])
        stats = {}
        counter = 0
        for n, domain in itertools.product(self.all_n_repetitions, domains):
            is_domain = self.all_values_and_predictions[domain_col] == domain
            df_domain = self.all_values_and_predictions[is_domain] 
            CV_col = All_Data.All_Data.name_CV_col(n)
            CVs = sorted(pd.unique(df_domain[CV_col]))
            for CV in CVs:
                df = df_domain[df_domain[CV_col] == CV]
                # Append descriptive data for this row.
                stats[counter] = {
                                N_REPETITION: n,
                                MODEL: modelname,
                                TEST_OR_TRAIN: CV,
                                domain_col: domain
                                }
                # Calculate statistics for each target.
                for target in self.targets:
                    col = All_Data.All_Data.name_preds_col(modelname, n, target)
                    y_pred = df[col]
                    y_true = df[target]
                    weights = None if self.metrics_sample_weights is None else df[self.metrics_sample_weights]
                    interesting_stats = self.result_statistics(y_true, y_pred, target, weights=weights)
                    stats[counter].update(interesting_stats)
                stats[counter].update({SIZE: len(df)})
                counter += 1
        df_domain_stats = pd.DataFrame.from_dict(data=stats, orient='index')
        return(df_domain_stats)
            
    def print_domain_stats(self, df_domain_stats, domain_col):
        """Prints test scores and other stats per domain.
        """
        # Print only test results otherwise it would be too much.
        df_domain_stats = df_domain_stats[df_domain_stats['test_or_train'] == 'test']
        # Average over multiple runs to have the final test score of each domain.
        df_domain_stats = df_domain_stats.groupby([MODEL, domain_col]).mean().reset_index()
        # Print domains with many datapoints first.
        df_domain_stats = df_domain_stats.sort_values(by='Size', ascending=False)
        domains = df_domain_stats[domain_col].tolist()
        
        print('\n')
        print_row(domains, 'Domain')
        # Print scores of targets.
        for score in self.all_scores.keys():
            for target in self.targets:
                score_name = f'{score}_{target}'
                vals = df_domain_stats[score_name]
                print_row(vals, score_name)
        # Print statistics of targets.
        for target in self.targets:
            for end in ['true', 'pred']:
                name = f'{target}_{end}'
                vals = df_domain_stats[f'mean_{target}_{end}'].tolist()
                stds = df_domain_stats[f'std_{target}_{end}'].tolist()
                print_row(vals, name, stds)
    
    def save_numerical_output(self, outdir, modelname, train_scores, test_scores, train_scores_std, test_scores_std):
        """Write formatted output scores to file.
        """
        with open(outdir + "/Numerical results", "a+") as f:
            f.write("\n   ###   {}:\n".format(modelname))
            with redirect_stdout(f):
                self.print_overall_performance_scores(train_scores, train_scores_std, test_scores, test_scores_std)
            f.close()
        return()
                
    def plot_feature_correlations(self, outdir, x, y, x_scaler, y_scaler):
        """Plot some feature correlations."""
        print("Plotting feature correlations...")
        if not os.path.exists("%s/feature_correlation"%(outdir)):
            os.makedirs("%s/feature_correlation"%(outdir))
    
        for idx in range(len(self.features)):
            #print("do best feature %i"%(idx))
            plt.figure()
            feature_name=self.features[idx]
    
            if "pH" not in feature_name and "cross" not in feature_name:
                plt.scatter(x_scaler.inverse_transform(x)[:,idx],y_scaler.inverse_transform(y))
            else:
                xs_unscaled_here=x_scaler.inverse_transform(x)[:,idx]
                xs_set=[xs_unscaled_here[0]]
                for value in xs_unscaled_here:
                    distances=[abs(value-v2) for v2 in xs_set]
                    if max(distances)>0.01:
                        xs_set.append(value)
                indeces=[]
                for v2 in xs_set:
                    indeces.append([])
                    for v_index, value in enumerate(xs_unscaled_here):
                        d=abs(value-v2)
                        if d<0.01:
                            indeces[-1].append(v_index)
    
                #x_min=np.min(x[:,idx])
                #x_max=np.max(x[:,idx])
                #x_mean=0.5*(x_min+x_max)
                #indeces_low=np.where(x[:,idx]<x_mean)[0]
                #indeces_high=np.where(x[:,idx]>x_mean)[0]
    
                for counter,indeces_here in enumerate(indeces):
                    parts=plt.violinplot([y_scaler.inverse_transform(y)[indeces_here].T[0]], positions=[float(xs_set[counter])], vert=True, widths=0.18, showmeans=False, showextrema=True, showmedians=False)
                    for pcidx, pc in enumerate(parts['bodies']):
                        pc.set_facecolor("C%i"%(counter))
                        pc.set_edgecolor('k')
                        pc.set_alpha(0.7)
                    parts['cbars'].set_color("k")
                    parts['cmaxes'].set_color("k")
                    parts['cmins'].set_color("k")
    
                #parts=plt.violinplot([y_scaler.inverse_transform(y)[indeces_high].T[0]], positions=[1.0], vert=True, widths=0.18, showmeans=False, showextrema=True, showmedians=False)
                #for pcidx, pc in enumerate(parts['bodies']):
                #    pc.set_facecolor("C2")
                #    pc.set_edgecolor('k')
                #    pc.set_alpha(0.7)
                #parts['cbars'].set_color("k")
                #parts['cmaxes'].set_color("k")
                #parts['cmins'].set_color("k")
                #if "pH" in feature_name:
                #    plt.xticks(xs_set, ["pH=1","pH=7"])
                #    #plt.xlim([-0.7,1.7])
                #num_off=len(indeces_low)
                #num_on=len(indeces_high)
                #plt.text(0.2,14.0,"%i"%(num_off))
                #plt.text(1.2,14.0,"%i"%(num_on))
    
            # r2 = sklearn.metrics.r2_score(y,x[:,idx])
            #plt.ylim([7.0,13.0])
    
            #plt.xlabel("%s (r$^2$ = %.3f)"%(feature_name, r2), fontname=fontname)
            plt.xlabel("%s"%(feature_name))
            plt.ylabel("Radius difference [px]")
            plt.savefig("%s/feature_correlation/best_20_%s.png"%(outdir, feature_name), dpi=300)
            plt.close()
                
        return()
    
    def save_scatter_plots(self, model_short, y_train, y_test, y_pred_train, y_pred_test, r2_train, r2_test, outdir, colors, index=None):
        """Make scatter plots of measured vs predicted data."""
        plt.figure()
        plt.scatter(y_train, y_pred_train, marker="o", c=colors[0], label="Training: r$^2$ = %.3f"%(r2_train))
        plt.scatter(y_test, y_pred_test, marker="o", c=colors[1], label="Testing: r$^2$ = %.3f"%(r2_test))
        plt.xlabel("Measured normalized radius (%)")
        plt.ylabel("Predicted normalized radius (%)")
        plt.legend(loc="upper left")
        if index != None:
            savepath = "%s/scatter_plots/full_data_%s_%s.png"%(outdir, model_short, index)
        else:
            savepath = "%s/scatter_plots/full_data_%s.png"%(outdir, model_short)
        plt.savefig(savepath,dpi=300)
        plt.close()
        
        return()        
    
    def make_result_df(self, train, test, y_pred_train, y_pred_test, modelname, i, x_train, y_train, x_test, y_test):
        """Returns a dataframe with all the initial data points and the predictions on them.
        """
        # Names of additional columns.
        test_train_colname = 'test_or_train'
        model_colname = 'Model'
        rand_instance_colname = 'rand_instance'
        
        # Get data and assert that it is in the correct order.
        df = self.df_data
        train_index = df.index[train]
        test_index = df.index[test]
        train_entries = df.index.isin(train_index)
        test_entries = df.index.isin(test_index)
        assert np.allclose(df[train_entries][self.features].to_numpy(), x_train)
        assert np.allclose(df[test_entries][self.features].to_numpy(), x_test)
        assert np.allclose(df[train_entries][self.targets].to_numpy(), y_train)
        assert np.allclose(df[test_entries][self.targets].to_numpy(), y_test)
        
        # Get true and predicted target names and rename df accordingly.
        pred_target_names = []
        true_target_names = []
        for target in self.targets:
            df = df.rename(columns={target: target+'_true'})
            true_target_names.append(target+'_true')
            pred_target_names.append(target+'_pred')
        
        # Build dataframes of predictions of test and train data.
        train_pred = pd.DataFrame(
                                    data=y_pred_train,
                                    index=train_index,
                                    columns=pred_target_names
                                    )
        train_pred[test_train_colname] = 'train'
        test_pred = pd.DataFrame(
                                    data=y_pred_test,
                                    index=test_index,
                                    columns=pred_target_names
                                    )
        test_pred[test_train_colname] = 'test'
        
        # Merge predictions and initial data.
        assert df.equals(df.reset_index(drop=True))
        predictions = train_pred.append(test_pred)
        df = df.join(predictions)
        df = movecol(df, cols=pred_target_names, to=true_target_names[-1])
        
        # Add name of used model and random instance.
        df[model_colname] = modelname
        df[rand_instance_colname] = i
        new_cols = [rand_instance_colname, model_colname, test_train_colname]
        column_order =  new_cols + [col for col in df.columns.tolist() if not col in new_cols]
        df = df[column_order]
        return(df)
    
    
    def custom_median(self, array):
        """Calculates the median of an array and make sure to return an element of the array even if it has even length.
        """
        sorted_array = np.sort(array)
        length = len(array)
        if length%2 == 1:
            idx = int(length/2 - 0.5)
        else:
            average = array.mean()
            possible_indices = [int(length/2), int(length/2-1)]
            if abs(sorted_array[possible_indices[0]] - average) <= abs(sorted_array[possible_indices[1]] - average):
                idx = possible_indices[0]
            else:
                idx = possible_indices[1]
        median = sorted_array[idx]
        return(median)
    
    def standard_models(self, hparams, n_features, n_targets, n_domains):
        """Definitions of models that I regularly use. Rather used as a library to quickly look up and copy paste models than as a function.
        """
        pass
# =============================================================================
#         ###################
#         # LINEAR REGRESSION
#         ###################
#         Linear_Regression = sklearn.linear_model.LinearRegression()
#       
#         ################
#         # Neural Network
#         ################
#         # Set some hyperparameter variables for the NN.
#         net_dims = net_pattern(
#                                 hparams['nn_layers'],
#                                 hparams['nn_base_dim'],
#                                 hparams['nn_end_size']
#                                )
#         Neural_Network = MLPRegressor(
#                             hidden_layer_sizes=net_dims,
#                             activation=hparams['nn_act'],
#                             solver="adam",
#                             max_iter=hparams["n_epochs"],
#                             early_stopping=False,
#                             # validation_fraction=0.1,
#                             alpha=0,#hparams["nn_l2"],
#                             # momentum=0,
#                             # learning_rate="invscaling",
#                             learning_rate="constant",
#                             batch_size=hparams['nn_batch_size'],
#                             learning_rate_init=hparams["learning_rate"],
#                             n_iter_no_change=9999#hparams["nn_patience"]
#                              )
#     
#     
#         ###############
#         # Random Forest
#         ###############
#         n_trees = hparams["RF_n_estimators"]
#         Random_Forest = RandomForestRegressor(n_estimators=n_trees)
#     
#     
#         ############################
#         # Gradient Boosting
#         ############################
#         n_trees = hparams["GB_n_estimators"]
#         Gradient_Boosting =  GradientBoostingRegressor(n_estimators=n_trees)
#         
#             
#         ############################
#         # Gaussian Process
#         ############################
#         kernel_scale = np.ones(n_features)
#         noise = hparams["GP_alpha"]
#         Gaussian_Process = GaussianProcessRegressor(kernel=RBF_sk(length_scale=kernel_scale), alpha=noise)
#             
#         
#         ############################
#         # Regret Minimization Network with sklearn
#         ############################        
#         input_layer_size = n_features
#         net_dims = net_pattern(
#                                 hparams['nn_layers'],
#                                 hparams['nn_base_dim'],
#                                 hparams['nn_end_size']
#                                 )
#         module = RGM_sk(input_layer_size, net_dims, n_targets)
#         RGM_sk_model = NeuralNetRegressor(
#                                         module,
#                                         optimizer=torch.optim.Adam,                                                 
#                                         lr=hparams["learning_rate"],
#                                         max_epochs=hparams["n_epochs"],
#                                         batch_size=hparams['nn_batch_size'],
#                                         train_split=None,
#                                         callbacks='disable'
#                                         )
#         
#         
#         def get_featurizer(input_layer_size, hidden_layer_sizes, activation):
#             """Returns the first part of the RGM, the featurizer or representation NN.
#             """
#             layers = []
#             for i in range(len(hidden_layer_sizes)):
#                 if i == 0:
#                     in_size = input_layer_size
#                 else:
#                     in_size = hidden_layer_sizes[i-1]
#                 out_size = hidden_layer_sizes[i]
#                 layers.append(nn.Linear(in_size, out_size))
#                 if activation == 'relu':
#                     activation_fn = nn.ReLU()
#                 else:
#                     raise ValueError('Activation function not known.')
#                 layers.append(activation_fn)
#             layers = tuple(layers)
#             featurizer = nn.Sequential(*layers)
#             return(featurizer)
#         
#         ############################
#         # Regret Minimization Network
#         ############################
#         input_layer_size = n_features
#         net_dims = net_pattern(
#                                 hparams['nn_layers'],
#                                 hparams['nn_base_dim'],
#                                 hparams['nn_end_size']
#                                 )
#         featurizer = get_featurizer(input_layer_size, net_dims, 'relu')
#         
#         # Set options for RGM.
#         RGM_args = namedtuple('RGM_args', ['linear', 'hidden_size', 'output_size', 'num_domains', 'rgm_e', 'erm_e', 'holdout_e', 'detach_classifier', 'oracle'])
#         RGM_args.linear = True
#         RGM_args.hidden_size = net_dims[-1]
#         RGM_args.output_size = n_targets
#         RGM_args.num_domains = n_domains
#         RGM_args.rgm_e = 1
#         RGM_args.erm_e = 1
#         RGM_args.holdout_e = 1
#         RGM_args.detach_classifier = True
#         RGM_args.oracle = False
#         
#         RGM = RGM_sklearn(hidden_layer_sizes=net_dims,
#                             activation='relu',
#                             solver='adam',
#                             max_iter=hparams["n_epochs"],
#                             batch_size=hparams['nn_batch_size'],
#                             learning_rate_init=hparams["learning_rate"],
#                             featurizer=featurizer,
#                             batch_mode='Conserve_ratio',
#                             RGM_args=RGM_args
#                             )
#         
#         all_models = {
#                         "Linear_Regression": Linear_Regression,
#                         "Neural_Network": Neural_Network,
#                         "Random_Forest": Random_Forest,
#                         "Gradient_Boosting": Gradient_Boosting,
#                         "Gaussian_Process": Gaussian_Process,
#                         "RGM_sk": RGM_sk_model,
#                         "RGM": RGM
#             }
#         return(all_models)
# =============================================================================
        

class CustomTransformedTargetRegressor(TransformedTargetRegressor):
    
    def predict(self, X, **predict_params):
        """Predict using the base regressor, applying inverse.
        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.
        Parameters
        """
        check_is_fitted(self)
        pred = self.regressor_.predict(X, **predict_params)
        
        returns_std = 'return_std' in predict_params.keys() and predict_params['return_std'] == True
        
        if returns_std:
            pred, std = pred
            # Use the specified degree of uncertainty throughout all scripts.
            std = std * SIGMA
            assert pred.shape == std.shape, 'Maybe you are trying to fit a GP to several targets or so? This doesn\'t work. Anyway, there\'s a shape mismatch.'
        
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        pred_trans = self.transformer_.inverse_transform(pred)
        
        if returns_std:
            if std.ndim == 1:
                std = std.reshape(-1, 1)
            lower_conf_trans, upper_conf_trans = inverse_transform_std(pred, std, self.transformer_)
            
            # Squeeze if prediction will be squeezed as well.
            if (
                self._training_dim == 1
                and lower_conf_trans.ndim == 2
                and lower_conf_trans.shape[1] == 1
            ):    
                lower_conf_trans = lower_conf_trans.squeeze(axis=1)
                upper_conf_trans = upper_conf_trans.squeeze(axis=1)
            
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)
        
        if returns_std:
            assert len(set([pred_trans.shape, std.shape, lower_conf_trans.shape, upper_conf_trans.shape])) == 1, 'Prediction and uncertainty have different shapes.'
            return pred_trans, (std, lower_conf_trans, upper_conf_trans)
        else:
            return pred_trans



        
        
        
# Ideas for refactoring:
    # Instead of giving n_repetitions, directly give a list with the CV column so that for repeatibility we can have CV columns for several things in the dataset. Write the columns to metadata of All_Data.
    # Make if possible to have different features/ targets per model?
    # Make it possible to calculate scores more flexible and calculate only some scores for each target (and maybe model?).
    # Train loop over all given models:
        # Function: Train all repetitions, save models.
            # Next functions: Either take passed on models (in case model can't be saved like tensorflow model) or read in saved models (for later analysis or if memory is not big enough)
                # Function: Calculate df All_Data.
                # Function: Calculate all train and test scores from All_Data.
                # For plotting function also either pass on all models or read in saved models.
    # cd into the correct directory at the begginning of Apply_ML_models and then cd into previos directory again.
    # In Apply_ML_models add single flag to debug and then set settings accordingly.