import warnings
# warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
import os
warnings.simplefilter("ignore", category=FutureWarning)
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
# import gpflow
import pandas as pd
# import torch
# import torch.nn as nn
import sklearn
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, LeaveOneGroupOut, GroupKFold, GroupShuffleSplit
import sklearn.linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import random
import yaml
import datetime
import argparse
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML
# from superconductors_3D.machine_learning.RGM_own import RGM_sklearn
import csv
from collections import namedtuple
import json
from copy import deepcopy
import time
# import mlflow
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Run import MLRun, get_hparams
# from superconductors_3D.machine_learning.own_libraries.models.GNN.MEGNet_tf import MEGNet_tf, read_json_file
# from contextlib import redirect_stdout
# from superconductors_3D.machine_learning.own_libraries.utils import Refactoring
# from superconductors_3D.machine_learning.own_libraries.models.GPflow_GP import GPflow_GP
# from superconductors_3D.machine_learning.own_libraries.own_functions import movecol
from superconductors_3D.utils.projectpaths import projectpath
# from superconductors_3D.machine_learning.own_libraries.models.NN import MLP_Lightning
from superconductors_3D.machine_learning.own_libraries.utils.Scalers import restricted_arcsinh, restricted_sinh, restricted_exp, restricted_log


def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_arguments(args_from_fn, use_models, experiment, output_note, outdirname, calcdir, save_models, CV, n_folds, n_repeats, CV_keep_groups, n_reps, train_frac, domain_colname, is_optimizing_NN, dataset, hparams_file, n_jobs, sample_weights, metrics_sample_weights, use_data_frac, add_params):
    """Parse arguments from command line and if not given from command line insert argument from function."""
    parser = argparse.ArgumentParser(description='Apply different ML models to some data and save output.')
    parser.add_argument('--dataset', '-d', type=str, help='The input dataset.')
    parser.add_argument('--calcdir', '-c', type=str, nargs='?', 
                        help='Change directory to this path.')
    parser.add_argument('--outdir', '-u', dest='outdirname', type=str, nargs='?', 
                        help='Directory of the output.')
    # parser.add_argument('--trainfracs', '-t', dest='all_trainfracs', type=float, nargs='*', help='List of train fractions.')
    parser.add_argument('--experiment', '-e', type=str, help='Identifier for a specific experiment.')
    parser.add_argument('--note', dest='output_note', type=str, help='Output text to add to main output file.')
    parser.add_argument('--n_reps', dest='n_reps', type=str, help='Number of repetitions of train test split if CV is Random.')
    parser.add_argument('--train-frac', dest='train_frac', type=float, help='Train fraction when random CV splitting is used.')
    parser.add_argument('--use-models', '-a', dest='use_models', type=str, nargs='*', help='Names of models that shall be trained.')
    parser.add_argument('--cv', dest='CV', type=str, help='Cross Validation mode.')
    parser.add_argument('--domain-colname', dest='domain_colname', type=none_or_str, help='Name of column in dataset that indicates groups.')
    parser.add_argument('--n_folds', dest='n_folds', type=str, help='Number of folds if CV is KFold.')
    parser.add_argument('--n_repeats', dest='n_repeats', type=str, help='Number of folds if CV is KFold.')
    parser.add_argument('--save-models', dest='save_models', type=bool, help='If models should be saved.')
    parser.add_argument('--optimizing', '-o', dest='is_optimizing_NN', help='If optimizing Neural Network.')
    parser.add_argument('--hparams-file', dest='hparams_file', help='Path to hyperparameter file.')
    parser.add_argument('--n-jobs', dest='n_jobs', help='Number of jobs to run in parallel when possible.')
    parser.add_argument('--sample-weights', dest='sample_weights', type=none_or_str, help='String indicating the weights column for each data point for training the models.')
    parser.add_argument('--metrics-sample-weights', dest='metrics_sample_weights', type=none_or_str, help='String indicating the weights column for each data point only for the metrics.')
    parser.add_argument('--CV-keep-groups', dest='CV_keep_groups', help='Group column name for a KFold with data points of each group in either test or train.')
    parser.add_argument('--use-data-frac', dest='use_data_frac', help='For debugging, use only this fraction of data.')
    parser.add_argument('--add-params', dest='add_params', type=json.loads, help='Add these parameters to the experiment parameters for easier recognition of results.')

    args = argparse.Namespace()     
    
    # Add manually defined arguments to args.
    args.dataset = dataset
    args.experiment = experiment
    args.output_note = output_note
    args.outdirname = outdirname
    args.calcdir = calcdir
    args.n_reps = n_reps
    args.train_frac = train_frac
    args.is_optimizing_NN = is_optimizing_NN
    args.dataset = dataset
    args.use_models = use_models
    args.CV = CV
    args.domain_colname = domain_colname
    args.n_folds = n_folds
    args.n_repeats = n_repeats
    args.save_models = save_models
    args.hparams_file = hparams_file
    args.n_jobs = n_jobs
    args.sample_weights = sample_weights
    args.metrics_sample_weights = metrics_sample_weights
    args.CV_keep_groups = CV_keep_groups
    args.use_data_frac = use_data_frac
    args.add_params = add_params
    
    # args given as cmd line arguments have the second highest priority.
    if __name__ == '__main__':
        cmd_args = parser.parse_args()
        if len(sys.argv) > 1:
           print('Parsed arguments:', cmd_args) 
        args_dict = vars(args)
        cmd_args_dict = {key: val for key, val in vars(cmd_args).items() if not val is None}
        args_dict.update(cmd_args_dict)
        args = argparse.Namespace(**args_dict)
        
    # args given as function arguments of main() have the highest priority.
    if args_from_fn:
        print(f'Updating args with arguments from function call: {args_from_fn}')
        args_dict = vars(args)
        args_dict.update(args_from_fn)
        args = argparse.Namespace(**args_dict)
        print(f'New arguments: {args}')
        
    return(args)

def print_title(string):
    print("""
    ============================
    {}
    ============================""".format(string))
    return()

def is_round(num, prec=6):
    """Checks if a given number is a round value with given precision."""
    is_round = round(num) == round(num, prec)
    return(is_round)

def make_output_directory(outdirname, label):
    """Get numbered output directory `outdir` in parent directory `outdirname`."""
    os.makedirs(outdirname, exist_ok=True)  
    
    # Get number that's not already used for a directory.
    num = 0
    dir_list = os.listdir(outdirname)
    while 'results_{}'.format(num) in '\t'.join(dir_list):
        num += 1
    
    outdir = os.path.join(outdirname, f'results_{num}_{label}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        raise Warning('You should not end up here.')
    return(outdir)

def get_train_test_data(df_data, CV, n_folds, domain_colname, trainfrac=None, random_n_reps=1, n_repeats=1, group=None):
    """Gets train and test data doing the specified cross validation.
    """
    data_array = df_data.to_numpy()
    if CV == 'KFold':
        # Convert train fraction and number of total repetititions to KFold input parameters.
        if group is None:
            split = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats).split(data_array)
        else:
            assert n_repeats == 1, 'This is not yet implemented.'
            groups = df_data[group].to_numpy()
            split = GroupKFold(n_splits=n_folds).split(data_array, groups=groups)
            
    elif CV == 'Random':
        assert trainfrac not in (0,1), "ShuffleSplit won't understand that this is a fraction."
        assert trainfrac != None
        if group is None:
            split = ShuffleSplit(train_size=trainfrac, n_splits=random_n_reps).split(data_array)
        else:
            groups = df_data[group].to_numpy()
            split = GroupShuffleSplit(train_size=trainfrac, n_splits=random_n_reps).split(data_array, groups=groups)
    elif CV == 'LOGO':
        # Work around: LeaveOneGroupOut.split() doesn't seem to work without specified y, probably a bug.
        y = np.zeros(len(df_data))
        domains = df_data[domain_colname]
        split = LeaveOneGroupOut().split(data_array, y, domains)
    
    # Concatenate train and test indices of each repetition.
    # train_indices, test_indices = [], []
    # for train_index, test_index in split:
    #     train_indices.append(list(train_index))
    #     test_indices.append(list(test_index))
    
    for i, (train_indices, test_indices) in enumerate(split):
        n_samples = len(df_data)
        assert n_samples == len(train_indices) + len(test_indices)
        
        empty = ''
        test_or_train = pd.Series(np.full(n_samples, empty))
        test_or_train[train_indices] = 'train'
        test_or_train[test_indices] = 'test'
        # So that I can be sure that in case the indices of df and series don't align it still just adds everything in the right order.
        test_or_train = list(test_or_train)
        
        colname = ML.All_Data.All_Data.name_CV_col(i)
        df_data[colname] = test_or_train
        
        assert all([df_data[colname].iloc[idx] == 'train' for idx in train_indices])
        assert all([df_data[colname].iloc[idx] == 'test' for idx in test_indices])
        assert not (df_data[colname] == empty).any(), 'Some of the columns are neither test nor train.'
    
    return(df_data)

# class Sin(nn.Module):
#     """Sin activation function."""
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input):
#         result = torch.Sin(input)
#         return(result)

# class Multiple_Acts(nn.Module):
#     """Layer with multiple different activation functions.
#     """
#     def __init__(self, acts):
#         """`acts` must have as many activation functions (torch functions) as the length of the input will be.
#         """
#         super().__init__()
#         self.acts = acts

#     def forward(self, input):
#         n_input = len(input)
#         assert n_input == len(self.acts)
#         result = torch.empty(n_input)
#         for i in range(n_input):
#             result[i] = self.acts[i](input[i])
#         result = torch.Sin(input)
#         return(result)
#
# def get_activation_fn(activation: str):
#     """Returns torch activation function based on string activation.
#     """
#     if activation == 'relu':
#         activation_fn = nn.ReLU()
#     elif activation == 'logistic':
#         activation_fn = nn.Sigmoid()
#     elif activation == 'tanh':
#         activation_fn = nn.Tanh()
#     elif activation == 'sin':
#         activation_fn = Sin()
#     else:
#         raise ValueError(f'Activation function {activation} not recognized. Activation functions are lowercase always.')
#     return(activation_fn)
#
#
# def get_sequential_NN(input_layer_size: int, hidden_layer_sizes: list, activation: str, output_layers: str):
#     """Returns a sequential (Feed Forward) NN. `last_linear` means if the last layer should be linear or with activation function."""
#     activation_fn = get_activation_fn(activation)
#     out_act_fn = output_layers
#     layers = []
#     num_layers = len(hidden_layer_sizes)
#     for i in range(num_layers):
#         if i == 0:
#             in_size = input_layer_size
#         else:
#             in_size = hidden_layer_sizes[i-1]
#         out_size = hidden_layer_sizes[i]
#         layers.append(nn.Linear(in_size, out_size))
#         last_layer = i == num_layers - 1
#         if not last_layer:
#             layers.append(activation_fn)
#         elif out_act_fn != None:
#             out_activation_fn = get_activation_fn(out_act_fn)
#             layers.append(out_activation_fn)
#
#     layers = tuple(layers)
#     network = nn.Sequential(*layers)
#     return(network)
#
#
# def get_featurizer(input_layer_size, hparams, mode):
#     """Returns the first part of the RGM, the featurizer or representation NN.
#     """
#     hidden_layer_sizes = ML.net_pattern(
#                                         hparams['nn_layers'],
#                                         hparams['nn_base_dim'],
#                                         hparams['nn_end_dim']
#                                         )
#     activation = hparams['nn_act']
#     # last_linear = False     # if last layer linear or with activation fn
#     output_layers = hparams['nn_act']
#     if mode == 'FeedForward':
#         featurizer = get_sequential_NN(input_layer_size, hidden_layer_sizes, activation, output_layers)
#     else:
#         raise ValueError('mode of Featurizer not recognized.')
#     return(featurizer)
#
#
# def get_classifier(output_layer_size, hparams, output_layers=None):
#     """Returns a torch sequential NN with specific layers and output_layer_size.
#     """
#     if hparams['RGM_classifier_layers'] < 1:
#         raise ValueError('Invalid "RGM_classifier_layers": {hparams["RGM_classifier_layers"]}')
#
#     num_hidden_layers = hparams['RGM_classifier_layers'] - 1
#     activation = hparams['nn_act']
#     classifier_layers = [hparams['nn_end_dim'] for _ in range(num_hidden_layers)]
#     classifier_layers.append(output_layer_size)
#
#     classifier = get_sequential_NN(hparams['nn_end_dim'], classifier_layers, activation, output_layers)
#     return(classifier)

def get_validation_columns(df_data, args, domain_colname):
    """Adds validation columns to a df based on the current CV columns, so that only the train rows are split again in test and train. The old CV columns will be renamed to 'test_`CV_col`' and the new validation columns will be named from 0 to nfolds*nfolds-1.
    """
    all_CV_cols = [col for col in df_data.columns if col.startswith('CV_')]
    # Keep old CV columns around for sanity checks.
    df_data = df_data.rename(columns={cv: 'test_'+cv for cv in all_CV_cols})
    counter = 0
    for cv in all_CV_cols:
        old_cv = 'test_' + cv
        df = df_data.loc[df_data[old_cv] == 'train']
        # Use same type of CV for validation columns as for test columns.
        df = get_train_test_data(
                                        df_data=df,
                                        CV=args.CV,
                                        n_folds=args.n_folds,
                                        n_repeats=args.n_repeats,
                                        domain_colname=domain_colname,
                                        trainfrac=args.train_frac,
                                        random_n_reps=args.n_reps,
                                        group=args.CV_keep_groups
                                        )
        cv_df = df[all_CV_cols]
        
        # Name validation columns from 0 to nfold*nfold-1.
        n_cvs = len(all_CV_cols)
        rename_cols = {f'CV_{i}': f'CV_{n_cvs*counter+i}' for i in range(n_cvs)}
        cv_df = cv_df.rename(columns=rename_cols)
        counter +=1
        
        df_data = df_data.join(cv_df)
        
        # Sanity check.
        new_cv_cols = list(rename_cols.values())
        assert df_data.loc[df_data[old_cv] == 'test', new_cv_cols].isna().all().all(), 'Some of the old test columns do not have NaN columns in the validation columns!'
        
    return df_data


def get_all_models(hparams, n_features, n_targets, use_models, n_domains=1, domain_col=None, output_layers=None, outdir=None, scaler=None, args=None):
    """Definitions of models that I regularly use.
    """
    all_models = {}
    ####################    
    # 1 NEAREST NEIGHBOR
    ####################
    if '1NN' in use_models:
        Nearest_Neighbors = KNeighborsRegressor(n_neighbors=1)
        all_models['1NN'] = Nearest_Neighbors
    
    ###################
    # LINEAR REGRESSION
    ###################
    if 'LR' in use_models:
        Linear_Regression = sklearn.linear_model.LinearRegression()
        all_models['LR'] = Linear_Regression
  
    ################
    # Neural Network
    ################
    # Set some hyperparameter variables for the NN.
    net_dims = ML.net_pattern(
                                hparams['nn_layers'],
                                hparams['nn_base_dim'],
                                hparams['nn_end_dim']
                                )
    net_dims2 = ML.net_pattern(
                                hparams['RGM_classifier_layers'],
                                hparams['nn_end_dim'],
                                hparams['nn_end_dim']
                                )
    net_dims = net_dims + net_dims2
    if 'NNsk' in use_models:
        NNsk = MLPRegressor(
                            hidden_layer_sizes=net_dims,
                            activation=hparams['nn_act'],
                            solver='adam',
                            max_iter=hparams["n_epochs"],
                            early_stopping=True,
                            validation_fraction=0.2,
                            alpha=hparams["nn_l2"],
                            batch_size=hparams['nn_batch_size'],
                            learning_rate_init=hparams["learning_rate"],
                            n_iter_no_change=hparams["nn_patience"]
                            )
        all_models['NNsk'] = NNsk

        
    # ###############
    # # Lightning MLP
    # ###############
    # if 'NNL' in use_models:
    #     NNL = MLP_Lightning.MLP(
    #                         hidden_layer_sizes=net_dims,
    #                         activation=hparams['nn_act'],
    #                         solver='adam',
    #                         n_epochs=hparams["n_epochs"],
    #                         validation_fraction=0.2,
    #                         alpha=hparams["nn_l2"],
    #                         batch_size=hparams['nn_batch_size'],
    #                         learning_rate=hparams["learning_rate"],
    #                         patience=hparams["nn_patience"]
    #                         )
    #     all_models['NNL'] = NNL

    ###############
    # Random Forest
    ###############
    if 'RF' in use_models:
        n_trees = hparams["RF_n_estimators"]
        Random_Forest = RandomForestRegressor(n_estimators=n_trees)
        all_models['RF'] = Random_Forest


    ############################
    # Gradient Boosting
    ############################
    if 'GB' in use_models:
        n_trees = hparams["GB_n_estimators"]
        Gradient_Boosting =  GradientBoostingRegressor(n_estimators=n_trees)
        all_models['GB'] = Gradient_Boosting
    
    #########
    # XGBoost
    #########
    if 'XGB' in use_models:
        XGBoost = XGBRegressor()
        all_models['XGB'] = XGBoost
    
        
    ############################
    # Gaussian Process
    ############################
    batch_size = 100
    epochs = 1000
    learning_rate = 0.1
    n_inducing_points = 100
    lengthscales = np.full(n_features, hparams['GP_lengthscale'])
    noise = hparams["GP_alpha"]

    if 'GPsk' in use_models:
        kernel = ConstantKernel() * RBF(length_scale=lengthscales)
        Gaussian_Process = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True)
        all_models['GPsk'] = Gaussian_Process
    
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'GPR' in use_models:
    #     model = gpflow.models.GPR
    #     GPR = GPflow_GP(model, kernel, alpha=noise)
    #     all_models['GPR'] = GPR
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SGPR' in use_models:
    #     model = gpflow.models.SGPR
    #     SGPR = GPflow_GP(model, kernel, alpha=noise, n_inducing_points=n_inducing_points, standard_scale=True)
    #     all_models['SGPR'] = SGPR
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'VGP' in use_models:
    #     model = gpflow.models.VGP
    #     VGP = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
    #     all_models['VGP'] = VGP
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SVGP' in use_models:
    #     model = gpflow.models.SVGP
    #     SVGP = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, train_noise=True, diff_std_for_sc_and_non_sc=False, natgrad=False, train_noise_scale=False, predict_y=False)
    #     all_models['SVGP'] = SVGP
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SVGP_single' in use_models:
    #     model = gpflow.models.SVGP
    #     SVGP = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, train_noise=True, diff_std_for_sc_and_non_sc=False, natgrad=False, train_noise_scale=False, predict_y=False)
    #     all_models['SVGP_single'] = SVGP
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SVGP_sc_non-sc' in use_models:
    #     model = gpflow.models.SVGP
    #     SVGP = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, train_noise=True, diff_std_for_sc_and_non_sc=True, natgrad=False, train_noise_scale=False, predict_y=False)
    #     all_models['SVGP_sc_non-sc'] = SVGP
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SVGP_features' in use_models:
    #     model = gpflow.models.SVGP
    #     SVGP = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, train_noise=False, diff_std_for_sc_and_non_sc=False, natgrad=False, train_noise_scale=True, predict_y=True)
    #     all_models['SVGP_features'] = SVGP
    #
    #     kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales) + gpflow.kernels.WhiteKernel(variance=noise**2)
    # if 'SVGP_white' in use_models:
    #     model = gpflow.models.SVGP
    #     SVGP = GPflow_GP(model, kernel, alpha=0.0011, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, train_noise=True, diff_std_for_sc_and_non_sc=False, natgrad=False, train_noise_scale=False, predict_y=False)
    #     all_models['SVGP_white'] = SVGP
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SVGP_RGM' in use_models:
    #     model = gpflow.models.SVGP
    #     NN_path = 'RGM'
    #     SVGP_RGM = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, NN_path=NN_path)
    #     all_models['SVGP_RGM'] = SVGP_RGM
    #
    # kernel = gpflow.kernels.Constant() * gpflow.kernels.RBF(lengthscales=lengthscales)
    # if 'SVGP_NN' in use_models:
    #     model = gpflow.models.SVGP
    #     NN_path = 'NN'
    #     SVGP_NN = GPflow_GP(model, kernel, alpha=noise, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, n_inducing_points=n_inducing_points, standard_scale=True, NN_path=NN_path)
    #     all_models['SVGP_NN'] = SVGP_NN
    #
    #
    # # For all pytorch models:
    # input_layer_size = n_features
    # featurizer = get_featurizer(input_layer_size, hparams, mode='FeedForward')
    # classifier = get_classifier(output_layer_size=n_targets, hparams=hparams, output_layers=output_layers)
    #
    #
    # ############################
    # # Regret Minimization Network
    # ############################
    # if 'RGM' in use_models:
    #     RGM = RGM_sklearn(
    #                         solver=hparams['nn_solver'],
    #                         max_iter=hparams["n_epochs"],
    #                         batch_size=hparams['nn_batch_size'],
    #                         learning_rate_init=hparams["learning_rate"],
    #                         featurizer=deepcopy(featurizer),
    #                         classifier=deepcopy(classifier),
    #                         batch_mode=hparams['RGM_batch_mode'],
    #                         weight_decay=hparams["nn_l2"],
    #                         rgm_e=hparams['RGM_rgm_e'],
    #                         erm_e=hparams['RGM_erm_e'],
    #                         holdout_e=hparams['RGM_holdout_e'],
    #                         detach_classifier=hparams['RGM_detach_classifier'],
    #                         oracle=hparams['RGM_oracle'],
    #                         ensemble_pred=hparams['RGM_ensemble_pred'],
    #                         validation_fraction=0.2,
    #                         early_stopping=hparams['RGM_early_stopping'],    # 'False', 'valid', 'extrapol', ...
    #                         n_iter_no_change=hparams["nn_patience"],
    #                         clip_grad=hparams['NN_clip_grad'],
    #                         num_train_domains=hparams['RGM_num_train_domains'],
    #                         max_n_classifiers=10,
    #                         if_log_metrics=False,
    #                         coeff_lr_classifier=hparams['coeff_lr_classifier'],
    #                         reduce_lr_factor=hparams['NN_reduce_lr_factor'],
    #                         use_tensorboard=False
    #                     )
    #     RGM.domain_col = domain_col
    #     all_models['RGM'] = RGM
    #
    #
    # ############################
    # # Regret Minimization Network without domains
    # ############################
    # if 'NN' in use_models:
    #     NN = RGM_sklearn(
    #                         solver=hparams['nn_solver'],
    #                         max_iter=hparams["n_epochs"],
    #                         batch_size=hparams['nn_batch_size'],
    #                         learning_rate_init=hparams["learning_rate"],
    #                         featurizer=deepcopy(featurizer),
    #                         classifier=deepcopy(classifier),
    #                         batch_mode='Conserve_ratio',
    #                         weight_decay=hparams["nn_l2"],
    #                         rgm_e=1,
    #                         erm_e=1,
    #                         holdout_e=1,
    #                         detach_classifier=False,
    #                         oracle=False,
    #                         ensemble_pred=False,
    #                         validation_fraction=0.2,
    #                         early_stopping='valid',
    #                         n_iter_no_change=hparams["nn_patience"],
    #                         clip_grad=hparams['NN_clip_grad'],
    #                         num_train_domains=1,
    #                         if_log_metrics=False,
    #                         reduce_lr_factor=hparams['NN_reduce_lr_factor'],
    #                         use_tensorboard=False
    #                     )
    #     all_models['NN'] = NN
    #
    # #################
    # # Original MEGNet
    # #################
    # if 'MEGNet' in use_models:
    #     validation_frac = 0.2
    #     # transfer_model = None if args.add_params['prev_model'] is None else projectpath(args.add_params['prev_model'])
    #     MEGNet0 = MEGNet_tf(
    #                         use_learnt_elemental_embedding=False,
    #                         epochs=hparams['n_epochs'],
    #                         lr=args.add_params['lr'],#hparams['learning_rate'],
    #                         batch_size=args.add_params['batch_size'],#hparams['nn_batch_size'],
    #                         patience=hparams['nn_patience'],
    #                         l2_coef=args.add_params['l2'],#None,
    #                         dropout=args.add_params['dropout'],
    #                         r_cutoff=4,
    #                         early_stopping=args.add_params['early_stopping'],
    #                         validation_frac=validation_frac,
    #                         loss='mse',
    #                         domain_col=domain_col,
    #                         optimizer_kwargs={'clipnorm': args.add_params['clipnorm']},
    #                         tensorboard=False,
    #                         nblocks=args.add_params['nblocks'],
    #                         n1=args.add_params['n1'],
    #                         n2=args.add_params['n2'],
    #                         n3=args.add_params['n3'],
    #                         lr_exp_decay=args.add_params['lr_exp_decay'],#0.997,
    #                         prev_model=None,
    #                         act=args.add_params['act'],
    #                         npass=args.add_params['npass'],
    #                         n_feat_bond=args.add_params['n_feat_bond']
    #                         )
    #     all_models['MEGNet'] = MEGNet0
    
    return(all_models)





def train_with_args(args):
    os.chdir(args.calcdir)

    print("JOB STARTED")
    print_title("Experiment: {}".format(args.experiment))
    print("Current working directory: %s"%(args.calcdir))
    
    # Get hyperparameters    )
    args.hparams = get_hparams(args.hparams_file)
    
    
    args.random_seed = args.hparams['random_seed']
    if args.random_seed != None:
        print(f"RANDOM SEED FIXED!!!: {args.random_seed}")
    else:
        args.random_seed = random.randint(0, 1e3)
        
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
            
    
    domain_colname = args.domain_colname
    print(f'Domain: {domain_colname}')

    # Add CV mode to name of output directory
    outdir = make_output_directory(args.outdirname, label=args.experiment)
    print(f'Output directory: {outdir}')

    
    # TODO
    # assert not 'n_repetitions' in args.add_params
    # wished_CV_cols = []
    # # wished_CV_cols = [f'CV_{i}' for i in range(args.add_params['n_repetitions'])]
    # if args.add_params['features'] == 'graph':
    #     usecols = lambda x: (not (x.startswith('SOAP') or x.startswith('PCA') or  x.startswith('MAGPIE'))) and ((not x.startswith('CV_')) or (x in wished_CV_cols))
    # else:
    #     usecols = lambda x: (not x.startswith('CV_')) or (x in wished_CV_cols)
    
    # print(f'n_repetitions = {len(wished_CV_cols)}')


    # Get dataset.
    df_data = pd.read_csv(args.dataset, header=1)
    if args.use_data_frac != None:
        warnings.warn(f'Using only a fraction of {args.use_data_frac} of data for debugging purposes.')
        df_data = df_data.sample(frac=args.use_data_frac).reset_index()
    
    # Add absolute path to graphs.
    if args.add_params['features'] == 'graph':
        df_data['graph'] = df_data['graph'].apply(projectpath)
    
    magpie_features = [magpie for magpie in df_data.columns if magpie.startswith('MAGPIE')]
    soap_features = [soap for soap in df_data.columns if soap.startswith('SOAP')]
    pca_soap_features = [f'PCA_SOAP_{i}' for i in range(100)]
    electronic_features = ['band_gap_2', 'energy_2', 'energy_per_atom_2', 'formation_energy_per_atom_2', 'total_magnetization_2', 'num_unique_magnetic_sites_2', 'true_total_magnetization_2']
    lattice_features = ['lata_2', 'latb_2', 'latc_2']
    sym_features = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic',  'tetragonal', 'triclinic', 'trigonal', 'primitive', 'base-centered', 'body-centered', 'face-centered']
    
    features = []
    if 'MAGPIE' in args.add_params['features']:
        features += magpie_features
    if 'PCA' in args.add_params['features']:
        features += pca_soap_features
    elif 'SOAP' in args.add_params['features']:
        features += soap_features
    if 'electro' in args.add_params['features']:
        features += electronic_features    
    if not (args.add_params['features'] == 'MAGPIE' or args.add_params['features'] == 'MAGPIE (all)' or args.add_params['features'] == 'graph'):
        features += ['crystal_temp_2']
        features += sym_features + lattice_features
    if args.add_params['features'] == 'graph':
        features += ['graph']    

    
    

    
    targets = ['tc']
    # Define scaling/ transformation of features and targets.
    log_transform = FunctionTransformer(
                                        func=restricted_arcsinh,
                                        inverse_func=restricted_sinh,
                                        check_inverse=False
                                        )
    Column_Transformer = {}
    for feat in features:
        Column_Transformer[feat] = StandardScaler() if features != ['graph'] else 'passthrough'
    for target in targets:
        Column_Transformer[target] = log_transform#StandardScaler()
    assert len(Column_Transformer) == len(features + targets)
    
    output_layers = None    # None or any from get_activation_fn()




    # # Train by physical group.
    # group = args.add_params['group']
    # print(f'Training only group {group}')
    # df_data = df_data.loc[df_data['sc_class'] == group]
    
    # # TODO: remove. For CV experiment: Exclude data points with unclear physical group
    # df_data = df_data[df_data['sc_class_unique_sc']]    
    
    
    # Get indices for order in which groups shall be displayed.
    if domain_colname != None:
        groupcol = df_data[domain_colname]
        sorted_grouplabels = ML.unique_sorted(groupcol)
        index_group_map = {index: grouplabel for index, grouplabel in enumerate(sorted_grouplabels)}
        print('Group indices:')
        print(index_group_map)
    
    # Get train and test columns with specified CV algorithm.
    if args.CV != None:
        df_data = get_train_test_data(
                                        df_data=df_data,
                                        CV=args.CV,
                                        n_folds=args.n_folds,
                                        n_repeats=args.n_repeats,
                                        domain_colname=domain_colname,
                                        trainfrac=args.train_frac,
                                        random_n_reps=args.n_reps,
                                        group=args.CV_keep_groups
                                        )
    
    
    
    # # Make some tests which parts of the matching & adaptation algorithm have which effects.
    # # if args.add_params['n_exclude_if_too_many_structures']:
    # #     n_max = 1
    # #     n_structures = df_data.groupby('formula_sc')['formula_sc'].transform(len)
    # #     too_many_structures = n_structures > n_max
    # #     n_exclude_sc = len(df_data[too_many_structures].drop_duplicates("formula_sc"))
    # #     df_data = df_data[~too_many_structures]
    # #     print(f'Excluding {n_exclude_sc} superconductors because they have more than {n_max} structures.')
    # if args.add_params['drop_duplicate_superconductors']:
    #     n_before = len(df_data)
    #     n_sc_before = df_data['formula_sc'].nunique()
    #     df_data = df_data.drop_duplicates('formula_sc')
    #     assert n_sc_before == df_data['formula_sc'].nunique()
    #     print(f'Lost {n_before - len(df_data)} duplicate structures but kept all superconductors.')
    # # if args.add_params['only_totreldiff=0']:
    # #     n_sc_before = df_data['formula_sc'].nunique()
    # #     df_data = df_data[df_data['totreldiff'] == 0]
    # #     n_sc_lost = n_sc_before - df_data['formula_sc'].nunique()
    # #     print(f'Lost {n_sc_lost} superconductors by filtering by totreldiff=0.')
    # # if args.add_params['only_abs_matches']:
    # #     n_sc_before = df_data['formula_sc'].nunique()
    # #     df_data = df_data[df_data['correct_formula_frac']]
    # #     n_sc_lost = n_sc_before - df_data['formula_sc'].nunique()
    # #     print(f'Lost {n_sc_lost} superconductors by keeping only absolute matches.')
    # # if args.add_params['without_lattice_feats']:
    # #     remove_features = sym_features + lattice_features
    # #     features = [feat for feat in features if not feat in remove_features]
    # df_data['weight'] = 1 / df_data.groupby('formula_sc')['formula_sc'].transform(len)
  
        
    # Make a train data experiment.
    # if not args.add_params['train_frac'] is None:
    #     fraction = args.add_params['train_frac']
    #     CV_cols = [col for col in df_data.columns if col.startswith('CV_')]
    #     for cv in CV_cols:
    #         is_test = df_data[cv] == 'test'
    #         is_train = df_data[cv] == 'train'
    #         all_train_groups = df_data.loc[is_train, args.CV_keep_groups].unique()
    #         n_train_groups = int(len(all_train_groups) * fraction)
    #         train_groups = np.random.choice(all_train_groups, size=n_train_groups, replace=False)
    #         df_data.loc[is_train, cv] = np.where(df_data.loc[is_train, args.CV_keep_groups].isin(train_groups), 'train', np.nan)
    #         assert all(df_data[is_test] == 'test')
    #         assert all(df_data[is_train].isna() | (df_data[is_train] == 'train'))
    #     mean_n_sc = (df_data.drop_duplicates('formula_sc')[CV_cols] == 'train').sum().mean()
    #     args.add_params['mean_n_train_sc'] = float(mean_n_sc)
        
        
        

    # Rename test columns and make validation columns instead.   
    # if args.add_params['HPO'] and CV != None:
    #     df_data = get_validation_columns(df_data, args, domain_colname)
        
    # Sanity check.
    if args.CV_keep_groups == 'chemical_composition_sc':
        CV_cols = [col for col in df_data.columns if col.startswith('CV_')]
        for CV_col in CV_cols:
            is_test = df_data[CV_col] == 'test'
            is_train = df_data[CV_col] == 'train'
            test_comps = df_data.loc[is_test, 'chemical_composition_sc'].unique()
            train_comps = df_data.loc[is_train, 'chemical_composition_sc'].unique()
            assert len(np.intersect1d(test_comps, train_comps)) == 0
        
    
    # Select data by criteria.
    # if CV != None:
    #     from dataset_preparation import _6_select_best_matches_and_prepare_df
    #     df_data = _6_select_best_matches_and_prepare_df.keep_only_best_matches(df=df_data, criteria=args.add_params['criteria'], n_exclude_if_more_structures=args.add_params['n_exclude_if_more_structures'], output_graph_dir=None)
    
    args.add_params['n_data_points'] = len(df_data)
    print('Number of data points:', len(df_data))
    
    
    

    # Save values of some variables in the final output file for later convenience.
    save_value_of_variables = {key: val for key, val in vars(args).items()}
    save_value_of_variables['Domain_colname'] = domain_colname
    
    n_domains = len(df_data[domain_colname].unique()) if domain_colname != None else 1
    all_models = get_all_models(args.hparams, len(features), len(targets), args.use_models, n_domains, domain_colname, output_layers, outdir=outdir, args=args)
    use_models = {modelname: all_models[modelname] for modelname in args.use_models}
    
    ml = ML.Machine_Learning(
                            data=df_data,
                            features=features,
                            targets=targets,
                            domain=domain_colname,
                            sample_weights=args.sample_weights,
                            metrics_sample_weights=args.metrics_sample_weights,
                            Column_Transformer=Column_Transformer,
                            save_models=args.save_models,
                            save_torch_models=True,
                            is_optimizing_NN=args.is_optimizing_NN,
                            save_value_of_variables=save_value_of_variables,
                            print_features=False,
                            print_targets=True,
                            print_domain_score=True,
                            random_seed=args.random_seed,
                            save_all_values_and_predictions=True,
                            n_jobs=args.n_jobs,
                            copy_files=[args.hparams_file]
                            )        

    print('Start training.')
    ml.train(use_models, outdir)
    
    
    # # Make lots of plots.
    # plt.ioff()
    # plot_dir = os.path.join(outdir, 'plots')
    # models_without_loss_curve = ['LR', '1NN', 'RF', 'XGB', 'GPsk', 'GPR', 'SGPR']
    # run = MLRun(outdir)
    # plot_models = [modelname for modelname in args.use_models if not modelname in models_without_loss_curve]
    # run.final_plots(plot_dir, plot_models, df_data, domain_colname, features, targets, use_models, outdir)
    
    # Check if refactoring was successful and everything has stayed the same as in comparison_dir.
    # comparison_dir = '/home/timo/superconductors_3D/analysis/results/testing/results_202_'
    # refactoring = Refactoring.Refactoring(comparison_dir)
    # refactoring.check(outdir)
       
    return(ml, outdir)


def main(args_from_fn):
    
    # =============================================================================
    #                      Define options.
    # =============================================================================

    # use_models = ['1NN', 'LR', 'XGB', 'SVGP', 'NNsk', 'NN', 'RGM']
    use_models = ['XGB']
    experiment = ''
    add_params =  {
              #        'features': 'graph',
              #       'database': 'MP',
              # "act": "relu",
              # "dropout": 0.8762999083316979,
              # "lr": 0.000008567057310854599,
              # "lr_exp_decay": 0.983512961357719,
              # "n1": 46,
              # "n2": 52,
              # "n3": 73,
              # "n_feat_bond": 18,
              # "nblocks": 2,
              # "npass": 8,
              # 'batch_size': 56,
              # 'clipnorm': 1.063448501785796,
              # 'l2': 2.1555727094418956e-7,
   # 'n_exclude_if_too_many_structures': False,
    # 'drop_duplicate_superconductors': False,
   # 'only_totreldiff=0': False,
   # 'only_abs_matches': False,
   # 'same_n_sc': True,
   # 'without_lattice_feats': True,
  # 'criteria': ['no_crystal_temp_given_2']
   # 'group': 'Oxide',
  # 'train_frac': 0.3,
  # 'early_stopping': True,
   'features': 'MAGPIE+DSOAP',
   # 'CV': 'LOGO',
   'CV_keep_groups': 'chemical_composition_sc',
  # 'domain_colname': None,#'num_elements_sc',
    # 'CV_name': 'LOCO_phys',
  }
    output_note = ''
    outdirname = '/home/timo/superconductors_3D/analysis/results/testing'#'/media/timo/ext42/academic_projects/superconductors_3D/analysis/results/testing'
    calcdir = os.getcwd()
    # Cross validation
    CV = 'Random'    # `KFold`, `LOGO`, `Random` or None
    n_folds = 5     # for KFold
    n_repeats = 1   # for KFold
    CV_keep_groups = 'chemical_composition_sc'     # for KFold, Random
    n_reps = 3      # for Random
    train_frac = 0.8 # for Random
    domain_colname = None  # for LOGO
    # Weights
    sample_weights = 'weight'
    metrics_sample_weights = 'weight'
    # Dataset
    dataset = '/home/timo/superconductors_3D/data_before/final/MP/SC_MP_matches.csv'#projectpath('data', 'final', 'ICSD', 'SC_ICSD_matches.csv')#projectpath('data', 'intermediate', 'MP', '5_features_SC_MP.csv')
    # Hyperparameters
    hparams_file = 'hparams.yml'
    n_jobs = 1
    is_optimizing_NN = False # Only run NN and don't plot anything when optimizing NN
    save_models = True
    # Debugging
    use_data_frac = None    # None for using all data.

    
# =============================================================================
#                   End define options.
# =============================================================================
    print('Input:\n', sys.argv)
    args = parse_arguments(args_from_fn, use_models, experiment, output_note, outdirname, calcdir, save_models, CV, n_folds, n_repeats, CV_keep_groups, n_reps, train_frac, domain_colname, is_optimizing_NN, dataset, hparams_file, n_jobs, sample_weights, metrics_sample_weights, use_data_frac, add_params)
    print('args.add_params:\n', args.add_params)
    
    # Run ML and measure time of run.
    starttime = datetime.datetime.now()
    ml, _ = train_with_args(args)
    duration = datetime.datetime.now() - starttime
    print(f'Duration:  {duration}')
    
    return ml
    
    
    
    
if __name__ == "__main__":
    ml = main(args_from_fn={})