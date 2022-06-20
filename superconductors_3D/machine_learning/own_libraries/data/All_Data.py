#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:23:32 2021

@author: Timo Sommer

This script contains the class All_Data for the output df with all values and predictions.
"""
import pandas as pd
import itertools
import warnings
import json
import os
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML

def tolist(*variables):
    results = []
    for var in variables:
        if isinstance(var, str) or isinstance(var, int) or isinstance(var, float):
            var = [var]
        results.append(var)
    return(results)
    
def save_All_Data(df, outpath, targets, n_repetitions, features, domains, models, **kwargs):
    """Save a df as csv with all the important metadata saved as json comment in the first line of the file.
    """
    metadata = {
                'models': models,
                'targets': targets,
                'features': features,
                'n_repetitions': n_repetitions,
                'domains': domains
                }
    for key, item in kwargs.items():
        metadata[key] = item
        
    ML.save_df_and_metadata(df, metadata, outpath)
    print(f'Saved all values and predictions with metadata to {outpath}.')
    return()

def load_All_Data(path, targets=[], features=[], models=[], domains=None, n_repetitions=[], **kwargs):
    """Loads a df with all values and predictions as All_Data object. Tries to automatically recognize targets and models if not given.
    """
    # TODO: Adapt to df with metadata.
    df = pd.read_csv(path, **kwargs)
    n_cols = len(df.columns)
    all_pred_cols = [col for col in df.columns if col.startswith('pred')]
    
    any_uncertainty = any([pred_col.count('_') != 3 for pred_col in all_pred_cols])
    if any_uncertainty and (targets == [] or models == []):
        raise ValueError('The prediction columns in the df have underscores either in the models or the targets. This will lead to errors if you don\'t specify both `targets` and `models`.')  
    targets = All_Data.get_all_targets() if targets == [] else targets
    models = All_Data.get_all_models() if models == [] else models
    n_repetitions = n_repetitions if n_repetitions != [] else [n for n in range(n_cols) if All_Data.name_CV_col(n) in df.columns]
    
    data = All_Data(df, targets, n_repetitions, features, domains, models)
    data.all_pred_cols = all_pred_cols
    
    # Sanity tests.
    assert all([int(pred_col.split('_')[-1] in n_repetitions for pred_col in all_pred_cols)])
    result_col_exists = [All_Data.name_preds_col(model, n, target) in df.columns for model, target, n in itertools.product(models, targets, n_repetitions)]
    if not all(result_col_exists):
        warnings.warn(UserWarning, 'Not all combinations of model, target and n_repetition seem to have a result column in the df. This can be a sign that the recognized models, targets or n_repetitions are wrong.')            
    return(data)


class All_Data():
    """This class represents the input and output df of Machine_Learning() where all the input features, targets, cross validation sets and predictions are collected in one df.
    """
    # TODO:
        # Add pred_cols to metadata.
        
    def __init__(self, path):
        # Note: It is important for many algorithms that the dataframe is never sorted.
        self.All_Data_path = path
        self.df, self.metadata = ML.load_df_and_metadata(self.All_Data_path)
        self.all_columns = self.df.columns
        self.n_cols = len(self.all_columns)
        self.n_entries = len(self.df)
        
        self.targets = self.get_all_targets()
        self.features = self.get_all_features()
        self.models = self.get_all_models()        
        self.domains = self.get_domains()
        self.all_CV_cols = self.get_all_CVs()
        self.all_pred_cols = self.get_preds_cols()
        
        self.has_features = len(self.features) > 0
        self.n_repetitions = list(range(len(self.all_CV_cols)))
        
        assert all([target in self.all_columns for target in self.targets])
        assert all([CV_col in self.all_columns for CV_col in self.all_CV_cols])
        
    def name_CV_col(n):
        """Retuns the name pattern of the CV column of `data` that contains the train and test indices for the nth cross validation run.
        """
        column_name = f'CV_{n}'
        return(column_name)
    
    def name_preds_col(modelname, idx, target):
        """Returns the name of the prediction results of the column of the idxth model.
        """
        column_name = f'pred_{target}_{modelname}_{idx}'
        return(column_name)
     
    def name_unc_col(modelname, idx, target, kind):
        """Returns the name of the std column of the idxth model. `kind` can be `std_lower`, `std_upper` or `std` for the lower/upper bound of the std and for the std itself.
        """
        sigma = ML.SIGMA
        if not kind in ['lower', 'upper', 'scaled_unc']:
            raise ValueError('Value of `kind` not recognizable.')
        column_name = f'{sigma}sigma_{kind}_{target}_{modelname}_{idx}'
        return(column_name)
    
    def get_all_targets(self):
        all_targets = self.metadata['targets']    
        return all_targets

    def get_all_scorenames(self):
        all_scorenames = self.metadata['scorenames']
        return all_scorenames
    
    def get_all_CVs(self):
        all_CVs = self.metadata['CVs']
        return all_CVs
    
    def get_all_models(self):
        all_models = self.metadata['models']
        return all_models
    
    def get_domains(self):
        domains = self.metadata['domains']
        return domains
    
    def get_preds_cols(self):
        preds_cols = []
        for model, idx, target in itertools.product(self.models, self.n_repetitions, self.targets):
            preds_cols.append(All_Data.name_preds_col(modelname=model, idx=idx, target=target))
        return preds_cols
    
    
    def is_train(df, i):
        """Returns a boolean which of the rows in the df are train data points for repetition i.
        """
        return df[All_Data.name_CV_col(i)] == 'train'

    def is_test(df, i):
        """Returns a boolean which of the rows in the df are test data points for repetition i.
        """
        return df[All_Data.name_CV_col(i)] == 'test'
    
    def check_if_default(variable, default):
        """Checks if a variable is defined and otherwise returns the default.
        """
        if isinstance(variable, str) and variable == 'all':
            variable = default
        return(variable)
    
    def append_if_not_in(entry, l):
        """Append `entry` to list `l` if it is not already in this list.
        """
        if not entry in l:
            l.append(entry)
        return(l)
    
    def append_pred_col(self, pred_col, model, n_repetition, target):
        """Appends prediction columns to the data df.
        """
        name = All_Data.name_preds_col(model, n_repetition, target)
        self.df[name] = pred_col
        try:
            self.models = All_Data.append_if_not_in(model, self.models)
        except AttributeError:
            self.models = [model]
        self.n_repetitions = All_Data.append_if_not_in(n_repetition, self.n_repetitions)
        self.targets = All_Data.append_if_not_in(target, self.targets)
        return()
        
    def get_preds_cols(self, models='all', targets='all', n_repetitions='all'):
        """Returns a list with all the prediction columns of the variables that are specified.
        """
        models = All_Data.check_if_default(models, self.all_models)
        targets = All_Data.check_if_default(targets, self.all_targets)
        n_repetitions = All_Data.check_if_default(n_repetitions, self.n_repetitions)
        models, targets, n_repetitions = tolist(models, targets, n_repetitions)
                
        all_colnames = []
        for model, target, n in itertools.product(models, targets, n_repetitions):
            colname = self.name_preds_col(model, n, target)
            assert colname in self.all_pred_cols
            all_colnames.append(colname)
        return(all_colnames)
    
    def get_preds_cols(models=[], targets=[], ns=[]):
        """Returns a list with the names of the columns of the df with the given models, targets and n_repetitions.
        """       
        all_colnames = []
        for model, target, n in itertools.product(models, targets, ns):
            colname = All_Data.name_preds_col(model, n, target)
            all_colnames.append(colname)
        return(all_colnames)
    
    def is_domain(self, domains):
        """Checks if data entries are from one or more columns in `domains` and returns a boolean list that indicates where this is the case.
        """
        domains = tolist(domains)
        if not self.has_domains:
            is_domain = [True for _ in range(self.n_entries)]
        else:
            is_domain = self.domain_col.isin(domains).tolist()
        return(is_domain)
    
    def get_predictions(self, models='all', targets='all', n_repetitions='all', domains='all', return_true=False):
        """Returns all the data of the variables that are specified. The default [] means everything. Returns a df if any of the input variables is a list, otherwise returns an array. If return_true=True, also returns the ground truth values of these predictions.
        """
        models = All_Data.check_if_default(models, self.all_models)
        targets = All_Data.check_if_default(targets, self.all_targets)
        n_repetitions = All_Data.check_if_default(n_repetitions, self.n_repetitions)
        if self.has_domains:
            domains = All_Data.check_if_default(domains, self.domains)
        

        if all([isinstance(var, str) or isinstance(var, float) or isinstance(var, int) for var in [models, targets, n_repetitions, domains]]):
            # If all of the variables are exactly defined with one value return a data array.
            pred_col = self.get_preds_cols(targets, n_repetitions)
            y_pred = self.df[pred_col][self.is_domain(domains)]
            y_true = self.df[targets][self.is_domain(domains)]
            assert y_pred.shape == y_true.shape
            if return_true:
                return(y_pred, y_true)
            else:
                return(y_pred)
        else:
            # If at least one of them is a list make all a list to be consistent.
            models, targets, n_repetitions, domains = tolist(models, targets, n_repetitions, domains)
        
        # Return a dataframe with the desired data.
        all_cols = []
        for target in targets:
            if return_true:
                all_cols.append(target)
            all_pred_cols = self.get_preds_cols(models, target, n_repetitions)
            all_cols.extend(all_pred_cols)            
        df_results = self.df[all_cols]
        df_results = df_results[self.is_domain(domains)]            
        return(df_results)
    
    def get_test_data(df, target, model, repetitions, domain, other_cols=[], true_target=None, pred_target=None, pred_lower_bound=None, pred_upper_bound=None, pred_scaled_unc=None):
        """Returns df with all test data.
        """
        true_target = f'true {target}' if true_target is None else true_target
        pred_target = f'pred {target}' if pred_target is None else pred_target
        pred_lower_bound = f'${ML.SIGMA} \sigma$ lower bound' if pred_lower_bound is None else pred_lower_bound
        pred_upper_bound = f'${ML.SIGMA} \sigma$ upper bound' if pred_upper_bound is None else pred_upper_bound
        pred_scaled_unc = f'${ML.SIGMA} \sigma$ (scaled)' if pred_scaled_unc is None else pred_scaled_unc
        
        data = []
        for repetition in repetitions:
            pred_target_name = All_Data.name_preds_col(model, repetition, target)
            
            pred_lower_bound_name = All_Data.name_unc_col(model, repetition, target, 'lower')
            pred_upper_bound_name = All_Data.name_unc_col(model, repetition, target, 'upper')
            pred_scaled_unc_name = All_Data.name_unc_col(model, repetition, target, 'scaled_unc')
            has_unc = pred_lower_bound_name in df.columns
            
            CV_col_name = All_Data.name_CV_col(repetition)
            CV = df[CV_col_name]
            test_data = df.loc[CV == 'test']

            for _, row in test_data.iterrows():
                new_row = {
                            true_target: row[target],
                            pred_target: row[pred_target_name],
                            'repetition': repetition
                            }
                
                if domain in row:
                    new_row['group'] = row[domain]
                    
                # Also include uncertainty if given.
                if has_unc:
                    new_row[pred_lower_bound] = row[pred_lower_bound_name]
                    new_row[pred_upper_bound] = row[pred_upper_bound_name]
                    new_row[pred_scaled_unc] = row[pred_scaled_unc_name]
                    
                # Append other rows that shall be included.
                for other_col in other_cols:
                    new_row[other_col] = row[other_col]
                    
                data.append(new_row)
        data = pd.DataFrame(data)
        return data
