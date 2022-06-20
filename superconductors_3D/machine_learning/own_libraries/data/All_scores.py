#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:37:50 2021

@author: Timo Sommer

This script contains a class for the output file `All_scores.csv`.
"""
import pandas as pd
import numpy as np
from itertools import product
import warnings
from scipy import stats
from collections import defaultdict
import os
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML


def tolist(*variables):
    results = []
    for var in variables:
        if isinstance(var, str) or isinstance(var, int) or isinstance(var, float):
            var = [var]
        results.append(var)
    return(results)

def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def save_All_scores(df, outpath, targets, scorenames, CVs, models, **kwargs):
    """Save a df as csv with all the important metadata saved as json comment in the first line of the file.
    """
    metadata = {
                'models': models,
                'targets': targets,
                'scorenames': scorenames,
                'CVs': CVs
                }
    for key, item in kwargs.items():
        metadata[key] = item
        
    ML.save_df_and_metadata(df, metadata, outpath)
    print(f'Saved All_scores df with metadata to {outpath}.')
    return()

all_scores_filename = 'All_scores.csv'

class All_scores():
    """This is a class for the output file `All_scores.csv`.
    """
    # TODO: 
        # Add support for returning only the score of some repetitions.
    
    def __init__(self, All_scores_path, all_targets=None, all_scorenames=None, all_CVs=None, all_models=None):
        
        self.model_col = 'Model'
        self.repetitions_col = 'Repetition'
        self.All_scores_path = All_scores_path        
        self.df, self.metadata = ML.load_df_and_metadata(self.All_scores_path)
        
        # Get all as defined in the files metadata.
        self.all_targets = self.get_all_targets() if all_targets == None else all_targets
        self.all_scorenames = self.get_all_scorenames() if all_scorenames == None else all_scorenames
        self.all_CVs = self.get_all_CVs() if all_CVs == None else all_CVs
        self.all_models = self.get_all_models() if all_models == None else all_models
        self.all_repetitions = list(self.df[self.repetitions_col].unique())
    
    def get_score_columns(self):
        columns = self.df.columns.drop([self.model_col, self.repetitions_col])
        return columns
        
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
    
    def name_score_col(target, score, CV):
        
        return f'{target}_{score}_{CV}'
    
    def check_if_default(variable, default):        
        """Checks if a variable is defined and otherwise returns the default.
        """
        
        if isinstance(variable, str) and variable == 'all':
            variable = default
        return(variable)
    
    def model_df(self, df, models):
        
        df = df[df[self.model_col].isin(models)]
        return df

    def get_score_cols(targets, scores, CVs):
        """Returns a list with the names of the columns of the df with the given targets, scores and CVs.
        """
        
        all_colnames = []
        for target, score, CV in product(targets, scores, CVs):
            colname = All_scores.name_score_col(target, score, CV)
            all_colnames.append(colname)
        return(all_colnames)
    
# =============================================================================
#           NOT TESTED YET
#     def get_score(self, target, score, model, CV, repetition):
#         """Return one specified score.
#         """
#         # Reduce to correct row.
#         df = self.model_df(self.df, [model])
#         df = df[df[self.repetitions_col] == repetition]
#         assert df.shape[1] == 1
#         # Reduce to correct column.
#         score_col = All_scores.name_score_col([target], [score], [CV])
#         value = float(df.loc[score_col].squeeze())
#         return(value)
# =============================================================================
    
    def get_score_stats(self, models='all', targets='all', scores='all', CVs='all', stats={'mean': np.mean, 'sem': stats.sem}):
        """Returns average scores over all repetitions. The result is a dict of model: target: scorename: stat: CV: scorevalue.
        """
        models = self.all_models if models == 'all' else models
        targets = self.all_targets if targets == 'all' else targets
        scores = self.all_scorenames if scores == 'all' else scores
        CVs = self.all_CVs if CVs == 'all' else CVs
        stats_names = list(stats.keys())
        
        score_dict = nested_dict(n=5, type=float)
        for model, target, score, CV in product(models, targets, scores, CVs):
            all_scores = self.get_scores(
                                            targets=target,
                                            scores=score,
                                            models=model,
                                            CVs=CV
                                        )
            for stat in stats_names:
                stat_score = stats[stat](all_scores)
                score_dict[model][target][score][stat][CV] = stat_score
        
        return score_dict

    def get_scores(self, targets, scores, models, CVs):
        """Return specified scores. Either specify everything as string, then you get the exact pd.Series with this data. Or specify one or more as list, then you get all in a pd.DataFrame.
        """
        # Return all targets/scores etc if `all`.
        # targets = All_scores.check_if_default(targets, self.all_targets)
        # scores = All_scores.check_if_default(scores, self.all_scorenames)
        # models = All_scores.check_if_default(models, self.all_models)
        # CVs = All_scores.check_if_default(CVs, self.all_CVs)
        # repetitions = All_scores.check_if_default(repetitions, self.all_repetitions)
        
        if all([isinstance(var, str) or isinstance(var, float) or isinstance(var, int) for var in [targets, scores, models, CVs]]):
            # If all of the variables are exactly defined with one value return a pd.Series.
            score_col = All_scores.name_score_col(targets, scores, CVs)
            df_model = self.model_df(self.df, [models])
            scores = df_model[score_col]
            return(scores)
        else:
            # If at least one of them is a list make all a list to be consistent.
            targets, scores, models, CVs = tolist(targets, scores, models, CVs)
        
            # Return a dataframe with the desired data.
            all_score_cols = All_scores.get_score_cols(targets, scores, CVs)
            df_model = self.model_df(self.df, models)
            df_scores = df_model[all_score_cols]
            return df_scores
    


# =============================================================================
#                               TESTS
# =============================================================================

if __name__ == '__main__':    
    # Set True if you want to overwrite the test files. Set to False to compare with the previously saved files.
    overwrite_previous_tests = True
    
    # Input one df for testing purposes.
    test_dir = '../tests/All_scores'
    test_df = os.path.join(test_dir, 'All_scores.csv')
    scores = All_scores(test_df)
    df = scores.df
    # Try some things and save them for comparison later.
    score_dict = scores.get_score_stats()
    
    # Specify everything when getting scores.
    # single_scores1 = scores.get_scores(targets='tc', scores='MAE', models='NN', CVs='test').reset_index(drop=True)
    # single_scores2 = scores.get_scores(targets=['tc'], scores='MAE', models='NN', CVs='test').reset_index(drop=True)
    # # Get multiple_scores.
    # multiple_scores = scores.get_scores(targets=['tc'], models='LR').reset_index(drop=True)
    # # Get everything.
    # all_scores = scores.get_scores().reset_index(drop=True)
    
    # tested_files = {'Single_scores1': single_scores1,
    #                 'all_scores': all_scores,
    #                 'Single_scores2': single_scores2,
    #                 'Multiple_scores': multiple_scores
    #                 }
    
    # for name, values in tested_files.items():
    #     savepath = os.path.join(test_dir, name)
    #     if overwrite_previous_tests:
    #         print('Overwrite test files.')
    #         values.to_csv(savepath, index=False)
    #         print(f'{name} overwritten')
    #     else:
    #         print('Compare with previous test files.')
    #         values = pd.DataFrame(values)
    #         prev_values = pd.read_csv(savepath)
    #         pd.testing.assert_frame_equal(values, prev_values)
    #         print(f'{name} good!')
    
    
    
    
    
    