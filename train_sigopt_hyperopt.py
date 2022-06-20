#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:05:13 2021

@author: Timo Sommer

Entry point for sigopt-hyperopt.
"""
import json
from superconductors_3D.machine_learning.Apply_ML_Models_v1_3 import main
import numpy as np
from itertools import product
import os
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Run import MLRun
import datetime
import shutil
from superconductors_3D.utils.projectpaths import projectpath



def params2submit(params):
    """Takes the parameters as dictionary and returns a string that specifies the submit options on the command line.
    """
    # Convert any dictionaries to json strings.
    for arg, val in params.items():
        if isinstance(val, dict):
            val = json.dumps(val)
            params[arg] = val
        
    submit_options = ''.join([f"--{arg} '{val}' " for arg, val in params.items()])
    return submit_options

def suggestions_to_input(params):
    """Converts a dictionary of suggestions to a dictionary of input parameters for ML. Suggestions with `__` in their name are put into a jsoned dictionary as input.
    """
    inp = {}
    inp['dataset'] = params['dataset_path']
    inp['outdirname'] = params['output_path']
    
    suggestions = params['suggestion']
    for key, value in suggestions.items():
        if not '__' in key:
            inp[key] = value
        else:
            subkey, val = key.split('__')
            # Add dictionary entry and make new if it doesn't exist yet.
            try:
                inp[subkey][val] = value
            except KeyError:
                inp[subkey] = {val: value}    
    return inp

def get_scores(outdir, target, CV, scores):
    """Return scores of run.
    """
    run_dir = os.path.join(outdir, 'results_0_')
    run = MLRun(run_dir)
    own_scores = run.get_scores()
    score_stats = own_scores.get_score_stats()
    
    models = np.unique(list(score_stats.keys()))
    assert len(models) == 1, 'Several models are trained, unclear which to optimize.'
    model = models[0]
    assert all(np.array(list(score_stats[model].keys())) == target), f'The specified target {target} doesn\'t exist in the scores.'
    
    evaluations = []
    for score in scores:
        mean = score_stats[model][target][score]['mean'][CV]
        err = score_stats[model][target][score]['sem'][CV]
        mean_and_err = [
                            {
                            'name': score,
                            'value': mean,
                            },
                            {
                            'name': f'{score}_sem',
                            'value': err
                            }
                        ]
        evaluations.extend(mean_and_err)
    return evaluations

def evaluate(params, target, CV, scores):
    """Evaluate ML model and return scores.
    """
    current_dir = os.getcwd()
    print('Current dir:', current_dir)
    ml_dir = os.path.join(current_dir, 'superconductors_3D', 'machine_learning')
    os.chdir(ml_dir)
    print('ML dir:', ml_dir)
    assert os.path.exists(ml_dir), f'ML_dir {ml_dir} doesn\'t exist!'
    
    try:
        main(params)
        failed = False
    except Exception as e:
        print('###################################')
        print('Error in ML evaluation encountered:')
        print(e)
        print('###################################')
        failed = True
        metadata = {'Error': str(e)[0:450]}    # length limit of 500 chars on metadata values.
        
    if not failed:
        outdir = params['outdirname']
        evaluations = get_scores(outdir, target, CV, scores)
        metadata = None
    else:
        evaluations = None
    
    os.chdir(current_dir)
    return evaluations, metadata
    
def train(config):
    """Train a model based on the suggestion of sigopt and return the results.
    """
    print('New parameters:', config)
    target = 'tc'
    CV = 'test'
    scores = ['MSLE', 'r2']
    
    params = suggestions_to_input(config)
    print('ML parameters:', params)
    
    if 'add_params' in params:
        params['add_params'].update({
                                        'drop_duplicate_superconductors': True,
                                        'features': 'graph',
                                        'database': 'ICSD',
                                        'early_stopping': True
            })
    
    
    starttime = datetime.datetime.now()
    evaluations, metadata = evaluate(params, target, CV, scores)
    # if len(scores) == 1:
    #     evaluations = np.random.randn()
    # else:
    #     evaluations = []
    #     for score in scores:
    #         evaluations.append({'name': score, 'value': np.random.randn()})
    #         evaluations.append({'name': f'{score}_sem', 'value': np.random.randn()})
    
    # Add train_time as score.
    duration = datetime.datetime.now() - starttime
    runtime = duration.total_seconds() / 3600
    if isinstance(evaluations, list):
        evaluations.append({'name': 'train_time', 'value': runtime})
    
    print('New evaluations', evaluations)
    return evaluations, metadata


if __name__ == '__main__':
    os.chdir(projectpath())
    outpath = projectpath('analysis/results/testing/test')
    results_dir = os.path.join(outpath, 'results_0_')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    
    x = {
        'suggestion': {
              "add_params__act": "relu",
              "add_params__dropout": 0.8762999083316979,
              "add_params__lr": 0.000008567057310854599,
              "add_params__lr_exp_decay": 0.983512961357719,
              "add_params__n1": 46,
              "add_params__n2": 52,
              "add_params__n3": 73,
              "add_params__n_feat_bond": 18,
              "add_params__nblocks": 2,
              "add_params__npass": 8,
              'add_params__batch_size': 56,
              'add_params__clipnorm': 1.063448501785796,
              'l2': 2.1555727094418956e-7,
              },
        'dataset_path': projectpath('data/final/ICSD/SC_ICSD_matches.csv'),
        'output_path': outpath}

    evaluations, metadata = train(x)
    
    
    
    
    