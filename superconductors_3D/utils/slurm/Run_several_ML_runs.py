#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:24:10 2021

@author: Timo Sommer

This script runs Slurm batch jobs.
"""
from simple_slurm import Slurm
import os
import sys
from shutil import copytree
from time import sleep
import json
import numpy as np
import itertools
from sklearn.model_selection import ParameterGrid
import warnings
import argparse
from superconductors_3D.utils.projectpaths import projectpath
from superconductors_3D.dataset_preparation._6_select_best_matches_and_prepare_df import get_all_combinations_of_criteria

# TODO: Write test function so that if this script is executed on the cluster it just executes, but if is exectuded locally it executes the test function and checks for common mistakes.

def check_ML_arguments(ML_arguments):
    """Checks common mistakes in ML_arguments.
    """
    if any(['add_params' in args for args in ML_arguments]):
        warnings.warn('"add_params" was written incorrectly. This arguments needs to be called "add-params"')

# Slurm options. Please adapt to your needs.
# Please do not specify `error`, `output`, `job_name` or `array`.
slurm = Slurm(
    time='5:30:00',
    partition='single',
    ntasks=1,
    mem='94000mb',
    # gres='gpu:1',
    # cpus_per_gpu=10,
    cpus_per_task=1
)






features = ['MAGPIE', 'MAGPIE+SOAP']
groups = ['Other', 'Heavy_fermion', 'Chevrel', 'Oxide', 'Cuprate', 'Ferrite',
       'Carbon']
databases = ['MP', 'ICSD']
ML_arguments = []
for feats, group, db in itertools.product(features, groups, databases):
    dataset = projectpath('data', 'final', db, f'SC_{db}_matches.csv')
    add_params = {'drop_duplicate_superconductors': False}
    add_params['features'] = feats
    add_params['group'] = group
    add_params['database'] = db
    args = {
                'add-params': add_params,
                'dataset': dataset,
                'experiment': f'{feats}+{group}+{db}'
        }
    ML_arguments.append(args) 
#%%
# =============================================================================
#               Try best runs found by sigopt
# best_runs = [
#     {'act': 'swish', 'batch_size': 32, 'clipnorm': 6.663534310204247, 'dropout': 0.0, 'l2': 4.414789695937935e-08, 'lr': 0.0013579752314905, 'lr_exp_decay': 0.9992562465392476, 'n1': 33, 'n2': 91, 'n3': 24, 'n_feat_bond': 26, 'nblocks': 11, 'npass': 3},
#     {'act': 'relu', 'batch_size': 63, 'clipnorm': 17.83379793912129, 'dropout': 0.3165653542189414, 'l2': 3.56113484834e-05, 'lr': 0.0003667183755712, 'lr_exp_decay': 0.98, 'n1': 200, 'n2': 110, 'n3': 34, 'n_feat_bond': 89, 'nblocks': 15, 'npass': 7},
#     {'act': 'relu', 'batch_size': 54, 'clipnorm': 14.342240091186415, 'dropout': 0.2440422662735349, 'l2': 2.80470789818e-05, 'lr': 0.0011775223555504, 'lr_exp_decay': 0.9914643176200776, 'n1': 193, 'n2': 88, 'n3': 22, 'n_feat_bond': 18, 'nblocks': 19, 'npass': 6},
#     {'act': 'swish', 'batch_size': 44, 'clipnorm': 9.483271708801295, 'dropout': 0.5988337501542391, 'l2': 1e-08, 'lr': 0.0008067577635833, 'lr_exp_decay': 0.9873717914638546, 'n1': 142, 'n2': 37, 'n3': 99, 'n_feat_bond': 99, 'nblocks': 1, 'npass': 2},
#     # Same with 8x the batch size due to bigger dataset.
#     # {'act': 'swish', 'batch_size': 8*32, 'clipnorm': 6.663534310204247, 'dropout': 0.0, 'l2': 4.414789695937935e-08, 'lr': 0.0013579752314905, 'lr_exp_decay': 0.9992562465392476, 'n1': 33, 'n2': 91, 'n3': 24, 'n_feat_bond': 26, 'nblocks': 11, 'npass': 3},
#     # {'act': 'relu', 'batch_size': 8*63, 'clipnorm': 17.83379793912129, 'dropout': 0.3165653542189414, 'l2': 3.56113484834e-05, 'lr': 0.0003667183755712, 'lr_exp_decay': 0.98, 'n1': 200, 'n2': 110, 'n3': 34, 'n_feat_bond': 89, 'nblocks': 15, 'npass': 7},
#     # {'act': 'relu', 'batch_size': 8*54, 'clipnorm': 14.342240091186415, 'dropout': 0.2440422662735349, 'l2': 2.80470789818e-05, 'lr': 0.0011775223555504, 'lr_exp_decay': 0.9914643176200776, 'n1': 193, 'n2': 88, 'n3': 22, 'n_feat_bond': 18, 'nblocks': 19, 'npass': 6},
#     # {'act': 'swish', 'batch_size': 8*44, 'clipnorm': 9.483271708801295, 'dropout': 0.5988337501542391, 'l2': 1e-08, 'lr': 0.0008067577635833, 'lr_exp_decay': 0.9873717914638546, 'n1': 142, 'n2': 37, 'n3': 99, 'n_feat_bond': 99, 'nblocks': 1, 'npass': 2},
#     # Try some other good combinations
#     {'act': 'relu', 'batch_size': 64, 'clipnorm': 3, 'dropout': 0.3, 'l2': 0.00001, 'lr': 0.001, 'lr_exp_decay': 0.995, 'n1': 4*64, 'n2': 4*32, 'n3': 4*16, 'n_feat_bond': 100, 'nblocks': 12, 'npass': 3},
#     {'act': 'relu', 'batch_size': 64, 'clipnorm': 7, 'dropout': 0.25, 'l2': 1e-07, 'lr': 0.001, 'lr_exp_decay': 0.999, 'n1': 300, 'n2': 90, 'n3': 20, 'n_feat_bond': 25, 'nblocks': 10, 'npass': 3}
#             ]

# database = 'MP'
# feats = 'graph'
# dataset = projectpath('data', 'final', database, f'SC_{database}_matches.csv')

# ML_arguments = []
# for counter, best_run in enumerate(best_runs):
#     add_params = best_run
#     add_params.update({
#                         'drop_duplicate_superconductors': False,
#                         'features': 'graph',
#                         'database': database,
#                         'early_stopping': True
#                         })
#     params = {
#         'add-params': add_params,
#         'experiment': f'{database}-{counter}',
#         'dataset': dataset
#         }
#     ML_arguments.append(params)
# =============================================================================



# =============================================================================
#                   Training MEGNet
# databases = ['ICSD']
# features = ['graph']
# # models = {'graph': 'MEGNet', 'MAGPIE': 'XGB', 'MAGPIE+SOAP': 'XGB'}
# acts = ['softplus']#, 'relu', 'swish']
# npasses = [12]
# ML_arguments = []
# counter = 0
# for database, feats, act, npass in itertools.product(databases, features, acts, npasses):    
#     dataset = projectpath('data', 'final', database, f'SC_{database}_matches.csv')
#     params = {  
#                 'add-params': {
#                     'database': database,
#                     'features': feats,
#                     'drop_duplicate_superconductors': True,
#                     'early_stopping': True,
#                     'nblocks': 25,
#                     'n1': 200,#64*4,
#                     'n2': 150,#32*4,
#                     'n3': 100,#16*4,
#                     'npass': npass,
#                     'act': act,
#                     'dropout': 0.3,
#                     'lr': 0.0001,#1,
#                     'lr_exp_decay': 1,
#                     'n_feat_bond': 5,
#                     'batch_size': 64,
#                     'clipnorm': 3,
#                     'l2': 0.0001,
#                     },
#                 'experiment': f'{database}-{counter}-{npass}',
#                 'dataset': dataset,

#         }    
#     ML_arguments.append(params)
#     counter += 1

# missing_indices = [0]
# ML_arguments = [ML_arguments[i] for i in missing_indices]
# =============================================================================





# =============================================================================
#                   Training per physical group
# databases = ['ICSD', 'MP']
# features = ['MAGPIE+SOAP', 'MAGPIE']
# groups = ['Other', 'Heavy_fermion', 'Chevrel', 'Oxide', 'Cuprate', 'Ferrite', 'Carbon']
# ML_arguments = []
# counter = 0
# for database, feats in itertools.product(databases, features):    
#     dataset = projectpath('data', 'final', database, f'SC_{database}_matches.csv')
#     for group in groups:
#         params = {  
#                     'add-params': {
#                         'database': database,
#                         'features': feats,
#                         'n_repetitions': 0,
#                         'group': group,
#                         },
#                     'experiment': f'{database}-{counter}',
#                     'dataset': dataset,
    
#             }    
#         ML_arguments.append(params)
#         counter += 1

# missing_indices = [0]
# ML_arguments = [ML_arguments[i] for i in missing_indices]
# =============================================================================




# =============================================================================
#                   Training data curve
# databases = ['ICSD', 'MP']
# features = ['MAGPIE+SOAP', 'MAGPIE']
# train_fracs = np.array(range(10)) / 10
# ML_arguments = []
# counter = 0
# for database, feats in itertools.product(databases, features):    
#     dataset = projectpath('data', 'final', database, f'SC_{database}_matches.csv')
#     for frac in train_fracs:
#         params = {  
#                     'add-params': {
#                         'database': database,
#                         'features': feats,
#                         'train_frac': frac,
#                         },
#                     'experiment': f'TC_{database}-{counter}',
#                     'dataset': dataset,

#             }    
#         ML_arguments.append(params)
#         counter += 1

# =============================================================================



# =============================================================================
#                   Ablation studies of dataset hyperparameters
# databases = ['ICSD', 'MP']
# features = ['MAGPIE+SOAP', 'MAGPIE']
# ablations = ['None', 'without_lattice_feats', 'drop_duplicate_superconductors']#, 'n_exclude_if_too_many_structures', 'only_totreldiff=0', 'only_abs_matches']
# ablations_with_same_n_sc = ['None', 'drop_duplicate_superconductors', 'without_lattice_feats']
# ML_arguments = []
# counter = 0
# for database, feats in itertools.product(databases, features):    
#     dataset = projectpath('data', 'final', database, f'SC_{database}_matches.csv')
#     for abl in ablations:
#         params = {  
#                     'add-params': {
#                         'database': database,
#                         'features': feats,
#                         'None': abl == 'None',
#                         'n_exclude_if_too_many_structures': abl == 'n_exclude_if_too_many_structures',
#                         'drop_duplicate_superconductors': abl == 'drop_duplicate_superconductors',
#                         'only_totreldiff=0': abl == 'only_totreldiff=0',
#                         'only_abs_matches': abl == 'only_abs_matches',
#                         'same_n_sc': abl in ablations_with_same_n_sc,
#                         'without_lattice_feats': abl == 'without_lattice_feats',
#                         'ablation': abl,
#                         'n_repetitions': 10,
#                         },
#                     'experiment': f'{database}-{counter}',
#                     'dataset': dataset,

#             }    
#         ML_arguments.append(params)
#         counter += 1
#         assert sum([params['add-params'][a] for a in ablations]) == 1

# missing_indices = [0]
# ML_arguments = [ML_arguments[i] for i in missing_indices]
# =============================================================================

# =============================================================================
# #                   Dataset preparation hyperparameters
# n_exclude_if_more_structures = 10000
# databases = ['ICSD']#, 'ICSD']
# electro = ['SOAP+electro', 'MAGPIE+SOAP+electro']#, 'MAGPIE+PCA(SOAP)+electro',  'PCA(SOAP)+electro']
# features = ['MAGPIE', 'SOAP', 'MAGPIE+SOAP']
# ML_arguments = []
# counter = 0
# for database in databases:
#     combs = get_all_combinations_of_criteria(database=database)
#     combs = [()] + combs
#     for comb, feat in itertools.product(combs, features):
#         comb = list(comb)
#         dataset = projectpath('data', 'intermediate', database, f'5_features_SC_{database}.csv')
#         arguments = {                        
#                         'add-params': {
#                                         'features': feat,
#                                         'database': database,
#                                         'n_exclude_if_more_structures': n_exclude_if_more_structures,
#                                         'criteria': comb,
#                                         },                        
#                         'dataset': dataset,
#                         'experiment': f'{database}-{counter}'
#                       }
#         ML_arguments.append(arguments)
#         counter += 1

# # missing_indices = [1, 2]
# # ML_arguments = [ML_arguments[i] for i in missing_indices]
# =============================================================================

# ML script command line options. Please adapt to your needs.
# Must be an iterable of dictionaries that specify the command line options.
# Please do not specify --calcdir.
# ML_arguments = [
                # {'experiment': 'x4FFNN',
                #   'add-params': {'nblocks': 3, 'n1': 64*4, 'n2': 32*4, 'n3': 16*4}
                  # },             
                # {'experiment': 'formation_energy',
                #   'add-params': {'prev_model': 'data/transfer_learning/MP/transfer_models/own_runs/formation_energy/models/MEGNet_0.hdf5'}
                #   },
                # {'experiment': 'band_gap',
                #   'add-params': {'prev_model': 'data/transfer_learning/MP/transfer_models/own_runs/band_gap/models/MEGNet_0.hdf5'}
                #   },
                # {'experiment': 'none',
                #   'add-params': {'prev_model': None}
                #   },
                                #   },
                # {'experiment': 'no_early_stopping',
                #  'add-params': {'early_stopping': False},
                #  'domain-colname': 'None'
                #   },
                # {'experiment': 'ungrouped',
                #  'add-params': {'early_stopping': True},
                #  'domain-colname': 'None'
                #   },
                # {'experiment': 'grouped',
                #  'add-params': {'early_stopping': True},
                #  'domain-colname': 'chemical_composition_sc'
                #   },
# ]


# sim_array_limit = 40








scriptsdir = projectpath('superconductors_3D', 'machine_learning')
run_python_in_env = projectpath('superconductors_3D', 'utils', 'slurm', 'run_in_env.sh')
ML_file = 'Apply_ML_Models_v1_3.py'

# =============================================================================
# Script starts here.
# =============================================================================
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

def pythonlist2basharray(l):
    """Converts a python list into a bash array.
    """
    s = ''
    for el in l:
        s += f'"{el}"' + ' '
        # s += json.dumps(el) + ' '
    # s = ' '.join(l)
    s = '(' + s + ')'
    return s

def submit_batch_job(params, run_python_in_env, ML_file, resultdir, scriptsdir, logdir):
    sleep(5.3)
    try:
        job_name = params['experiment']
    except KeyError:
        job_name = 'ML_run'
    # Set name of slurm job.
    slurm.add_arguments(job_name=job_name)
    # Add output and error file with names including the job_name.
    output_file = os.path.join(logdir, f'{job_name}_{Slurm.JOB_ID}.out')
    error_file = os.path.join(logdir, f'{job_name}_{Slurm.JOB_ID}.err')
    slurm.add_arguments(output=output_file)
    slurm.add_arguments(error=error_file)
    # Set command line arguments for srun.
    submit_options = params2submit(params)
    submit_job = f"srun '{run_python_in_env}' '{ML_file}' --calcdir '{scriptsdir}' --outdir '{resultdir}' {submit_options}"
    slurm.sbatch(submit_job)
    print('\nExecuted slurm script:\n' + str(slurm) + '\n' + submit_job + '\n')
    return

# def submit_job_array(all_params, run_python_in_env, ML_file, resultdir, scriptsdir, logdir, job_name, sim_array_limit):
#     sleep(1)
#     slurm.add_arguments(job_name=job_name)
    
#     # Add output and error file with names including the job_name.
#     output_file = os.path.join(logdir, f'{job_name}_{Slurm.JOB_ID}.out')
#     error_file = os.path.join(logdir, f'{job_name}_{Slurm.JOB_ID}.err')
#     slurm.add_arguments(output=output_file)
#     slurm.add_arguments(error=error_file)
    
#     n_jobs = len(all_params)
#     slurm.add_arguments(array=f'0-{n_jobs-1}%{sim_array_limit}')
    
#     # Set command line arguments for srun.
#     all_submit_options = [params2submit(params) for params in all_params]
#     options = pythonlist2basharray(all_submit_options)
#     submit_job = f"OPTIONS={options}\nOPT=echo ${{OPTIONS[$SLURM_ARRAY_TASK_ID]}}\necho ${{OPT@Q}}\nsrun '{run_python_in_env}' '{ML_file}' --calcdir '{scriptsdir}' --outdir '{resultdir}' $OPT"
#     print('\nExecuted slurm script:\n' + str(slurm) + '\n' + submit_job + '\n')

#     slurm.sbatch(submit_job, shell='/bin/bash')

#     return

check_ML_arguments(ML_arguments)

parser = argparse.ArgumentParser(description='Run an ML script with Slurm several times with different options.')
parser.add_argument('--rundirname', '-r', type=str, help='The output directory of all runs.')
args = parser.parse_args()

# Specified run directory.
rundir = args.rundirname
# To not overwrite anything.
if os.path.exists(rundir):
    print(f'{rundir} already exists. Exiting.')
    sys.exit()
else:
    print(f'Make new directory for this run: {rundir}')
    os.mkdir(rundir)

jobname = os.path.basename(rundir)

# Copy scripts directory to directory in RUNDIR for reproducibility.
base_scripts_name = os.path.basename(scriptsdir)
new_scriptsdir = os.path.join(rundir, base_scripts_name)
copytree(scriptsdir, new_scriptsdir)
scriptsdir = new_scriptsdir
print(f'Copied {scriptsdir} to {rundir} for reproducibility.')

# Make result directory.
resultdir = os.path.join(rundir, 'results')
os.mkdir(resultdir)

# Write logs to logs directory of this run.
logdir = os.path.join(rundir, 'logs')
os.mkdir(logdir)

ML_file = os.path.join(scriptsdir, ML_file)

# Submit all jobs with all combinations of options.
if len(ML_arguments) > 0:
    # submit_job_array(ML_arguments, run_python_in_env, ML_file, resultdir, scriptsdir, logdir, jobname, sim_array_limit)
    for params in ML_arguments:
        submit_batch_job(params, run_python_in_env, ML_file, resultdir, scriptsdir, logdir)
else:
    # Submit one job with default options in script.
    args = {}
    submit_batch_job(args, run_python_in_env, ML_file, resultdir, scriptsdir, logdir)

print('Submitted all jobs.')
    



