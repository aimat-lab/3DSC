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
from sklearn.model_selection import ParameterGrid
import argparse

# Slurm options. Please adapt to your needs.
# Please do not specify `error`, `output` or `job_name`.
slurm = Slurm(
    time='48:00:00',
    partition='multiple',
    # partition='dev_single',
    ntasks=1,
    mem='16G',
    # gres='gpu:1',
    # cpus_per_gpu=2
    cpus_per_task=2
)

# ML script command line options. Please adapt to your needs.
# Must be an iterable of dictionaries that specify the command line options.
# Please do not specify --calcdir.
ML_arguments = [
                {'experiment': 'Standard',
                 'KFold-group': 'None'
                 },
                {'experiment': 'formula_sc',
                 'KFold-group': 'formula_sc'
                 },
                {'experiment': 'chemical_composition',
                 'KFold-group': 'chemical_composition'
                 }
]
# ML_arguments = []





# =============================================================================
# Script starts here.
# =============================================================================

def submit_batch_job(params, run_python_in_env, ML_file, resultdir, scriptsdir, logdir):
    sleep(1)
    # Set name of slurm job.
    try:
        job_name = params['experiment']
    except KeyError:
        job_name = 'ML_run'
    slurm.add_arguments(job_name=job_name)
    # Add output and error file with names including the job_name.
    output_file = os.path.join(logdir, f'{job_name}_{Slurm.JOB_ID}.out')
    error_file = os.path.join(logdir, f'{job_name}_{Slurm.JOB_ID}.err')
    slurm.add_arguments(output=output_file)
    slurm.add_arguments(error=error_file)
    # Set command line arguments for srun.
    submit_options = ''.join([f"--{arg} '{val}' " for arg, val in params.items()])
    submit_job = f"srun '{run_python_in_env}' '{ML_file}' --calcdir '{scriptsdir}' {submit_options}"
    print('\nExecuted slurm script:\n' + str(slurm) + '\n' + submit_job + '\n')
    slurm.sbatch(submit_job)
    return
        
parser = argparse.ArgumentParser(description='Run an ML script with Slurm several times with different options.')
parser.add_argument('--rundirname', '-r', type=str, help='The output directory of all runs.')
args = parser.parse_args()

# Specified run directory.
parent_rundir = os.path.join(os.path.expanduser('~'), 'Masterarbeit/Rechnungen/main_results')
rundir = os.path.join(parent_rundir, args.rundirname)

# To not overwrite anything.
if os.path.exists(rundir):
    print(f'{rundir} already exists. Exiting.')
    sys.exit()
else:
    print(f'Make new directory for this run: {rundir}')
    os.mkdir(rundir)

# Copy scripts directory to directory in RUNDIR for reproducibility.
scriptsdir = os.path.join(os.path.expanduser('~'), 'Masterarbeit/Rechnungen/Skripte')
new_scriptsdir = os.path.join(rundir, 'Skripte')
copytree(scriptsdir, new_scriptsdir)
scriptsdir = os.path.join(rundir, 'Skripte')
print(f'Copied {scriptsdir} to {rundir} for reproducibility.')

# Make result directory.
resultdir = os.path.join(rundir, 'results')
os.mkdir(resultdir)

# Write logs to logs directory of this run.
logdir = os.path.join(rundir, 'logs')
os.mkdir(logdir)

ML_file = os.path.join(scriptsdir, 'Apply_ML_Models_v1_3.py')
# Run python in environment.
run_python_in_env = os.path.join(scriptsdir, 'Own_libraries/Slurm/run_in_env.sh')

# Submit all jobs with all combinations of options.
if len(ML_arguments) > 0:
    for args in ML_arguments:
        submit_batch_job(args, run_python_in_env, ML_file, resultdir, scriptsdir, logdir)
else:
    # Submit one job with default options in script.
    args = {}
    submit_batch_job(args, run_python_in_env, ML_file, resultdir, scriptsdir, logdir)

print('Submitted all jobs.')
    



