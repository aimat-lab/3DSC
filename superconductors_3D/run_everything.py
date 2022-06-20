#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:42:38 2022

@author: Timo Sommer

Runs the whole pipeline (except downloading data): First the 3DSC is generated, then statistical plots are made and XGB models are trained on this data.
"""
import argparse
from superconductors_3D import generate_3DSC, train_ML_models, plot_dataset_statistics

def parse_input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', dest='database', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    args = parser.parse_args()
    return args

def main(database, n_cpus):
    
    generate_3DSC.main(crystal_database=database, n_cpus=n_cpus)
    
    plot_dataset_statistics.main(database=database)
    
    n_reps = 100 if database == 'MP' else 25   # TODO
    start_train_frac = 0.1
    end_train_frac = 0.8
    n_train_fracs = 10
    train_ML_models.main(args_from_fn={}, database=database, n_cpus=n_cpus, n_reps=n_reps, start_train_frac=start_train_frac, end_train_frac=end_train_frac, n_train_fracs=n_train_fracs)
    
    return


if __name__ == '__main__':
    
    database = 'MP'
    n_cpus = 1
    
    args = parse_input_parameters()
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    
    main(database=database, n_cpus=n_cpus)