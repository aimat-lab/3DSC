#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:01:53 2021

@author: Timo Sommer

This script checks whether all train and test splits are equal in all run directories of a given experiment directory.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Experiment import Experiment

exp_dir = projectpath('analysis', 'results', '211208_dataset_hyperparameter_optimization_ICSD_25_reps_correct_comps')


exp = Experiment(exp_dir, check_same_train_test_splits='formula_sc')



