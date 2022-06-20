#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:02:38 2021

@author: Timo Sommer

Make parity plots of one run.
"""


run_dir = '/home/timo/Masterarbeit/Rechnungen/main_results/210915_Try_multiple_CVs_SOAP_100_and_1000/results/results_0_SOAP_100_Class1_sc'
plot_models = ['RF', 'NNsk', 'RGM']
repetitions = [0, 1, 2, 3, 4, 5, 6]
domain_colname = 'Class1_sc'
targets = ['tc']
duplicate_col = 'formula_sc'
plot_log_log = True



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
from superconductors_3D.machine_learning.own_libraries.data.All_scores import All_scores
import os
from superconductors_3D.machine_learning.own_libraries.analysis.Experiments.Run import MLRun
sns.set_theme()



if __name__ == '__main__':
    
    run = MLRun(run_dir)
    for target, model in product(targets, plot_models):
        run.parity_plot_all_together(target, model, repetitions, domain_colname, duplicate_col=duplicate_col, log_log=plot_log_log)
    for model, repetition, target in product(plot_models, repetitions, targets):
        run.parity_plot(target, model, repetition, hue=domain_colname, duplicate_col=duplicate_col, log_log=plot_log_log)
