#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:32:51 2021

@author: Timo Sommer

This script includes a class for the file Domain_statistics.csv.
"""

import pandas as pd
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML

def get_score_col(score, target):
    return f'{score}_{target}'

def save_Domain_statistics(df, outpath, domains, targets, scorenames, models, **kwargs):
    """Save a df as csv with all the important metadata saved as json comment in the first line of the file.
    """
    metadata = {
                'models': models,
                'targets': targets,
                'scorenames': scorenames,
                'domains': domains
                }
    for key, item in kwargs.items():
        metadata[key] = item
        
    ML.save_df_and_metadata(df, metadata, outpath)
    print(f'Saved Domain_statistics with metadata.')
    return()

class Domain_statistics():
    
    def __init__(self, path, domain_col):
        self.df = ML.load_df_and_metadata(path)
        
        self.model_col = 'Model'
        self.repetition_col = 'rand_instance'
        self.CV_col = 'test_or_train'
        self.domain_col = domain_col
        self.all_domains = sorted(self.df[self.domain_col].unique())
    
    def get_score(self, model, domain, score, target, CV):
        """Returns the score of the corresponding model, domain, score, target and CV.
        """
        df = self.df
        # Reduce to correct row.
        models = df[self.model_col]
        domains = df[self.domain_col]
        CVs = df[self.CV_col]
        df = df.loc[(models == model) & (domains == domain) & (CVs == CV)]
        # Reduce to correct column.
        col = get_score_col(score, target)
        value = df[col].mean()
        return value


if __name__ == '__main__':
    csv = '/home/timo/Masterarbeit/Rechnungen/main_results/210915_Try_multiple_CVs_SOAP_100_and_1000/results/results_0_SOAP_100_Class1_sc/Domain_statistics.csv'
    domain_stats = Domain_statistics(csv, 'Class1_sc')
    
    value = domain_stats.get_score(model='XGB', domain='Oxide', score='MAE', target='tc', CV='train')
        
        
        