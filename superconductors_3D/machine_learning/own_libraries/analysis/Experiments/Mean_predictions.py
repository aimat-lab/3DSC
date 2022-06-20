#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:36:18 2021

@author: timo
This script outputs the mean true and predicted Tc for every group.
"""

data_and_pred = '/home/timo/Dokumente/Masterarbeit/Rechnungen/Experimente/210418 Kfold CV vs Groupwise CV/LOGO/All_values_and_predictions.csv'

# Parameters
models = ['Neural_Network']
test_or_train = 'test'
class_col = 'Class1_sc'
target_true = 'Tc_true'
target_pred = 'Tc_pred'


import pandas as pd


df = pd.read_csv(data_and_pred)

# Get only interesting data points.
df = df[df['Model'].isin(models)]
df = df[df['test_or_train'] == test_or_train]

# Get only interesting columns.
interesting_columns = [class_col, target_true, target_pred]
df = df[interesting_columns]

mean_df = df.groupby(by=class_col).mean()
std_df = df.groupby(by=class_col).std()
counts = df[class_col].value_counts().rename('Occurrences')
num_total = len(df)
fraction = counts/num_total
fraction = fraction.rename('Fraction')
mean_df = mean_df.join(std_df, lsuffix='_mean', rsuffix='_std')
mean_df = mean_df.join(counts)
mean_df = mean_df.join(fraction)
print('The mean true and predicted target per group are:')
print(mean_df)


