#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:04:02 2022

@author: Timo Sommer

Plots a simple catplot of an existing csv.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme()


input_csv = '/home/timo/superconductors_3D/analysis/results/testing/results_378_/Domain_statistics.csv'

save_file = '/home/timo/superconductors_3D/analysis/results/testing/results_378_/Domain_statistics_logr2_test.png'

plot_x_category = 'sc_class'
plot_y_value = 'logr2_tc'


only_columns = ['id', f'{plot_y_value}_MLPRGM', f'{plot_y_value}_MLPsk']

rename_columns = {
                    '_test_RGM_domains_mean': 'RGM',
                    '_test_RGM_no_domains_mean': '$\mathrm{MLP_{RGM}}$',
                    '_test_Neural_Network_mean': '$\mathrm{MLP_{sk}}$',
                    '_MLPRGM': '$\mathrm{MLP_{RGM}}$',
                    '_MLPsk': '$\mathrm{MLP_{sk}}$',
                    'Heavy_fermion': 'Heavy F.'
    }

rename_y_value = {'MSLE_tc': 'MSLE', 'SMAPE_tc': 'SMAPE', 'MAE_tc': 'MAE', 'logr2_tc': '$r^2_\log$'}
rename_x_value = {'sc_class': 'superconductor class'}

def make_long_df(df_wide):
        
    df_wide['id'] = df_wide.index
    df_wide = df_wide[only_columns]

    df_long = pd.wide_to_long(df_wide, stubnames=plot_y_value, i='id', j='model', suffix='\w+').reset_index(drop=False)
    
    
    return df_long


if __name__ == '__main__':
    
    df_wide = pd.read_csv(input_csv, header=1)
    df_long = df_wide
    df_long = df_long[df_long['test_or_train'] == 'test']
    
    # Sort by y value
    # df_long = df_long.sort_values(by=plot_y_value)
    
    


    # df_long = make_long_df(df_wide)
    df_long[plot_x_category] = df_long[plot_x_category].replace(rename_columns)
    
    sns.catplot(data=df_long, x=plot_x_category, y=plot_y_value, kind='bar')
    
    # plt.axhline(y=0.0, color='k', linestyle='--')


    if plot_y_value in rename_y_value:
        label = rename_y_value[plot_y_value]
    else:
        label = plot_y_value
    plt.ylabel(label.replace('r2', '$r^2$'))

    if plot_x_category in rename_x_value:
        plt.xlabel(rename_x_value[plot_x_category])
    
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    
    
    
