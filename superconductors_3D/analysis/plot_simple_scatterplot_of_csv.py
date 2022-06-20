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

save_file = '/home/timo/Masterarbeit/Rechnungen/Datasets/Toy_datasets/Gradient1D/Gradient1D_dataset.png'

plot_x_category = 'sc_class'
plot_y_value = 'MSLE_tc'
group = ''


only_columns = ['id', f'{plot_y_value}_MLPRGM', f'{plot_y_value}_MLPsk']

rename_columns = {
                    '_test_RGM_domains_mean': 'RGM',
                    '_test_RGM_no_domains_mean': '$\mathrm{MLP_{RGM}}$',
                    '_test_Neural_Network_mean': '$\mathrm{MLP_{sk}}$',
                    '_MLPRGM': '$\mathrm{MLP_{RGM}}$',
                    '_MLPsk': '$\mathrm{MLP_{sk}}$'
    }

def make_long_df(df_wide):
        
    df_wide['id'] = df_wide.index
    df_wide = df_wide[only_columns]

    df_long = pd.wide_to_long(df_wide, stubnames=plot_y_value, i='id', j='model', suffix='\w+').reset_index(drop=False)
    
    df_long['model'] = df_long['model'].replace(rename_columns)
    
    return df_long


if __name__ == '__main__':
    
    df_wide = pd.read_csv(input_csv, header=1)
    df_long = df_wide
    df_long = df_long[df_long['test_or_train'] == 'test']

    # df_long = make_long_df(df_wide)
    
    
    plt.figure()
    sns.relplot(data=df_long, x=plot_x_category, y=plot_y_value, kind='scatter')
    
    plt.legend(title='')
    

    
    if 'r2' in plot_y_value:
        plt.ylabel(plot_y_value.replace('r2', '$r^2$'))
    
    # plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.show()
    
    
    
