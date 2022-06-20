#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:04:02 2022

@author: Timo Sommer

Plots a simple catplot of specified dictionary.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme()

# only_abs_matches True or False and no other parameter from exp_dir = projectpath('analysis', 'results', '211213_ablation_studies_25_reps_3')
input_dict = [{'method': 'chemical formulas\nmatch perfectly', '# materials': 1183}, {'method': 'match after\nnormalization', '# materials': 2725}, {'method': 'match after\nnormalization\n+ artificial doping', '# materials': 9150}]

save_file = '/home/timo/superconductors_3D/paper/images/statistics/yield_of_artificial_doping_ICSD.png'

plot_x_category = 'method'
plot_y_value = '# materials'
plot_y_error = None
plot_title = None





if __name__ == '__main__':
    
    sns.set_theme()
    
    df = pd.DataFrame.from_dict(input_dict)
    
    yerr = df[plot_y_error] if not plot_y_error is None else None
    n_categories = df[plot_x_category].nunique()
    plt.bar(x=df[plot_x_category], height=df[plot_y_value], yerr=yerr, capsize=5, color=sns.color_palette('deep', n_categories))
    # sns.catplot(data=df, x=plot_x_category, y=plot_y_value, kind='bar')
    
    if plot_y_value== 'r2':
        plt.ylabel('$r^2$')
    else:
        plt.ylabel(plot_y_value)
    # plt.xlabel(plot_x_category)
    
    # plt.axhline(y=16400, color='k', linestyle='--')
    plt.ylim((0, 10000))

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.show()
    
    
    
