#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:37:39 2021

@author: Timo Sommer

This script makes a cake diagram of how many SuperCon entries were matched with crystal structures and how many were lost in the different steps of the matching algorithm.
"""
from superconductors_3D.utils.projectpaths import projectpath

database = 'ICSD'     # change this
save = True


# All datasets in the different steps of the matching algorithm.
cleaned_2 = projectpath('data', 'source', 'SuperCon', 'cleaned', '2.0_all_data_SuperCon_cleaned.csv')
cleaned_excluded = projectpath('data', 'source', 'SuperCon', 'cleaned', 'excluded_2.0_all_data_SuperCon_cleaned.csv')
matched_3 = projectpath('data', 'intermediate', database, f'3_SC_{database}_matches.csv')
synth_doped_4 = projectpath('data', 'intermediate', database, f'4_SC_{database}_synthetically_doped.csv')
final_5 = projectpath('data', 'final', database, f'3DSC_{database}.csv')

save_plot_dir = projectpath('..', 'analysis', 'dataset_statistics', f'SC_{database}_matches')


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
sns.set_theme()


usecols = ['formula_sc']
df_cleaned_excluded = pd.read_csv(cleaned_excluded, header=1)
df_cleaned_2 = pd.read_csv(cleaned_2, header=1, usecols=usecols)
df_raw_1 = df_cleaned_2.append(df_cleaned_excluded.rename(columns={'formula': 'formula_sc'}))
df_matched_3 = pd.read_csv(matched_3, header=1, usecols=usecols)
df_synth_doped_4 = pd.read_csv(synth_doped_4, header=1, usecols=usecols)
df_final_5 = pd.read_csv(final_5, header=1, usecols=usecols)


# Get number of different superconductors in each step.
assert not any(df_cleaned_2.duplicated('formula_sc'))

n_raw_1 = sum(~df_raw_1.duplicated('formula_sc'))
n_cleaned_2 = sum(~df_cleaned_2.duplicated('formula_sc'))
n_matched_3 = sum(~df_matched_3.duplicated('formula_sc'))
n_synth_doped_4 = sum(~df_synth_doped_4.duplicated('formula_sc'))
n_final_5 = sum(~df_final_5.duplicated('formula_sc'))

technicalities = n_synth_doped_4 - n_final_5
losses = {'cleaning': n_raw_1 - n_cleaned_2 + technicalities,
          'no similar\nchemical\nformulas': n_cleaned_2 - n_matched_3,
          'no artificial\ndoping possible': n_matched_3 - n_synth_doped_4,
          'successful\nmatch': n_final_5
          }


numbers = list(losses.values())#[::-1]
labels = list(losses.keys())#[::-1]
explode = [0, 0., 0, 0.1]
colors = sns.color_palette('tab10')[0:5]
colors = [colors[3], colors[0], colors[1], colors[2]]

fig1, ax1 = plt.subplots()
total = sum(numbers)
label_numbers = lambda p: f'{p * total / 100:.0f}'
ax1.pie(numbers, explode=explode, labels=labels, autopct=label_numbers, shadow=False, startangle=90, colors=colors, textprops={'fontsize': 18})
ax1.axis('equal')
filename = os.path.join(save_plot_dir, 'pie_loss_of_SuperCon_entries.png')
if save:
    plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
















