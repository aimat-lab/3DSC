#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:02:27 2021

@author: timo
Script for plotting some nice statistical plots of the matched dataset.
"""
from superconductors_3D.utils.projectpaths import projectpath

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import numpy as np
import argparse
from sklearn.manifold import TSNE
from copy import deepcopy
from sklearn.decomposition import PCA
from scipy.stats import kde
from superconductors_3D.dataset_preparation.utils.check_dataset import get_chem_dict
from superconductors_3D.analysis.analysis_variables import get_MAGPIE_features, get_structural_features, get_PCA_structural_features
import filecmp

# Set plt theme
sns.set_theme()
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize


def parse_input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', dest='database', type=str)
    args = parser.parse_args()
    return args

def get_filename(title):
    filename = title + '.png'
    return(filename)

def count_Cuprate_layers(chem_formula):
    """Counts how many layers a cuprate has by counting the Cu atoms. One Cu atom means one layer.
    """
    chemdict = get_chem_dict(chem_formula)
    n_Cu = chemdict['Cu']
    n_layers = int(round(n_Cu))
    return(n_layers)

def get_max_count(series, **kwargs):
    """Returns the maximum value of several histograms of df (grouped by groupcol), where countcol is the column of df that we want to caluclate the maximum count for. Useful for fixing the height of several histograms.
    """
    count, _ = np.histogram(series, **kwargs)
    max_count = max(count)
    return(max_count)
    
def remove_legend_title(ax, loc='best', **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = None
    ax.legend(handles, labels, loc=loc, title=title, **kws)

def get_formula_frac(formula1, formula2):
    chemdict1 = get_chem_dict(formula1)
    chemdict2 = get_chem_dict(formula2)
    values1 = np.array(list(chemdict1.values()))
    values2 = np.array(list(chemdict2.values()))
    fracs = values1 / values2
    frac = np.mean(fracs)
    # assert np.allclose(fracs, frac, rtol=0.01), f'frac={frac}, fracs={fracs}'
    return(frac)

def move_legend_for_histplot(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
    return

def make_3DSC_statistics_plots(dataset_csv, save_plot_dir, database):
    
    groups = ['Other', 'Cuprate', 'Oxide', 'Heavy_fermion', 'Chevrel', 'Ferrite', 'Carbon']
    
    usecols = lambda x: not (x.startswith('MAGPIE') or x.startswith('PCA') or x.startswith('SOAP'))
    df = pd.read_csv(dataset_csv, header=1, usecols=usecols)   
    df['if_sc'] = df['tc'].apply(lambda tc: 'sc' if tc > 0 else 'non sc')
    df['tc > 0'] = df['tc'] > 0
    
    df_sc = df.drop_duplicates(subset='formula_sc')
    
    df_groups = df[df['sc_class'].isin(groups)]
    df_sc_groups = df_sc[df_sc['sc_class'].isin(groups)]
    
    os.makedirs(save_plot_dir, exist_ok=True)
    np.random.seed(0)
    
    # =============================================================================
    # Make histogram of how many superconductors and non sc there are per group for all crystal structures.
    # =============================================================================
    plt.figure()
    df_long = pd.melt(df_groups, id_vars=['sc_class'], value_vars=['tc > 0'], value_name='tc > 0').rename(columns={'sc_class': 'Group'})
    df_num_sc = df_long.groupby(['Group', 'tc > 0']).apply(len)
    yname = 'Number of entries'
    df_num_sc = df_num_sc.rename(yname).reset_index()
    df_num_sc['Legend:'] = df_num_sc['tc > 0'].apply(lambda x: 'sc' if x else 'non sc')
    sns.barplot(data=df_num_sc, x='Group', y='Number of entries', hue='Legend:')
    plt.title('Structures')
    plt.legend(loc='best')
    plt.xticks(rotation=10)
    plt.yscale('log')
    plt.tight_layout()
    savefile = get_filename('group_non_sc_3D_structs_hist')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    
    # =============================================================================
    # Make histogram of how many superconductors and non sc there are per group for SuperCon entries.
    # =============================================================================
    plt.figure()
    df_long = pd.melt(df_sc_groups, id_vars=['sc_class'], value_vars=['tc > 0'], value_name='tc > 0').rename(columns={'sc_class': 'Group'})
    df_num_sc = df_long.groupby(['Group', 'tc > 0']).apply(len)
    yname = 'Number of entries'
    df_num_sc = df_num_sc.rename(yname).reset_index()
    df_num_sc['Legend:'] = df_num_sc['tc > 0'].apply(lambda x: 'sc' if x else 'non sc')
    sns.barplot(data=df_num_sc, x='Group', y='Number of entries', hue='Legend:')
    plt.title('Superconductors')
    plt.legend(loc='best')
    plt.xticks(rotation=10)
    plt.yscale('log')
    plt.tight_layout()
    savefile = get_filename('group_non_sc_superconductors_hist')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    
# =============================================================================
#     # =============================================================================
#     # Embedding plot of MAGPIE vs SOAP features.
#     # =============================================================================
#     
#     embeddings = ['PCA', 'TSNE']
#     magpie_cols = get_MAGPIE_features(database)
#     struct_cols = get_PCA_structural_features(database)
#     
#     df_groups_here = df_groups[df_groups['sc_class'] != 'Other']
#     feats_dict = {'3D': df_groups_here, 'MAGPIE': df_groups_here}
#     hue_order = df_groups_here['sc_class'].value_counts().index.tolist()
#     for name, data in feats_dict.items():
#         if name == 'MAGPIE':
#             feats = data[magpie_cols].to_numpy()
#         elif name == '3D':
#             feats = data[struct_cols].to_numpy()
#         tc_log = np.arcsinh(data['tc']).to_numpy()
#         group = data['sc_class']
#         
#         for emb in embeddings:
#             if emb == 'PCA':
#                 vecs = PCA(n_components=2).fit_transform(feats)
#             elif emb == 'TSNE':
#                 early_exaggeration = 12
#                 lr = max(len(feats) / early_exaggeration / 4, 50)
#                 vecs = TSNE(
#                             n_components=2,
#                             early_exaggeration=early_exaggeration,
#                             learning_rate=lr,
#                             init='random'
#                             ).fit_transform(feats)
#                 
#             # Plot.
#             hue = group
#             plt.figure()
#             sns.scatterplot(x=vecs[:,0], y=vecs[:,1], hue=hue, hue_order=hue_order, s=25)
#             plt.xlabel(f'{emb} component 0')
#             plt.ylabel(f'{emb} component 1')
#             plt.title(name)
#             plt.legend(title='')
#             plt.tight_layout()
#             savefile = get_filename(f'{emb}_{name}_scatterplot')
#             plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
#             # plt.show()
# =============================================================================
    
    # =============================================================================
    # Histogram of how many structure data points with each totreldiff exist.
    # =============================================================================
    plt.figure()
    sns.histplot(data=df, x='totreldiff', bins=30)
    plt.xlabel('$\Delta_\mathrm{totrel}$')
    plt.ylabel('Number of crystal structures')
    plt.yscale('log')
    plt.ylim((1, 2e4))
    plt.tight_layout()
    savefile = get_filename('totreldiff_hist')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    
    # =============================================================================
    # Pie diagram of the yield of matching relative chemical formulas and synthetic doping
    # =============================================================================
    fig1, ax1 = plt.subplots()
    same_rel_formula = df['totreldiff'] == 0
    same_abs_formula = df['formula_frac'] == 1
    n_perfect = len(df[same_rel_formula &  same_abs_formula].drop_duplicates('formula_sc'))
    n_rel_perfect = len(df[same_rel_formula & (~same_abs_formula)].drop_duplicates('formula_sc'))
    n_tot = len(df[df['totreldiff'] != 0].drop_duplicates('formula_sc'))
    numbers = [n_perfect, n_rel_perfect, n_tot]
    total = sum(numbers)
    labels = ['perfect\nabs. match', 'perfect\nrel. match', 'synth.\ndoped']
    label_numbers = lambda p: f'{p*total/100:.0f}'
    ax1.pie(numbers, labels=labels, autopct=label_numbers, startangle=90)
    ax1.axis('equal')
    savefile = get_filename('pie_yield_datapoints')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    # =============================================================================
    # Pie diagram of how many data points per classes.
    # =============================================================================
    fig1, ax1 = plt.subplots()
    counts = df_sc_groups['sc_class'].value_counts().to_dict()
    numbers = list(counts.values())
    labels = list(counts.keys())
    total = sum(numbers)
    label_numbers = lambda p: f'{p * total / 100:.0f}'
    ax1.pie(numbers, labels=labels, autopct=label_numbers, startangle=90)
    ax1.axis('equal')
    savefile = get_filename('pie_sc_classes')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    
    # =============================================================================
    # Make histogram of how often chemical formulas have how many different structures.
    # =============================================================================
    plt.figure()
    exclude_sc_with_more_than_n_structures = 4
    num_structures = df.groupby('formula_sc').apply(len)
    sns.histplot(x=num_structures, discrete=True)
    plt.xlabel('Number of crystal structures per SuperCon entry')
    plt.ylabel('Number of SuperCon entries')
    plt.yscale('log')
    # plt.vlines(exclude_sc_with_more_than_n_structures + 0.5, 0, 100, colors='r', linewidth=3)
    # plt.xlim(1, 15)
    plt.tight_layout()
    savefile = get_filename('num_diff_crystal_structures')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    # =============================================================================
    # Make histogram of how many space groups there are per SuperCon entry
    # =============================================================================
    plt.figure()
    num_spgs = df.groupby(['formula_sc']).agg({"spacegroup_2": "nunique"})['spacegroup_2']
    ax = sns.histplot(x=num_spgs, discrete=True)
    # remove_legend_title(ax, loc='best')
    ax.xaxis.get_major_locator().set_params(integer=True)
    # plt.title('Number of different space groups per superconductor')
    plt.xlabel('Number of different space groups')
    plt.ylabel('Number of SuperCon entries')
    plt.yscale('log')
    savefile = get_filename('num_diff_spgs')
    plt.tight_layout()
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    # =============================================================================
    # Make histogram of how many space groups there are per SuperCon entry inclduing for room temp.
    # =============================================================================
    if database == 'ICSD':
        plt.figure()
        num_spgs = df.groupby(['formula_sc']).agg({"spacegroup_2": "nunique"}).reset_index(drop=True)
        num_spgs['crystal temperature'] = 'all temperatures'
        df_room_temp = df[df['crystal_temp_2'].between(270, 301)]
        df_room_temp = df_room_temp[df_room_temp['no_crystal_temp_given_2'] == False]
        num_spgs_room_temp = df_room_temp.groupby(['formula_sc']).agg({"spacegroup_2": "nunique"}).reset_index(drop=True)
        num_spgs_room_temp['crystal temperature'] = 'room temperature'
        num_spgs = pd.concat([num_spgs, num_spgs_room_temp])[::-1]
        rgb_colors = [[116/255, 144/255, 192/255], [224/255, 157/255, 122/255]]
        ax = sns.histplot(data=num_spgs, x='spacegroup_2', hue='crystal temperature', discrete=True, legend=True, alpha=1, palette=rgb_colors)
        remove_legend_title(ax, loc='best')
        ax.xaxis.get_major_locator().set_params(integer=True)
        # plt.title('Number of different space groups per superconductor')
        plt.xlabel('Number of different space groups')
        plt.ylabel('Number of SuperCon entries')
        plt.yscale('log')
        savefile = get_filename('num_diff_spgs_with_room_temp')
        plt.tight_layout()
        plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
        # plt.show()
    
    # =============================================================================
    # Histogramm of critical temperature for 1-,2- and 3-layer systems of Cuprates
    # =============================================================================
    dfs = {'Superconductors': df_sc_groups, 'Structures': df_groups}
    for name, df0 in dfs.items():
        df_cuprates = df0.loc[df0['sc_class'] == 'Cuprate']
        df_cuprates['n_layers'] = df_cuprates['formula_sc'].apply(count_Cuprate_layers)
        layers = [1, 2, 3]
        df_cuprates = df_cuprates.loc[df_cuprates['n_layers'].isin(layers)]
        max_tc = max(df_cuprates['tc'])
        num_bins = round(max_tc / 5)
        eps = 1e-6
        bin_width = max_tc / (num_bins - 2)
        bins = np.linspace(-bin_width, max_tc, num=num_bins) + eps     # right inclusive
        global_max_count = max(df_cuprates.groupby('n_layers')['tc'].apply(get_max_count, bins=bins))*1.05
        for layer in layers:
            plt.figure()
            df_layer = df_cuprates.loc[df_cuprates['n_layers'] == layer]
            df_layer = df_layer.sort_values('if_sc', ascending=False)
            ax = sns.histplot(data=df_layer, x='tc', bins=bins, hue='if_sc', alpha=0.7)
            plt.xlim(None, max_tc)
            plt.ylim(None, global_max_count)
            remove_legend_title(ax)
            plt.title(f'{name} ({layer} layers)')
            plt.xlabel('$T_\mathrm{c}$ (K)')
            savefile = get_filename(f'Cuprates_hist_{name}_{layer}_layer')
            plt.tight_layout()
            plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
            # plt.show()
            
    
    # =============================================================================
    # Make a plot of how often there is a relative factor between the chemical formula of SuperCon entry and crystal structure.
    # =============================================================================
    plt.figure()
    df['formula_fracs'] = df.apply(lambda row: get_formula_frac(row['formula_2'], row['formula_sc']), axis=1)
    max_bin = 11#int(max(df['formula_fracs'])) + 1
    min_bin = 11#int(1/ min(df['formula_fracs'])) + 1
    max_bins = [i + 0.5 for i in range(1, max_bin)]
    min_bins = [1/(i + 0.5) for i in range(1, min_bin)][::-1]
    bins = min_bins + max_bins
    bin_names = [fr'$\frac{{1}}{{{i:.0f}}}$' for i in range(2, min_bin)][::-1] + [f'{i}' for i in range(1, max_bin)]
    counts, _ = np.histogram(df['formula_fracs'], bins=bins)
    sns.barplot(x=bin_names, y=counts, color='b')
    plt.ylim((0, 1e5))
    plt.xlabel('normalization factor')
    plt.ylabel('Number of crystal structures')
    plt.yscale('symlog')
    plt.tight_layout()
    savefile = get_filename('chem_formulas_factors_hist')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    
    
    # =============================================================================
    # Make histogram of elements in dataset.
    # =============================================================================
    plt.figure(figsize=(15, 4.8))
    chem_sys = df_sc['chemical_composition_sc']
    all_els = chem_sys.str.split('-').to_list()
    all_els = [el for els in all_els for el in els]    # flatten
    all_els = pd.Series(data=all_els, name='Count').value_counts()
    all_els = all_els.reset_index().rename(columns={'index': 'Element'})
    ax = sns.barplot(data=all_els, x='Element', y='Count', color='b')
    plt.title('Elemental prevalence plot for the SuperCon entries')
    plt.yscale('log')
    plt.ylabel('Number of occurrences')
    plt.ylim(top=1e4)
    # Make labels on top of bars
    rects = ax.patches
    labels = all_els['Element']
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom', fontsize=10)
    plt.tick_params(
        axis='x', which='both', labelbottom=False)
    plt.tight_layout()
    savefile = get_filename('elemental_prevalence_hist')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=1000)
    # plt.show()
    
    
    # =============================================================================
    # Make histogram of distribution of cell measurement temperature.
    # =============================================================================
    if database == 'ICSD':
        plt.figure()
        temp = pd.Series(df['crystal_temp_2'], name='$T_\mathrm{cry}$ (K)')
        max_temp = 300
        # collect data points higher than 300K in last bin
        temp[temp > max_temp] = max_temp + 10
        sns.histplot(x=temp, binrange=(0, max_temp+10))
        plt.yscale('log')
        plt.ylim((1, 1e5))
        plt.title('Number of structures')
        plt.ylabel('Number of crystal structures')
        plt.tight_layout()
        savefile = get_filename('crystal_temp_hist')
        plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
        # plt.show()
    else:
        print('Dataset has no crystal temperature.')
    
    
    # =============================================================================
    # Make histogram of how many space groups there are per superconductor at room temperature
    # =============================================================================
    if database == 'ICSD':
        plt.figure()
        df_room_temp = df[df['crystal_temp_2'].between(270, 301)]
        num_spgs = df_room_temp.groupby(['formula_sc']).agg({"spacegroup_2": "nunique"})['spacegroup_2']
        ax = sns.histplot(x=num_spgs, discrete=True)
        ax.xaxis.get_major_locator().set_params(integer=True)
        plt.title('Num. of diff. space groups per superconductor at room temp.')
        plt.xlabel('# diff. space groups')
        plt.ylabel('# superconductors')
        plt.yscale('log')
        savefile = get_filename('diff_spgs_room_temp')
        plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
        # plt.show()
    
    
    # =============================================================================
    # Make histogram of how many space groups there are per superconductor at room temperature only for structures where Tcrystal was clear in the ICSD
    # =============================================================================
    if database == 'ICSD':
        plt.figure()
        df_room_temp = df[df['crystal_temp_2'].between(270, 301)]
        df_room_temp = df_room_temp[df_room_temp['no_crystal_temp_given_2'] == False]
        num_spgs = df_room_temp.groupby(['formula_sc']).agg({"spacegroup_2": "nunique"})['spacegroup_2']
        ax = sns.histplot(x=num_spgs, discrete=True)
        ax.xaxis.get_major_locator().set_params(integer=True)
        # plt.title('Num. of diff. space groups per sc certainly at room temp.')
        plt.xlabel('# diff. space groups')
        plt.ylabel('# superconductors')
        plt.yscale('log')
        savefile = get_filename('num_diff_spgs_certainly_room_temp')
        plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
        # plt.show()
    
    # =============================================================================
    # Scatterplot tc over crystal_temp
    # =============================================================================
    if database == 'ICSD':
        plt.figure()
        assert all(df['crystal_temp_2'].notna())
        sns.scatterplot(data=df, x='crystal_temp_2', y='tc', s=20)
        plt.title('Correlation of $T_\mathrm{c}$ and $T_\mathrm{cry}$')
        plt.xlabel('$T_\mathrm{cry}$ (K)')
        plt.ylabel('$T_\mathrm{c}$ (K)')
        plt.tight_layout()
        savefile = get_filename('scatter_tc_over_tcell')
        plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
        # plt.show()
    
    
    
    # =============================================================================
    # Scatterplot tc over totreldiff
    # =============================================================================
    plt.figure()
    assert all(df['totreldiff'].notna())
    sns.scatterplot(data=df, x='totreldiff', y='tc', s=20)
    # plt.title('Correlation of $T_\mathrm{c}$ and rel. chem. difference')
    plt.xlabel('$\Delta_{totrel}$')
    plt.ylabel('$T_\mathrm{c}$ (K)')
    plt.tight_layout()
    savefile = get_filename('corr_tc_and_totreldiff')
    plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
    # plt.show()
    
    # =============================================================================
    # Histogram of tc for total and each group for all 3D structures and only SuperCon entries.
    # =============================================================================
    xscale = np.arcsinh         # make tc scale logarithmic
    reverse_xscale = np.sinh
    for df0, name in zip([df_groups, df_sc_groups], ['crystals', 'SuperCon']):            
        max_tc = max(df0['tc'])
        num_bins = 30
        eps = 1e-6
        bins = reverse_xscale(np.linspace(0, xscale(max_tc), num=num_bins-1)+eps)     # right inclusive
        non_sc_bin = -bins[1]
        bins = np.insert(bins, 0, non_sc_bin)     # include bin at the left for non-sc
        assert all(df0['sc_class'].isin(groups)), 'Issue with undefined superconductor groups.'
        
        for group in groups + ['Total']:
            if group == 'Total':
                df0 = df0.sort_values(by='if_sc')[::-1]     # hue is chosen equal
                tc = df0['tc']
                hue = df0['if_sc']
            else:
                df_group = df0[df0['sc_class'] == group]
                df_group = df_group.sort_values(by='if_sc')[::-1] # hue is chosen equal
                tc = df_group['tc']
                hue = df_group['if_sc']
            hue = hue.replace({'sc': 'superconductor', 'non sc': 'non superconductor'}).tolist()
            
            plt.figure()
            ax = sns.histplot(x=tc, bins=bins, hue=hue)
            if group == 'Total':
                move_legend_for_histplot(ax, new_loc='lower right')    
            else:
                move_legend_for_histplot(ax, new_loc='upper right')
            plt.yscale('symlog')
            plt.xscale('function', functions=(xscale, reverse_xscale))
            plt.xlim(2*non_sc_bin, max_tc+30)
            ylim = (0, 1e5) if (name == 'crystals' and database == 'ICSD') else (0, 1e4)
            plt.ylim(ylim)
            xticks = [0, 1, 3, 10, 30, 100]
            xticklabels = ['0', '1', '3', '10', '30', '100']
            plt.xticks(xticks, labels=xticklabels)
            if name == 'crystals':
                plt.title(f'Crystal structures ({group})')
            elif name == 'SuperCon':
                plt.title(f'SuperCon entries ({group})')
            plt.xlabel('$T_\mathrm{c}$ (K)')
            if name == 'SuperCon':
                plt.ylabel('Number of SuperCon entries')
            else:
                plt.ylabel('Number of crystal structures')
            plt.tight_layout()
            savefile = get_filename(f'tc_hist_{name}_{group}')
            plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
            # plt.show()
            
            
        
    
# =============================================================================
#     # =============================================================================
#     # Density plot tc over form_reldiff
#     # =============================================================================
#     plt.figure()
#     x = df['totreldiff']
#     y = df['tc']
#     vmin = 1e-4
#     vmax = None
#     nbins=300
#     k = kde.gaussian_kde([x,y])
#     xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
#     zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#     zi /= max(zi)
#     plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
#     plt.title('Density of $T_\mathrm{c}$ and $\Delta_{totrel}$')
#     plt.xlabel('$\Delta_{totrel}$')
#     plt.ylabel('$T_\mathrm{c}$ (K)')
#     plt.colorbar()
#     plt.tight_layout()
#     savefile = get_filename('cor_density_tc_over_totreldiff')
#     plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
#     # plt.show()
#     
#     
#     # =============================================================================
#     # Density plot tc over crystal_temp
#     # =============================================================================
#     if database == 'ICSD':
#         plt.figure()
#         x = df['crystal_temp_2']
#         y = df['tc']
#         vmin = 1e-4
#         vmax = None
#         nbins=300
#         k = kde.gaussian_kde([x,y])
#         xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
#         zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#         zi /= max(zi)
#         plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
#         plt.title('Density of $T_\mathrm{c}$ and $T_\mathrm{cry}$')
#         plt.xlabel('$T_\mathrm{cry}$ (K)')
#         plt.ylabel('$T_\mathrm{c}$ (K)')
#         plt.colorbar()
#         plt.tight_layout()
#         savefile = get_filename('normal_density_tc_over_tcell')
#         plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
#         # plt.show()
#     
#     
#     # =============================================================================
#     # Density plot tc over crystal_temp for cold temperatures.
#     # =============================================================================
#     if database == 'ICSD':
#         plt.figure()
#         df_cold = df[df['crystal_temp_2'] <= 250]
#         x = df_cold['crystal_temp_2']
#         y = df_cold['tc']
#         nbins=300
#         k = kde.gaussian_kde([x,y])
#         xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
#         zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#         zi /= max(zi)
#         plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='viridis')
#         plt.title('Density of $T_\mathrm{c}$ and $T_\mathrm{cry}$')
#         plt.xlabel('$T_\mathrm{cry}$ (K)')
#         plt.ylabel('$T_\mathrm{c}$ (K)')
#         plt.colorbar()
#         plt.tight_layout()
#         savefile = get_filename('cold_density_tc_over_tcell')
#         plt.savefig(os.path.join(save_plot_dir, savefile), dpi=300)
#         # plt.show()
# =============================================================================

def main(database):
    
    dataset_csv = projectpath('data', 'final', database, f'3DSC_{database}.csv')
    
    save_plot_dir = projectpath('..', 'results', 'dataset_statistics', f'SC_{database}_matches')
    
    make_3DSC_statistics_plots(dataset_csv, save_plot_dir, database)   
    
    print(f'Successfully saved all statistics plots in directory {save_plot_dir}!')
    
if __name__ == '__main__':
    
    database = 'ICSD'
    
    args = parse_input_parameters()
    database = args.database if not args.database is None else database
    main(database)

