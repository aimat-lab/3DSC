#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:00:35 2021

@author: Timo Sommer


"""
from superconductors_3D.utils.projectpaths import projectpath
import os
import pandas as pd
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, movecol
from superconductors_3D.dataset_preparation.utils.check_dataset import get_chem_dict
from copy import deepcopy
from sklearn.model_selection import GroupKFold
import numpy as np
from superconductors_3D.machine_learning.own_libraries.models.GNN.GNN_utils import graph_features
import os 
import shutil
import itertools
from superconductors_3D.dataset_preparation.utils.crystal_utils import all_crys_sys, One_Hot_crystal_system_and_point_group, point_group_from_space_group, all_bravais_centrings, One_Hot_bravais_centring, state_features





def get_best_matches(subdf, criteria):
    subdf = subdf[criteria]
    assert subdf.notna().all().all()
    first_row = subdf.iloc[0]
    best = (subdf == first_row).all(axis=1)
    assert sum(best) >= 1
    return best

def chem_formula_frac(formula_sc, formula_2):
    """This function calculates the fraction of two chemical formulas with same relative elemental quantities.
    """
    chemdict_sc = get_chem_dict(formula_sc)
    chemdict_2 = get_chem_dict(formula_2)
    
    quantities_sc = np.array(list(chemdict_sc.values()))
    quantities_2 = np.array(list(chemdict_2.values()))
    
    fracs = quantities_2 / quantities_sc
    frac = np.mean(fracs)
    assert np.allclose(fracs, frac, rtol=1e-1)
    
    return frac

def select_best_matches(df, criteria, lower_is_better_dict):
    """If there are several 3D crystal structures that match the same SuperCon entry this function selects the best ones depending on some criteria. In this process the dataframe gets sorted.
    """
    n_superconductors = df['formula_sc'].nunique()
    df = deepcopy(df)
    
    lower_is_better = [lower_is_better_dict[crit] for crit in criteria]
    
    # Calculate the fraction of the elemental quantities of the chemical formulas of SuperCon and crystal database entry.
    df['formula_frac'] = df.apply(lambda row: chem_formula_frac(row['formula_sc'], row['formula']), axis=1)
    df['correct_formula_frac'] = df['formula_frac'].between(0.8, 1.2)
    df = movecol(df, cols=['formula_similarity', 'totreldiff', 'formula_frac', 'correct_formula_frac'], to='formula_sc')
    
    sortby = ['formula_sc'] + criteria
    ascending = [True] + lower_is_better
    
    df = df.sort_values(by=sortby, ascending=ascending).reset_index(drop=True)
    
    best_matches = df.groupby('formula_sc').apply(lambda subdf: get_best_matches(subdf, criteria)).tolist()
    # Debugging
    # df['best_matches'] = best_matches
    # df = movecol(df, ['best_matches'] + criteria, 'formula_sc')
    
    df = df[best_matches].reset_index(drop=True)
    assert all(df.duplicated(['formula_sc']) == df.duplicated(sortby))
    
    assert n_superconductors == df['formula_sc'].nunique()
        
    return df



def keep_only_best_matches(df, criteria, n_exclude_if_more_structures, output_graph_dir):

    # Add pseudo crystal temperature to Materials Project so that all graphs have the same features to enable transfer learning.
    if not 'crystal_temp_2' in df.columns:
        assert not 'no_crystal_temp_given_2' in df.columns
        df['crystal_temp_2'] = 0        
        df['no_crystal_temp_given_2'] = True
    
    # df = df[df['totreldiff'] == 0]  # TODO   
    
    # In case we want to sort by e_above_hull.
    if 'e_above_hull_2' in df:
        is_nan = df['e_above_hull_2'].isna()
        df = df[~is_nan]
        print(f'Excluded {sum(is_nan)} data points because `e_above_hull_2` had NaN values.')
        
    # Remove a few entries without crystal system.
    no_crystal_system = df['crystal_system_2'].isna() | df['spacegroup_2'].isna()
    df = df[~ no_crystal_system]
    print(f'Excluded {sum(no_crystal_system)} entries because no crystal system or space group is given.')
    print(f'After filtering crystal systems: {df["formula_sc"].nunique()} superconductors.')
    
    # Feature processing
        
    # Crystal system One-Hot encoded with corresponding point groups as integers.    
    df['point_group_2'] = df['spacegroup_2'].apply(point_group_from_space_group)
    df[all_crys_sys] = df.apply(lambda row: One_Hot_crystal_system_and_point_group(row['crystal_system_2'], row['point_group_2']), axis=1)
    # One Hot encode the centering of the bravais lattice.
    df[all_bravais_centrings] = df.apply(lambda row: One_Hot_bravais_centring(row['spacegroup_2']), axis=1)
    
    # Determine features of atoms and state for the graphs.
    print('Determine graph features.')
    if not output_graph_dir is None:      
        if os.path.exists(projectpath(output_graph_dir)):
            shutil.rmtree(projectpath(output_graph_dir))
        os.mkdir(projectpath(output_graph_dir))
    df['graph'] = df.apply(lambda row: \
                    graph_features(row=row,
                                   state_features=state_features,
                                   graph_dir=output_graph_dir
                                   ),
                            axis=1)
    
    # Exclude bad graphs
    good_graphs = df['graph'].notna()
    df = df[good_graphs]
    print(f'Excluded {sum(~good_graphs)} graphs.')
    print(f'After filtering graphs: {df["formula_sc"].nunique()} superconductors.')

    lower_is_better_dict = {'no_crystal_temp_given_2': True, 'totreldiff': True, 'correct_formula_frac': False, 'e_above_hull_2': True}
    # Selects the best matches according to some criteria.
    n_formulas = df['formula_sc'].nunique()
    df = select_best_matches(df, criteria, lower_is_better_dict)
    
    print(f'After filtering criteria: {df["formula_sc"].nunique()} superconductors.')
    assert df['formula_sc'].nunique() == n_formulas
    
    # Get weights based on how often each SuperCon entry is in the dataset.
    df['weight'] = 1 / df.groupby('formula_sc')['formula_sc'].transform(len)
    
    # Exclude SuperCon entries with too many matches.
    exclude_weights = 1 / n_exclude_if_more_structures
    not_too_many_structures = df['weight'] >= exclude_weights
    print(f'Excluded {sum(~not_too_many_structures)} crystals and {len(df[~not_too_many_structures].drop_duplicates("formula_sc"))} SuperCon entries because these SuperCon entries have more than {n_exclude_if_more_structures} crystal structures.')
    df = df[not_too_many_structures]
    print(f'After filtering weights: {df["formula_sc"].nunique()} superconductors.')
        
    rename_columns = {
                'formula': 'formula_2',
                'tc_sc': 'tc',
                'cif_2': 'cif',
                'sc_class_sc': 'sc_class'
        }
    df = df.rename(columns=rename_columns)
    
    print(f'Returning filtered dataframe with {df["formula_sc"].nunique()} superconductors and {len(df)} structures.')
    
    return df

def get_all_combinations_of_criteria(database):
    """Returns a list with all possibilities for the criteria in all different orders.
    """
    if database == 'MP':
        all_criteria = ['totreldiff', 'correct_formula_frac', 'e_above_hull_2']
    elif database == 'ICSD':
        all_criteria = ['no_crystal_temp_given_2', 'totreldiff', 'correct_formula_frac']
    
    all_combs = []
    for i in range(len(all_criteria)):  
        all_combs += list(itertools.permutations(all_criteria, i+1))
    
    return all_combs

def select_best_matches_and_prepare_df(input_csv_data, output_graph_dir, output_csv_data, criteria, n_exclude_if_more_structures):
    
    # usecols = lambda colname: not ('SOAP' in colname or 'MAGPIE' in colname)  # debugging
    df = pd.read_csv(input_csv_data, header=1)
    
    df_selected = keep_only_best_matches(df, criteria, n_exclude_if_more_structures, output_graph_dir)
    
    comment = 'The final 3DSC dataframe of matched SuperCon and the crytal structure database entries. It includes SOAP features, MAGPIE features and the original columns of SuperCon and the crystal structure database.'
    write_to_csv(df_selected, output_csv_data, comment)
    
    
    
if __name__ == '__main__':
    
    
    
    database = 'ICSD'     # change this

    # The best matches will be chosen based on these columns and in this order.
    # Criteria were selected by hyperparameter optimization.
    if database == 'MP':
        criteria = ['e_above_hull_2', 'totreldiff']
    elif database == 'ICSD':
        criteria = ['no_crystal_temp_given_2']
    
    # Exclude all Supercon entries that have more than this structures.
    n_exclude_if_more_structures = 10000
    
    input_csv_data = projectpath('data', 'intermediate', database, f'5_features_SC_{database}.csv')
    output_graph_dir = os.path.join('data', 'final', database, 'graphs')
    output_csv_data = projectpath('data', 'final', database, f'SC_{database}_matches.csv')
    
    select_best_matches_and_prepare_df(input_csv_data, output_graph_dir, output_csv_data, criteria, n_exclude_if_more_structures)
    

    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        