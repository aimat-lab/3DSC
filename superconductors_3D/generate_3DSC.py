#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:53:39 2022

@author: Timo Sommer

Generate the 3DSC by running the whole matching and artificial doping pipeline, from cleaning the SuperCon and the crystal structure database until outputting the final database.
"""
from superconductors_3D.utils.projectpaths import projectpath, Data_Paths
import pandas as pd
import argparse
import os
import warnings
from tqdm import tqdm
import numpy as np
import random
import datetime
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
from superconductors_3D.dataset_preparation.dataset_cleaning._0_extract_cifs import extract_cifs
from superconductors_3D.dataset_preparation.dataset_cleaning._1_clean_cifs import clean_cifs
from superconductors_3D.dataset_preparation.dataset_cleaning._2_0_clean_SuperCon import clean_SuperCon
from superconductors_3D.dataset_preparation.dataset_cleaning._2_1_clean_MP import clean_MP
from superconductors_3D.dataset_preparation.dataset_cleaning._2_2_clean_ICSD import clean_ICSD
from superconductors_3D.dataset_preparation._3_match_SC_and_3D_crystals import match_SC_and_3D_crystals
from superconductors_3D.dataset_preparation._4_synthetic_doping import perform_synthetic_doping
from superconductors_3D.dataset_preparation._5_generate_features import generate_features
from superconductors_3D.dataset_preparation._6_select_best_matches_and_prepare_df import select_best_matches_and_prepare_df




def standardize_df(df):
    """Standardizes unimportant properties of a df for comparing with assert_frame_equal, which otherwise throw errors.
    """
    return df.sort_values(by=list(df.columns), axis='index', ignore_index=True)

def assert_same_df_or_same_subset(df1, df2, id_cols):
    """Asserts that either the two df are the same or the bigger df simply has some additional rows compared to the smaller df. ´id_cols´ is a list of the columns which are used to identify two rows as the same.
    """
    df1 = standardize_df(df1)
    df2 = standardize_df(df2)
    if len(df1) == len(df2):
        pd.testing.assert_frame_equal(df1, df2, check_like=True, check_dtype=False)
        print('Refactoring of this step successful! New and old dataframes are equivalent!')
    else:
        df_larger = df1 if len(df1) > len(df2) else df2
        df_smaller = df2 if len(df1) > len(df2) else df1
        overlap = pd.Series([True]*len(df_larger))
        for col in id_cols:
            overlap = overlap & df_larger[col].isin(df_smaller[col])
        df_overlap = standardize_df(df_larger[overlap])
        try:
            pd.testing.assert_frame_equal(df_overlap, df_smaller, check_like=True, check_dtype=False)
            print(f'Refactoring mostly successful. df1 has {len(df1)} rows and df2 has {len(df2)} rows but at least the smaller of them is a subset of the bigger.')
        except AssertionError as e:
            print(f'Exception raised: {e}')
            assert sum(overlap) == len(df_smaller), f'Refactoring unsucessful. sum(overlap)={sum(overlap)} and len(df_smaller)={len(df_smaller)}'
            print('Refactoring partly sucessful. All matches in the new algorithm were also found with the old algorithm. But, the rows were not exactly the same.')
            
    return

def check_if_same_as_old(new_csv_path, compare_overlap=True):
    """Only for refactoring. Checks if the dataframes of the old and the new algorithm are the same at this step.
    """
    old_csv_path = new_csv_path.replace('/data/', '/../data_before/')   # path to old data
    if os.path.exists(old_csv_path):
        print('Check refactoring of this step by comparing old and new df...')
        df_old = pd.read_csv(old_csv_path, header=1)
        df = pd.read_csv(new_csv_path, header=1)
        
        # Remove unneccessary columns from old df.
        unneccessary_cols_startwith = ['CV_', 'PCA_SOAP_', 'best_matches']
        unneccessary_cols = [col for col in df_old.columns if any([col.startswith(name) for name in unneccessary_cols_startwith])]
        good_cols = [col for col in df_old.columns if not any([col.startswith(name) for name in unneccessary_cols_startwith])]
        df_old = df_old[good_cols]
        # Adjust historical but correct deviations in old df.
        if 'original_cif_2' in df_old:
            df_old['original_cif_2'] = df_old['original_cif_2'].str.replace('/cif/', '/cifs/')
        
        try:
            if 'formula_sc' in df.columns and 'database_id_2' in df_old:
                id_cols = ['formula_sc', 'database_id_2']        
            elif 'formula_sc' in df.columns:
                id_cols = ['formula_sc']
            elif 'database_id_2' in df.columns:
                id_cols = ['database_id_2']
            else:
                raise UserWarning('Neither "formula_sc" nor "database_id_2" exists in the columns!')
                
            print(f'Identification columns: {list(id_cols)}')
            assert_same_df_or_same_subset(df, df_old, id_cols)
        
        except Exception as e:
            print('EXCEPTION RAISED when checking refactoring:')
            print(e)
    # else:
    #     print('Check of refactoring not possible because of missing old data.')
        
    return
      
        
def print_title(txt):
    print()
    print('=================================================')
    print(txt)
    return

def set_reproducible_random_seed(seed=None):
    """Sets and prints a random seed.
    """
    if seed is None:
        seed = np.random.randint(0, 99)
    print(f'Set random seed: {seed}')
    np.random.seed(seed)
    random.seed(seed)
    return

def parse_input_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', dest='database', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    args = parser.parse_args()
    return args

def doublecheck_dataset_shape(data):
    print('3DSC dataset generation finished. Double checking shape of df with the expected shape from the paper...')
    paper_dataset_shape = {'MP': (5773, 8952)}[data.database]
    df = pd.read_csv(data.final_3DSC, header=1)
    if df.shape == paper_dataset_shape:
        print(f'Success! The final 3DSC has the shape {paper_dataset_shape} which is exactly as expected.')
    else:
        warnings.warn(f'Problem: The expected shape for the df is {paper_dataset_shape}, but the actual shape of the df is {df.shape}.')
        
def main(crystal_database, n_cpus):
        
    random_seed = 58
    
    
    # Step 1: Clean cifs.
    timeout_min = 2     # Timeout in min for very big crystal structures

    
    # Step 3: Hyperparameters for matching SuperCon entries with 3D crystal structures:        
    # Maximally allowed number of elements that are in SuperCon entry but not in crystal entry (except for pure elements where this number is set to 0).
    n_max_doping_elements = 1        
    # For similarity == 2
    lower_max_relcutoff = 0.10001
    lower_total_relcutoff = 0.05001
    lower_min_abscutoff = 0.15001        
    # For similarity == 3
    higher_max_relcutoff = 0.20001
    higher_total_relcutoff = 0.15001
    higher_min_abscutoff = 0.3001 
    
    # Step 4: Artificial doping
    # Tolerance parameter for symmetry recognition. Use 0.1 for computational databases and 0.01 for experimental databases.
    symprec = 0.1 if crystal_database == 'MP' else 0.01     # for symmetry recognition

    
    # Step 5: Generate features
# =============================================================================
#     generate_graphs = False     # If generating graphs for GNNs is wished. Not included in paper.
# =============================================================================
    # Reduce high-dimensional SOAP features using PCA and add them as additional features, next to the original SOAP features.
    n_pca_components = 0    # no PCA at all
    # Hyperparameters for Disordered SOAP (here the same as from 2021 Fung: Benchmarking graph neural networks for materials chemistry) --> Around 8000-dimensional SOAP features
    rcut = 4.270
    nmax = 6
    lmax = 4
    sigma = 0.336
    crossover = False
    
    # Step 6: Filter best matches and prepare df.
    # Criteria were selected by hyperparameter optimization. The best matches will be chosen based on these columns and in this order.
    if crystal_database == 'MP':
        criteria = ['e_above_hull_2', 'totreldiff']
    elif crystal_database == 'ICSD':
        criteria = ['no_crystal_temp_given_2'] 
    else:
        raise NotImplementedError(f'Match selection criteria for database {crystal_database} not implemented.')
    # Exclude all Supercon entries that have matched more than this number of structures.
    n_exclude_if_more_structures = 10000
    
    # For debugging: use only a fraction of each database
    sc_frac = 1
    crystal_db_frac = 1     # TODO
    

    
    starttime = datetime.datetime.now()
    print(f'Start generating the 3DSC database using the {crystal_database}.')
    data = Data_Paths(crystal_db_name=crystal_database)
    print(f'The SuperCon database is read in from {data.supercon_csv}.')
    print(f'The {crystal_database} database is read in from {data.crystal_db_csv}.')
    data.create_directories()
    set_reproducible_random_seed(random_seed)
    
    
    print_title('Step 1: Clean cifs.')
    clean_cifs(
                input_raw_csv_data=data.crystal_db_csv, 
                output_csv_cleaned_with_pymatgen=data.cifs_normalized,
                output_excluded=data.cifs_normalized_excluded,
                output_dir_cleaned_cifs=data.output_dir_cleaned_cifs, 
                comment=data.cifs_normalized_comment, 
                comment_excluded=data.cifs_normalized_comment_excluded, 
                database=data.database,
                n_cpus=n_cpus, 
                timeout_min=timeout_min,
                crystal_db_frac=crystal_db_frac,
                verbose=False
                )

    print_title('Step 2.0: Clean SuperCon.')       
    clean_SuperCon(
                    in_filename=data.supercon_csv,
                    out_filename=data.supercon_cleaned,
                    out_excluded_filename=data.supercon_cleaned_excluded,
                    comment=data.supercon_cleaned_comment,
                    comment_excluded=data.supercon_cleaned_comment_excluded,
                    sc_frac=sc_frac
                    )
    check_if_same_as_old(new_csv_path=data.supercon_cleaned)
    
    print_title(f'Step 2.1: Clean crystal database {crystal_database}.')
    if crystal_database == 'MP':
        clean_MP(
                    in_filename=data.cifs_normalized,
                    out_filename=data.crystal_db_cleaned,
                    out_excluded_filename=data.crystal_db_cleaned_excluded,
                    comment=data.crystal_db_cleaned_comment,
                    comment_excluded=data.crystal_db_cleaned_comment_excluded
                    )
    elif crystal_database == 'ICSD':
        clean_ICSD(
                    in_filename=data.cifs_normalized,
                    in_type_filename=data.icsd_type_filename,
                    out_filename=data.crystal_db_cleaned,
                    out_excluded_filename=data.crystal_db_cleaned_excluded,
                    comment=data.crystal_db_cleaned_comment,
                    comment_excluded=data.crystal_db_cleaned_comment_excluded
                    )
    else:
        raise NotImplementedError(f'For the crystal database {crystal_database} no database cleaning script was implemented or found.')
    check_if_same_as_old(new_csv_path=data.crystal_db_cleaned)
    
    print_title('Step 3: Match SuperCon and 3D crystal structure entries by chemical formula.')
    match_SC_and_3D_crystals(
                                input_sc=data.supercon_cleaned,
                                input_2=data.crystal_db_cleaned,
                                out_merged_sc_cif_file=data.merged_sc_crystal_db,
                                comment=data.merged_sc_crystal_db_comment,
                                n_cpus=n_cpus,
                                n_max_doping_elements=n_max_doping_elements,
                                lower_max_relcutoff=lower_max_relcutoff,
                                lower_total_relcutoff=lower_total_relcutoff,
                                lower_min_abscutoff=lower_min_abscutoff,
                                higher_max_relcutoff=higher_max_relcutoff,
                                higher_total_relcutoff=higher_total_relcutoff,
                                higher_min_abscutoff=higher_min_abscutoff
                                )
    check_if_same_as_old(new_csv_path=data.merged_sc_crystal_db)
    
    print_title('Step 4: Artificial (aka synthetic) doping.')
    perform_synthetic_doping(
                        input_file=data.merged_sc_crystal_db,
                        output_cif_dir=data.artificially_doped_cif_dir,
                        output_file=data.artificially_doped_db,
                        output_file_excluded=data.artificially_doped_db_excluded,
                        database=data.database,
                        n_cpus=n_cpus,
                        symprec=symprec
                        )
    check_if_same_as_old(new_csv_path=data.artificially_doped_db)
    
    symprec = 0.01  # historically different symprec
    print_title('Step 5: Generate MAGPIE and SOAP features and graphs.')
    generate_features(
                        input_csv_data=data.artificially_doped_db,
                        output_graph_dir=data.db_graph_dir,
                        output_csv_data=data.db_with_features,
                        excluded_output_csv_data=data.db_with_features_excluded,
                        rcut=rcut,
                        nmax=nmax,
                        lmax=lmax,
                        sigma=sigma,
                        crossover=crossover,
                        n_pca_components=n_pca_components,
                        n_cpus=n_cpus,
                        symprec=symprec
                        # generate_graphs=generate_graphs
                        )
    check_if_same_as_old(new_csv_path=data.db_with_features)
    
    print_title('Step 6: Filter best matches and prepare final database.')
    select_best_matches_and_prepare_df(
                                        input_csv_data=data.db_with_features,
                                        output_graph_dir=data.final_graph_dir,
                                        output_csv_data=data.final_3DSC,
                                        criteria=criteria,
                                        n_exclude_if_more_structures=n_exclude_if_more_structures
                                        )
    check_if_same_as_old(new_csv_path=data.final_3DSC, compare_overlap=False)   
         
    duration = datetime.datetime.now() - starttime
    print(f'Duration: {duration}') 


if __name__ == '__main__':
    
    database = 'ICSD'
    n_cpus = 1
    
    args = parse_input_parameters()
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    
    main(database, n_cpus)
                    