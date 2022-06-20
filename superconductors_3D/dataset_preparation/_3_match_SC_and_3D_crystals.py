#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:48:01 2021

author: timo
This script takes csv files of the Supercon database and crystal databases (ICSD, COD and Materials Project) and tries to match the entries based on the chemical formula. Not only the same chemical formula is matched but similar ones as well. The similarity metrics are written to the columns of the output csv.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os
import pandas as pd
import pymatgen as mg
import re
from joblib import Parallel, delayed, cpu_count
import copy
from collections import namedtuple
import numpy as np
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv, movecol
from superconductors_3D.dataset_preparation.utils.check_dataset import standardise_chem_formula, if_valid_formula, get_chem_dict, entries_to_check
from superconductors_3D.dataset_preparation.utils.calc_similarities import check_numeric_deviation, calculate_numeric_deviation, calculate_similarity, similarity_chem_formula, similarity_lattice_constants, get_formula_similarity, get_structure_similarity
import itertools



def rough_filter(comp, composition_sc, composition_2, num_elements_2, df_sc, df_2, n_max_doping_elements):
    """Roughly filter entries that won't match anyway by number of elements and by checking which elements are in the cif formulas.
    """
    elements_sc = np.array(comp.split("-"))
    num_elements_sc = len(elements_sc)
    # Allow maximally one additional doped elements in the Supercon formula.
    if num_elements_sc >= 2:
        num_allowed_doped_el = n_max_doping_elements
    elif num_elements_sc == 1:
        num_allowed_doped_el = 0
    correct_num_elements = (num_elements_2 <= num_elements_sc) & (num_elements_2 >= num_elements_sc - num_allowed_doped_el)
    matched_index_2 = correct_num_elements & (np.isin(composition_2, elements_sc).sum(axis=1) == num_elements_2)
    
    # Quick check.
    num_same_comps = sum(comp == df_2['chemical_composition_2'])
    num_similar_comps = sum(matched_index_2)
    assert num_similar_comps >= num_same_comps
    # print(f'Check {comp}: {num_similar_comps} >= {num_same_comps}')
    
    # Get subgroup of entries with correct chemical system.
    matched_idx_sc = comp == composition_sc
    df_sc_match = df_sc[matched_idx_sc].reset_index()
    df_2_match = df_2[matched_index_2].reset_index() 
    
    return(df_sc_match, df_2_match)



class MergeDatasets():
    """This class is for merging the Supercon Dataset with other datasets that contain the cif files of the crystals.
    """
    
    def __init__(self, df_sc, df_2, n_cpus, n_max_doping_elements, lower_max_relcutoff, lower_total_relcutoff, lower_min_abscutoff, higher_max_relcutoff, higher_total_relcutoff, higher_min_abscutoff):        
        self.df_sc = df_sc
        self.df_2 = df_2
        self.df_sc_no_match = pd.DataFrame()
        
        self.n_cpus = n_cpus
        self.n_max_doping_elements = n_max_doping_elements
        self.lower_max_relcutoff = lower_max_relcutoff
        self.lower_total_relcutoff = lower_total_relcutoff
        self.lower_min_abscutoff = lower_min_abscutoff
        self.higher_max_relcutoff = higher_max_relcutoff
        self.higher_total_relcutoff = higher_total_relcutoff
        self.higher_min_abscutoff = higher_min_abscutoff
        
        # Initialise data dictionary.
        all_columns = df_sc.columns.to_series().append(df_2.columns.to_series(), verify_integrity=True)
        self.similarity_features = ['formula_similarity', 'totreldiff']
        all_columns = all_columns.append(pd.Series(self.similarity_features), verify_integrity=True)
        all_columns = all_columns.sort_values()
        self.datadict = {}
        for col in all_columns.values:
            self.datadict[col] = []

    
    def match_entries(self, row_sc, row2):
        """Compares chemical formula and returns if they are similar.
        """
        formula_sc = row_sc.formula_sc
        formula_2 = row2.formula_2

        # Get sorted dictionaries with chemical elements as keys and quantities as values.
        chemdict_sc = get_chem_dict(formula_sc)
        chemdict_2 = get_chem_dict(formula_2)
        all_elements1 = list(chemdict_sc.keys())
        all_elements2 = list(chemdict_2.keys())
        
        # Check similarity of formulas.
        formula_similarity, totreldiff = get_formula_similarity(
                                                    chemdict_sc,
                                                    chemdict_2,
                                                    self.lower_max_relcutoff,
                                                    self.lower_total_relcutoff,
                                                    self.lower_min_abscutoff, 
                                                    self.higher_max_relcutoff, 
                                                    self.higher_total_relcutoff, 
                                                    self.higher_min_abscutoff
                                                                )
        if pd.isna(formula_similarity):
            # Return because formulas are dissimilar.
            return(False, [])
        
        # Append all data to the dictionary that hosts the merged data.
        similarity_dict = {
                            'formula_similarity': formula_similarity,
                            'totreldiff': totreldiff
                           }
        # Append features of both datasets to single row in new df together with similarities.
        cols_similarity = list(similarity_dict.keys())
        assert sorted(self.similarity_features) == sorted(cols_similarity)
        dict_sc = row_sc._asdict()
        cols_sc = list(dict_sc.keys())
        dict_2 = row2._asdict()
        cols_2 = list(dict_2.keys())
        
        all_cols = cols_sc + cols_2 + cols_similarity
        all_cols = [col for col in all_cols if not col in ['index', 'Index']]
        assert len(all_cols) == len(np.unique(all_cols))
        row = {}
        for col in all_cols:        
            if col in cols_sc:
                row[col] = dict_sc[col]
            elif col in cols_2:
                row[col] = dict_2[col]
            elif col in cols_similarity:
                row[col] = similarity_dict[col]
        
        return(True, row)
    
    def match_compositions(self, comp, composition_sc, composition_2, num_elements_2, df_sc, df_2):
        # Roughly filter out combinations that are very different (very fast).
        df_sc_match, df_2_match = rough_filter(comp, composition_sc, composition_2, num_elements_2, df_sc, df_2, self.n_max_doping_elements)
        # Start matching the rows of Supercon and crystal database.
        all_rows = []
        for row_sc in df_sc_match.itertuples():
            for row2 in df_2_match.itertuples():
                match, row = self.match_entries(row_sc, row2)
                if match == True:
                    all_rows.append(row)

        return(all_rows)
                
    def merge_csvs(self, output_path, comment):
        """Merges the two dataframes so that each row consists of the data from a superconductor from the Supercon and the data of a matched entry in the COD or another database.
        """
        
        df_sc = self.df_sc
        df_2 = self.df_2
        
        composition_2 = df_2["chemical_composition_2"].apply(lambda x: x.split("-")).tolist()
        num_elements_2 = np.array([len(list_2) for list_2 in composition_2])
        composition_2 = np.array(list(itertools.zip_longest(*composition_2,fillvalue=np.nan))).T
        
        # Get new composition arrays for the loop.
        composition_sc = df_sc["chemical_composition_sc"].to_numpy()
        unique_comp = np.unique(composition_sc)
        
        # Iterate over all compositions and test each entry in the Supercon with each entry in the crystal database.
        # print(f'Matching {len(unique_comp)} compositions.')
        with Parallel(n_jobs=self.n_cpus, verbose=1) as parallel:
            data1 = parallel(delayed(self.match_compositions)(comp, composition_sc, composition_2, num_elements_2, df_sc, df_2) for comp in unique_comp)

        # Flatten list.
        data = []
        for comp_data in data1:
            data += comp_data
        
        # Save dataset.
        df = pd.DataFrame(data=data)
        write_to_csv(df, output_path, comment)
        print("Saved merged dataframe of Supercon with cif database.") 
        return(df)
        
def match_SC_and_3D_crystals(input_sc, input_2, out_merged_sc_cif_file, comment, n_cpus, n_max_doping_elements, lower_max_relcutoff, lower_total_relcutoff, lower_min_abscutoff, higher_max_relcutoff, higher_total_relcutoff, higher_min_abscutoff):
    
    # Read in datasets.
    df_sc = pd.read_csv(input_sc, header=1)     # SuperCon
    df_2 = pd.read_csv(input_2, header=1)       # Crystal database
    
    # Start matching and merging datasets.
    print("Start merging the Supercon and the cif database.")
    merge_sc_cif = MergeDatasets(
                                    df_sc=df_sc,
                                    df_2=df_2,
                                    n_cpus=n_cpus,
                                    n_max_doping_elements=n_max_doping_elements,
                                    lower_max_relcutoff=lower_max_relcutoff,
                                    lower_total_relcutoff=lower_total_relcutoff,
                                    lower_min_abscutoff=lower_min_abscutoff,
                                    higher_max_relcutoff=higher_max_relcutoff,
                                    higher_total_relcutoff=higher_total_relcutoff,
                                    higher_min_abscutoff=higher_min_abscutoff
                                    )
    
    df_sc_cif = merge_sc_cif.merge_csvs(
                                        output_path = out_merged_sc_cif_file,
                                        comment = comment
                                        ) 
   
   
if __name__ == "__main__":
    
    database = 'ICSD'     # change this
    n_cpus = 3          # change this
    
    
    
    # Parse input and overwrite.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    args = parser.parse_args()
    
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    
    
    input_sc = projectpath('data', 'source', 'SuperCon', 'cleaned', '2.0_all_data_SuperCon_cleaned.csv')
    input_2 = projectpath('data', 'source', database, 'cleaned', f'2.1_all_data_{database}_cleaned.csv')
    out_merged_sc_cif_file = projectpath('data', 'intermediate', database, f'3_SC_{database}_matches.csv')
    
    comment = f'The merged dataset of the Supercon and the {database} based on the chemical formula.'
    
    # Hyperparameters for matching conditions
    
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
    

    match_SC_and_3D_crystals(input_sc, input_2, out_merged_sc_cif_file, comment, n_cpus, n_max_doping_elements, lower_max_relcutoff, lower_total_relcutoff, lower_min_abscutoff, higher_max_relcutoff, higher_total_relcutoff, higher_min_abscutoff)
    
    
    
