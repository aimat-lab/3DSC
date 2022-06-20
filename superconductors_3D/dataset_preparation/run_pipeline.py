#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:53:39 2022

@author: Timo Sommer

Runs the whole matching and artificial doping pipeline, from cleaning the SuperCon and the crystal structure database until outputting the final database.
"""
from superconductors_3D.utils.projectpaths import projectpath
import pandas as pd
import os
from tqdm import tqdm
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
from superconductors_3D.dataset_preparation.dataset_cleaning._0_extract_cifs import extract_cifs
from superconductors_3D.dataset_preparation.dataset_cleaning._1_clean_cifs import clean_cifs
from superconductors_3D.dataset_preparation.dataset_cleaning._2_0_clean_SuperCon import clean_SuperCon
from superconductors_3D.dataset_preparation.dataset_cleaning._2_1_clean_MP import clean_MP
from superconductors_3D.dataset_preparation.dataset_cleaning._2_2_clean_ICSD import clean_ICSD
from superconductors_3D.dataset_preparation._3_match_SC_and_3D_crystals import match_SC_and_3D_crystals
from superconductors_3D.dataset_preparation._4_synthetic_doping import synthetic_doping
from superconductors_3D.dataset_preparation._5_generate_features import generate_features
from superconductors_3D.dataset_preparation._6_select_best_matches_and_prepare_df import select_best_matches_and_prepare_df


class Data_Paths():
    
    def __init__(self, data_dir, supercon_csv, crystal_db_csv, crystal_db_name, crystal_db_cifs_dir):
        self.dir = data_dir
        self.supercon_csv = supercon_csv
        self.database = crystal_db_name
        self.crystal_db_csv = crystal_db_csv
        self.crystal_db_cifs_dir = crystal_db_cifs_dir
        
        # print(f'Change directory into {self.dir}.')
        # os.chdir(self.dir)
        
        
        # Step 1: Output of cleaning cifs.
        self.cifs_normalized = os.path.join(self.dir, 'source', self.database, 'cleaned', f'1_all_data_{self.database}_cifs_normalized.csv')
        self.cifs_normalized_excluded = os.path.join(self.dir, 'source', self.database, 'cleaned', f'excluded_1_all_data_{self.database}_cifs_normalized.csv')
        self.output_dir_cleaned_cifs = os.path.join('superconductors_3D', 'data', 'source', self.database, 'cleaned', 'cifs')
        self.cifs_normalized_comment = f"All the data from {self.crystal_db_csv}, but the cif files are cleaned/normalised by reading them in a pymatgen structure and writing them in a cif file again. Additionally chemical formula, spacegroup and crystal system are calculated by pymatgen and written in own columns. If a cif file has too bad errors the entry is excluded."
        self.cifs_normalized_comment_excluded = f"All the data that was not included in {self.cifs_normalized} because the cif file was too bad."


        # Step 2: Output of cleaning Supercon.
        self.supercon_cleaned = os.path.join(self.dir, 'source', 'SuperCon', 'cleaned', '2.0_all_data_SuperCon_cleaned.csv')
        self.supercon_cleaned_excluded = os.path.join(self.dir, 'source', 'SuperCon', 'cleaned', 'excluded_2.0_all_data_SuperCon_cleaned.csv')
        self.supercon_cleaned_comment = f'All the cleaned data from the Supercon main table {self.supercon_csv}.'
        self.supercon_cleaned_comment_excluded = f'All the data that was not included in {self.supercon_cleaned} because it was filtered out.'
        
        # Step 2.2: Output of cleaning the crystal database
        self.crystal_db_cleaned = os.path.join(self.dir, 'source', self.database, 'cleaned', f'2.1_all_data_{self.database}_cleaned.csv')
        self.crystal_db_cleaned_excluded = os.path.join(self.dir, 'source', self.database, 'cleaned', f'excluded_2.1_all_data_{self.database}_cleaned.csv')
        self.crystal_db_cleaned_comment = f'All the data from {self.cifs_normalized}, but the important columns are cleaned.'
        self.crystal_db_cleaned_comment_excluded = f'All the data that was not included in {self.crystal_db_cleaned} because it was filtered out.'
        # Only for the ICSD:
        self.icsd_type_filename = os.path.join(self.dir, 'source', 'ICSD', 'raw', 'ICSD_content_type.csv')  
        
        # Step 3: Match SuperCon and 3D crystal structure entries by chemical formula
        self.merged_sc_crystal_db = os.path.join(self.dir, 'intermediate', self.database, f'3_SC_{self.database}_matches.csv')
        self.merged_sc_crystal_db_comment = f'The merged dataset of the Supercon and the {self.database} based on the chemical formula.'
        
        # Step 4: Artificial (or synthetic) doping.
        self.artificially_doped_cif_dir = os.path.join('superconductors_3D', 'data', 'final', self.database, 'cifs')    
        self.artificially_doped_db = os.path.join(self.dir, 'intermediate', self.database, f'4_SC_{self.database}_synthetically_doped.csv')
        self.artificially_doped_db_excluded = os.path.join(self.dir, 'intermediate', self.database, f'excluded_4_SC_{self.database}_synthetically_doped.csv')    
        
        # Step 5: Generate SOAP and MAGPIE features and graphs.
        self.db_graph_dir = os.path.join('superconductors_3D', 'intermediate', self.database, 'graphs')
        self.db_with_features = os.path.join(self.dir, 'intermediate', self.database, f'5_features_SC_{self.database}.csv')
        self.db_with_features_excluded = os.path.join(self.dir, 'intermediate', self.database, f'excluded_5_features_SC_{self.database}.csv')
        
        # Step 6: Select best matches and prepare database
        self.final_graph_dir = os.path.join('superconductors_3D', 'data', 'final', self.database, 'graphs')
        self.final_3DSC = os.path.join(self.dir, 'final', self.database, f'SC_{self.database}_matches.csv')
    
    def create_directories(self):
        """Creates the needed directories for the datasets.
        """
        print(f"Create needed directories in {self.dir} if they don't exist yet.")
        os.makedirs(self.dir, exist_ok=True)        
        
        dirs = [
                os.path.join('final', self.database, 'cifs'),
                os.path.join('final', self.database, 'graphs'),
                os.path.join('intermediate', self.database, 'graphs'),
                os.path.join('source', self.database, 'raw', 'cifs'),
                os.path.join('source', self.database, 'cleaned', 'cifs'),
                os.path.join('source', 'SuperCon', 'cleaned'),
                os.path.join('source', 'SuperCon', 'raw')
                ]        
        for d in dirs:
            new_dir = os.path.join(self.dir, d)
            os.makedirs(new_dir, exist_ok=True)


        





        
if __name__ == '__main__':
        
    n_cpus = 1
    crystal_database = 'MP'
    data_dir = '/home/timo/superconductors_3D/superconductors_3D/data'
    supercon_csv = os.path.join(data_dir, 'source', 'SuperCon', 'raw', 'Supercon_data_by_2018_Stanev.csv')
    crystal_db_csv = os.path.join(data_dir,  'source', crystal_database, 'raw', f'0_all_data_{crystal_database}.csv')
    crystal_db_cifs_dir = os.path.join('superconductors_3D', 'data', 'source', crystal_database, 'raw', 'cifs')
    
    
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
    # Reduce high-dimensional SOAP features using PCA
    n_pca_components = 100
    # Hyperparameters for Disordered SOAP (here the same as from 2021 Fung: Benchmarking graph neural networks for materials chemistry) --> Makes around 8000-dimensional features
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
    sc_frac = 0.1
    crystal_db_frac = 0.03
    


    
    print(f'Crystal structure database: {crystal_database}')
    data = Data_Paths(
                        data_dir=data_dir,
                        supercon_csv=supercon_csv,
                        crystal_db_csv=crystal_db_csv,
                        crystal_db_name=crystal_database,
                        crystal_db_cifs_dir=crystal_db_cifs_dir
                        )
    
    print("Create directories if they don't exist yet.")
    data.create_directories()
    
    
    
    print('Step 1: Clean cifs.')
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
                crystal_db_frac=crystal_db_frac
                )

    
    
    print('Step 2.0: Clean SuperCon.')       
    clean_SuperCon(
                    in_filename=data.supercon_csv,
                    out_filename=data.supercon_cleaned,
                    out_excluded_filename=data.supercon_cleaned_excluded,
                    comment=data.supercon_cleaned_comment,
                    comment_excluded=data.supercon_cleaned_comment_excluded,
                    sc_frac=sc_frac
                    )
    
    print(f'Step 2.1: Clean crystal database {crystal_database}.')
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
    
    
    
    print('Step 3: Match SuperCon and 3D crystal structure entries by chemical formula.')
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



    print('Step 4: Artificial (or synthetic) doping.')
    synthetic_doping(
                        input_file=data.merged_sc_crystal_db,
                        output_cif_dir=data.artificially_doped_cif_dir,
                        output_file=data.artificially_doped_db,
                        output_file_excluded=data.artificially_doped_db_excluded,
                        n_cpus=n_cpus,
                        symprec=symprec
                        )
    
    
    
    print('Step 5: Generate MAGPIE and SOAP features and graphs.')
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
                        n_pca_components=n_pca_components
                        )
    
    
    
    print('Step 6: Filter best matches and prepare final database.')
    select_best_matches_and_prepare_df(
                                        input_csv_data=data.db_with_features,
                                        output_graph_dir=data.final_graph_dir,
                                        output_csv_data=data.final_3DSC,
                                        criteria=criteria,
                                        n_exclude_if_more_structures=n_exclude_if_more_structures
                                        )
