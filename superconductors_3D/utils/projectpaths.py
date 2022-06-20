#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:18:34 2021

@author: Timo Sommer

Contains the path to the source directory of this paper.
"""
import os 
import inspect
import superconductors_3D

source_dir = os.path.dirname(inspect.getfile(superconductors_3D))

def projectpath(*relpaths):
    return os.path.join(source_dir, *relpaths)


class Data_Paths():
    
    def __init__(self, crystal_db_name):
        self.database = crystal_db_name
        self.dir = projectpath('data')
        self.supercon_csv = os.path.join(self.dir, 'source', 'SuperCon', 'raw', 'Supercon_data_by_2018_Stanev.csv')
        self.crystal_db_csv = os.path.join(self.dir,  'source', self.database, 'raw', f'{self.database}_subset.csv')
        self.crystal_db_cifs_dir = projectpath('data', 'source', self.database, 'raw', 'cifs')        
        
        # Step 1: Output of cleaning cifs.
        self.cifs_normalized = os.path.join(self.dir, 'source', self.database, 'cleaned', f'1_all_data_{self.database}_cifs_normalized.csv')
        self.cifs_normalized_excluded = os.path.join(self.dir, 'source', self.database, 'cleaned', f'excluded_1_all_data_{self.database}_cifs_normalized.csv')
        self.output_dir_cleaned_cifs = os.path.join('data', 'source', self.database, 'cleaned', 'cifs')
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
        self.artificially_doped_cif_dir = os.path.join('data', 'final', self.database, 'cifs')    
        self.artificially_doped_db = os.path.join(self.dir, 'intermediate', self.database, f'4_SC_{self.database}_synthetically_doped.csv')
        self.artificially_doped_db_excluded = os.path.join(self.dir, 'intermediate', self.database, f'excluded_4_SC_{self.database}_synthetically_doped.csv')    
        
        # Step 5: Generate SOAP and MAGPIE features and graphs.
        self.db_graph_dir = os.path.join('data', 'intermediate', self.database, 'graphs')
        self.db_with_features = os.path.join(self.dir, 'intermediate', self.database, f'5_features_SC_{self.database}.csv')
        self.db_with_features_excluded = os.path.join(self.dir, 'intermediate', self.database, f'excluded_5_features_SC_{self.database}.csv')
        
        # Step 6: Select best matches and prepare database
        self.final_graph_dir = os.path.join('data', 'final', self.database, 'graphs')
        self.final_3DSC = os.path.join(self.dir, 'final', self.database, f'3DSC_{self.database}.csv')
    
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