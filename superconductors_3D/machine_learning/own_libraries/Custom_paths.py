#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 06:41:53 2021

@author: timo
This library contains paths that I use frequently.
"""


from os.path import join, exists

# =============================================================================
# class Custom_paths():
#     
#     def init(self):
#         # Raw cif datasets
#         self.raw_cif_dir = self.dataset_dir_path() # needs work
#         
#         ## ICSD
#         self.raw_icsd_dir = join(self.raw_cif_dir, 'ICSD')
#         
#         self.raw_icsd_cifs = join(self.raw_icsd_dir, 'cif')
#         self.all_data_icsd = join(self.raw_icsd_dir, 'All_data_ICSD.csv')
#         self.icsd_content_type = join(self.raw_icsd_dir, 'ICSD_content_type.csv')
#         self.icsd_data_from_api = join(self.raw_icsd_dir, 'ICSD_data_from_API.csv')
#         
#         
#         # Cleaned cif datasets
#         ## ICSD
#         ### cif
#         self.cleaned_icsd_dir = '/home/timo/Masterarbeit/Rechnungen/Datasets/cif_files/ICSD'
#         self.cleaned_icsd_cifs = join(self.cleaned_icsd_dir, 'pymatgen_cleaned_cifs')
#         ### csv
#         self.All_data_ICSD_pymatgen_cifs_normalised_cleaned = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/ICSD/All_data_ICSD_pymatgen_cifs_normalised_cleaned.csv'
#         self.Excluded_All_data_ICSD_pymatgen_cifs_normalised_cleaned = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/ICSD/Excluded_All_data_ICSD_pymatgen_cifs_normalised_cleaned.csv'
#         
#         
#         # Supercon dataset
#         self.supercon_dir = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/Supercon'
#         self.All_data_Supercon_cleaned = join(self.supercon_dir, 'All_data_Supercon_cleaned.csv')
#         
#         self.All_data_Supercon_only_main_table = join(self.supercon_dir, 'All_data_Supercon_only_main_table.csv')
#         self.All_data_Supercon_with_tcfig_raw_content = join(self.supercon_dir, 'All_data_Supercon_with_tcfig_raw_content.csv')
#         self.All_poss_sc = join(self.supercon_dir, 'All_poss_sc.csv')
#         self.Excluded_All_data_Supercon_cleaned = join(self.supercon_dir, 'Excluded_All_data_Supercon_cleaned.csv')
#         self.Excluded_Supercon_tcfig_content_data = join(self.supercon_dir, 'Excluded_Supercon_tcfig_content_data.csv')
#         self.Supercon_tcfig_content_data = join(self.supercon_dir, 'Supercon_tcfig_content_data.csv')
#         self.Supercon_tcfig_content_data_not_sc = join(self.supercon_dir, 'Supercon_tcfig_content_data_not_sc.csv')
#         self.Tried_Hosono_by_Konno = join(self.supercon_dir, 'Tried_Hosono_by_Konno.csv')
#         
#         
#         # Superconductor cif matches
#         ## After matching
#         self.Sc_ICSD_chem_formula_merged = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/Sc_cif_matches/Sc_ICSD_chem_formula_merged.csv'
#         ## After choosing best matches
#         self.Good_matches_Sc_ICSD = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/Sc_cif_matches/Good_matches_Sc_ICSD.csv'
#         self.Excluded_Good_matches_Sc_ICSD = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/Sc_cif_matches/Excluded_Good_matches_Sc_ICSD.csv'
#         ## After manipulating
#         self.Sc_ICSD_merged_and_manipulated = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/Manipulated_Sc_cif_matches/Sc_ICSD_merged_and_manipulated.csv'
#         self.Excluded_Sc_ICSD_merged_and_manipulated = '/home/timo/Masterarbeit/Rechnungen/Datasets/Sonstiges/Manipulated_Sc_cif_matches/Excluded_Sc_ICSD_merged_and_manipulated.csv'
#         
#         
#         # Cleaned and manipulated icsd and sc cifs for chemical formula matching
#         self.sc_icsd_chem_formula_manipulated_cifs = '/home/timo/Masterarbeit/Rechnungen/Datasets/cif_files/Sc_ICSD_chem_formula_manipulated'
#         
#         
# 
#     
#     def cif_dataset_path(self, dataset):
#         raw_cif_dir = join(self.raw_cif_dir, dataset)
#         
#         raw_icsd_cifs = join(raw_cif_dir, 'cif')
#         all_data_icsd = join(raw_cif_dir, f'All_data_ICSD.csv')
# =============================================================================
    
def project_path(path=None):
    """Returns path to the directory 'Masterarbeit' either on the local laptop or the cluster.
    """
    local_path = '/home/timo/Masterarbeit'
    cluster_path = '/home/kit/stud/uoeci/Masterarbeit'
    if exists(local_path):
        path = local_path if path == None else join(local_path, path)
    elif exists(cluster_path):
        path = cluster_path if path == None else join(cluster_path, path)
    else:
        raise Warning(f'No project path of {local_path} or {cluster_path} found.')
    return(path)


def dataset_dir_path():
    """Returns path to the database directory."""
    cluster_path = '/pfs/work7/workspace/scratch/uoeci-cifs-0/Masterarbeit/Datasets'
    local_path = '/media/timo/ext42/Masterarbeit/Datasets'
    if exists(cluster_path):
        path = cluster_path
    elif exists(local_path):
        path = local_path
    else:
        raise Warning('Database directory not found.')
    return(path)


def MP_API_key_path():
    """Returns path to the MP API key."""
    cluster_path = '/home/kit/stud/uoeci/mp_credentials.json'
    local_path = '/home/timo/mp_credentials.json'
    if exists(cluster_path):
        api_key_path = cluster_path
    elif exists(local_path):
        api_key_path = local_path
    else:
        raise Warning('MP API key not found.')
    return(api_key_path)