#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:57:05 2021

@author: Timo Sommer

This is a collection of variables for the analysis.
"""
import warnings

def get_MAGPIE_features(database):
    """Returns MAGPIE features, independently of the database.
    """
    return magpie_features

def get_structural_features(database):
    """Returns structural features. Depends on the database.
    """
    if database == 'MP':
        feats = soap_features + electronic_features + sym_features + lattice_features + ['crystal_temp_2']
    elif database == 'ICSD':
        feats = soap_features + sym_features + lattice_features + ['crystal_temp_2']
    else:
        raise ValueError('Unknown database.')
    return feats

def get_PCA_structural_features(database):
    """Returns structural features. Depends on the database.
    """
    if database == 'MP':
        feats = pca_soap_features + electronic_features + sym_features + lattice_features + ['crystal_temp_2']
    elif database == 'ICSD':
        feats = pca_soap_features + sym_features + lattice_features + ['crystal_temp_2']
    else:
        raise ValueError('Unknown database.')
    return feats

soap_features = [f'SOAP_{i}' for i in range(8715)]

pca_soap_features = [f'PCA_SOAP_{i}' for i in range(100)]

electronic_features = ['band_gap_2', 'energy_2', 'energy_per_atom_2', 'formation_energy_per_atom_2', 'total_magnetization_2', 'num_unique_magnetic_sites_2', 'true_total_magnetization_2']
lattice_features = ['lata_2', 'latb_2', 'latc_2']
sym_features = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic',  'tetragonal', 'triclinic', 'trigonal', 'primitive', 'base-centered', 'body-centered', 'face-centered']

magpie_features = ['MAGPIE_frac_sValence', 'MAGPIE_frac_pValence', 'MAGPIE_frac_dValence', 'MAGPIE_frac_fValence', 'MAGPIE_mean_Number', 'MAGPIE_maxdiff_Number', 'MAGPIE_dev_Number', 'MAGPIE_max_Number', 'MAGPIE_min_Number', 'MAGPIE_most_Number', 'MAGPIE_mean_MendeleevNumber', 'MAGPIE_maxdiff_MendeleevNumber', 'MAGPIE_dev_MendeleevNumber', 'MAGPIE_max_MendeleevNumber', 'MAGPIE_min_MendeleevNumber', 'MAGPIE_most_MendeleevNumber', 'MAGPIE_mean_AtomicWeight', 'MAGPIE_maxdiff_AtomicWeight', 'MAGPIE_dev_AtomicWeight', 'MAGPIE_max_AtomicWeight', 'MAGPIE_min_AtomicWeight', 'MAGPIE_most_AtomicWeight', 'MAGPIE_mean_MeltingT', 'MAGPIE_maxdiff_MeltingT', 'MAGPIE_dev_MeltingT', 'MAGPIE_max_MeltingT', 'MAGPIE_min_MeltingT', 'MAGPIE_most_MeltingT', 'MAGPIE_mean_Column', 'MAGPIE_maxdiff_Column', 'MAGPIE_dev_Column', 'MAGPIE_max_Column', 'MAGPIE_min_Column', 'MAGPIE_most_Column', 'MAGPIE_mean_Row', 'MAGPIE_maxdiff_Row', 'MAGPIE_dev_Row', 'MAGPIE_max_Row', 'MAGPIE_min_Row', 'MAGPIE_most_Row', 'MAGPIE_mean_CovalentRadius', 'MAGPIE_maxdiff_CovalentRadius', 'MAGPIE_dev_CovalentRadius', 'MAGPIE_max_CovalentRadius', 'MAGPIE_min_CovalentRadius', 'MAGPIE_most_CovalentRadius', 'MAGPIE_mean_Electronegativity', 'MAGPIE_maxdiff_Electronegativity', 'MAGPIE_dev_Electronegativity', 'MAGPIE_max_Electronegativity', 'MAGPIE_min_Electronegativity', 'MAGPIE_most_Electronegativity', 'MAGPIE_mean_NsValence', 'MAGPIE_maxdiff_NsValence', 'MAGPIE_dev_NsValence', 'MAGPIE_max_NsValence', 'MAGPIE_min_NsValence', 'MAGPIE_most_NsValence', 'MAGPIE_mean_NpValence', 'MAGPIE_maxdiff_NpValence', 'MAGPIE_dev_NpValence', 'MAGPIE_max_NpValence', 'MAGPIE_min_NpValence', 'MAGPIE_most_NpValence', 'MAGPIE_mean_NdValence', 'MAGPIE_maxdiff_NdValence', 'MAGPIE_dev_NdValence', 'MAGPIE_max_NdValence', 'MAGPIE_min_NdValence', 'MAGPIE_most_NdValence', 'MAGPIE_mean_NfValence', 'MAGPIE_maxdiff_NfValence', 'MAGPIE_dev_NfValence', 'MAGPIE_max_NfValence', 'MAGPIE_min_NfValence', 'MAGPIE_most_NfValence', 'MAGPIE_mean_NValance', 'MAGPIE_maxdiff_NValance', 'MAGPIE_dev_NValance', 'MAGPIE_max_NValance', 'MAGPIE_min_NValance', 'MAGPIE_most_NValance', 'MAGPIE_mean_NsUnfilled', 'MAGPIE_maxdiff_NsUnfilled', 'MAGPIE_dev_NsUnfilled', 'MAGPIE_max_NsUnfilled', 'MAGPIE_min_NsUnfilled', 'MAGPIE_most_NsUnfilled', 'MAGPIE_mean_NpUnfilled', 'MAGPIE_maxdiff_NpUnfilled', 'MAGPIE_dev_NpUnfilled', 'MAGPIE_max_NpUnfilled', 'MAGPIE_min_NpUnfilled', 'MAGPIE_most_NpUnfilled', 'MAGPIE_mean_NdUnfilled', 'MAGPIE_maxdiff_NdUnfilled', 'MAGPIE_dev_NdUnfilled', 'MAGPIE_max_NdUnfilled', 'MAGPIE_min_NdUnfilled', 'MAGPIE_most_NdUnfilled', 'MAGPIE_mean_NfUnfilled', 'MAGPIE_maxdiff_NfUnfilled', 'MAGPIE_dev_NfUnfilled', 'MAGPIE_max_NfUnfilled', 'MAGPIE_min_NfUnfilled', 'MAGPIE_most_NfUnfilled', 'MAGPIE_mean_NUnfilled', 'MAGPIE_maxdiff_NUnfilled', 'MAGPIE_dev_NUnfilled', 'MAGPIE_max_NUnfilled', 'MAGPIE_min_NUnfilled', 'MAGPIE_most_NUnfilled', 'MAGPIE_mean_GSvolume_pa', 'MAGPIE_maxdiff_GSvolume_pa', 'MAGPIE_dev_GSvolume_pa', 'MAGPIE_max_GSvolume_pa', 'MAGPIE_min_GSvolume_pa', 'MAGPIE_most_GSvolume_pa', 'MAGPIE_mean_GSbandgap', 'MAGPIE_maxdiff_GSbandgap', 'MAGPIE_dev_GSbandgap', 'MAGPIE_max_GSbandgap', 'MAGPIE_min_GSbandgap', 'MAGPIE_most_GSbandgap', 'MAGPIE_mean_GSmagmom', 'MAGPIE_maxdiff_GSmagmom', 'MAGPIE_dev_GSmagmom', 'MAGPIE_max_GSmagmom', 'MAGPIE_min_GSmagmom', 'MAGPIE_most_GSmagmom', 'MAGPIE_mean_SpaceGroupNumber', 'MAGPIE_maxdiff_SpaceGroupNumber', 'MAGPIE_dev_SpaceGroupNumber', 'MAGPIE_max_SpaceGroupNumber', 'MAGPIE_min_SpaceGroupNumber', 'MAGPIE_most_SpaceGroupNumber', 'MAGPIE_NComp', 'MAGPIE_Comp_L2Norm', 'MAGPIE_Comp_L3Norm', 'MAGPIE_Comp_L5Norm', 'MAGPIE_Comp_L7Norm', 'MAGPIE_Comp_L10Norm', 'MAGPIE_CanFormIonic', 'MAGPIE_MaxIonicChar', 'MAGPIE_MeanIonicChar']