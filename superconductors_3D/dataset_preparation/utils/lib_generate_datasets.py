#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:15:00 2021

@author: timo
This script contains functions for generating final datasets.
"""

import pandas as pd
import chemml.chem.magpie_python as magpie
import copy



def MAGPIE_features(chem_formulas):
    """Calculates MAGPIE features for all given chemical formulas and returns a df with them.
    """
    # Get list with all compositions
    all_comps = []
    for chem_formula in chem_formulas:    
        comp = magpie.CompositionEntry()
        comp_dict = comp.parse_composition(chem_formula)
        comp.set_composition(amounts=comp_dict.values(), element_ids=comp_dict.keys())
        all_comps.append(copy.deepcopy(comp))
    
    # Generate features for all compositions
    # Same features as in 2017 Stanev.ArithmeticError
    feature_generators = [magpie.ValenceShellAttributeGenerator(), magpie.ElementalPropertyAttributeGenerator(), magpie.StoichiometricAttributeGenerator(), magpie.IonicityAttributeGenerator()]
    for f in feature_generators:
        features = f.generate_features(entries=all_comps)
        try:
            magpie_features = magpie_features.join(features)
        except UnboundLocalError:
            magpie_features = features
    
    # Check for NAN values
    not_nan_cols = ~magpie_features.isna().any(axis=0)
    magpie_features = magpie_features.loc[:, not_nan_cols]
    print(f"There were {sum(~not_nan_cols)} columns excluded because of missing values.")
    
    n_magpie_features = magpie_features.shape[1]
    print(f'Number of calculated MAGPIE features: {n_magpie_features}.')
    
    assert magpie_features.notna().all().all()
    
    return(magpie_features)

















