#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:25:09 2021

@author: timo
This script is for testing the similarity calculation of chemical formulas from the Supercon and the cif databases.
The similarities for the chemical formulas are calculated like this:
    - Assert that in the Supercon formula there are the same or more elements than in the cif formula.
    - If there are more elements in the  Supercon formula and there is only one element in the cif formula, discard match.
    - Normalise the cif formula by the Supercon formula.
    - If there are elements in the Supercon formula that are not in the cif formula add these to the cif formula with quantity zero.
    - Calculate abs. diff, rel. diff. and tot. rel. diff.:
        - Same: Tot. rel. diff. = 0
        - Similar: Tot. rel. diff. < 0.05, rel. diff. < 0.10, abs. diff. < 0.15
        - Doped:   Tot. rel. diff. < 0.15, rel. diff. < 0.20, abs. diff. < 0.30
"""
import numpy as np
import pandas as pd
import copy
from superconductors_3D.dataset_preparation.utils.check_dataset import get_chem_dict

def get_formula_diff(formula1, formula2, mode):
    chemdict1 = get_chem_dict(formula1)
    chemdict2 = get_chem_dict(formula2)
    for el, val in chemdict2.items():
        if not el in chemdict1:
            chemdict1[el] = 0
    for el, val in chemdict1.items():
        if not el in chemdict2:
            chemdict2[el] = 0
    quantities1 = np.array([chemdict1[el] for el in sorted(chemdict1)])
    quantities2 = np.array([chemdict2[el] for el in sorted(chemdict2)])
    diffs, reldiffs, totreldiff = calculate_numeric_deviation(quantities1, quantities2)
    if mode == 'abs':
        diff = np.sum(diffs)
    elif mode == 'rel':
        diff = totreldiff
    elif mode == 'rel_per_element':
        diff = np.sum(reldiffs) / len(reldiffs)
    return(diff)

def similarity_lattice_constants(lat_sc, lat_2):
    """Checks if lattice constants are "similar" as defined. Takes tuples of the lattice constants.
    """
    single_relcutoff = 0.1
    total_relcutoff = 0.05
    # Remove nan from tuples.  
    lat_sc = copy.deepcopy(tuple(x for x in lat_sc if not np.isnan(x)))
    lat_2 = copy.deepcopy(tuple(x for x in lat_2 if not np.isnan(x)))
    len1, len2 = len(lat_sc), len(lat_2)


    # If lattice constants are not given for one side just return.
    if len1 == 0 or len2 == 0:
        return(None, np.nan)
    
    # If only one lattice constant is given this implies that all three are equal.
    if len1 == 1:
        lat_sc = lat_sc*3
        len1 = len(lat_sc)
    if len2 == 1:
        lat_2 =  lat_2*3
        len2 = len(lat_2)

    
    # Simply check deviations if there's the same number of lattice constants in each tuple.
    if (len1 == 3 and len2 == 3) or (len1 == 2 and len2 == 2):
        lat_sc_sorted, lat_2_sorted = sorted(lat_sc), sorted(lat_2)
        check = check_numeric_deviation(nums1 = lat_sc_sorted,
                                        nums2 = lat_2_sorted,
                                        single_relcutoff = single_relcutoff,
                                        total_relcutoff = total_relcutoff,
                                        single_abscutoff = np.inf,
                                        total_abscutoff = np.inf)
        if check == False:
            return(False, np.nan)
        
    # Check all possibilities of combinations if one set has three lattice constants and the other one has two.
    elif len1 == 2 or len2 == 2:
        for i, lat in enumerate((lat_sc, lat_2)):
            if len(lat) == 2:
                if i == 0:
                    otherlat = lat_2
                elif i == 1:
                    otherlat = lat_sc
                assert len(otherlat) == 3
                latposs1 = sorted([lat[0], lat[0], lat[1]])
                latposs2 = sorted([lat[0], lat[1], lat[1]])
                otherlat = sorted(otherlat)
                checks = []
                for latposs in (latposs1, latposs2):
                    check = check_numeric_deviation(nums1 = latposs,
                                                    nums2 = otherlat,
                                                    single_relcutoff = single_relcutoff,
                                                    total_relcutoff = total_relcutoff,
                                                    single_abscutoff = np.inf,
                                                    total_abscutoff = np.inf)
                    checks.append(check)
                if any(checks) == False:
                    return(False, np.nan)
                else:
                    # Assign the correct latposs to the correct lat.
                    if i == 0:
                        if checks[0] == True:
                            lat_sc = latposs1
                        elif checks[1] == True:
                            lat_sc = latposs2
                        else:
                            raise Warning("You should not end up here.")
                    elif i == 1:
                        if checks[0] == True:
                            lat_2 = latposs1
                        elif checks[1] == True:
                            lat_2 = latposs2
                        else:
                            raise Warning("You should not end up here.")
                    else:
                        raise Warning("You should not end up here.")
    else:
        raise Warning("You should not end up here, one of the conditions above seems faulty.")
    
    diffs, reldiffs, totreldiff = calculate_numeric_deviation(lat_sc, lat_2)
    
    return(True, totreldiff)


def similarity_chem_formula(chem_dict_sc, chem_dict_2, max_relcutoff, total_relcutoff, min_abscutoff):
    """Checks if the chemical formula is "similar" as defined. Assumes that chemdicts have the same elements and are sorted.
    """
    # To not modify the actual dictionary from out of this function.
    chemdict_sc, chemdict_2 = copy.deepcopy(chem_dict_sc), copy.deepcopy(chem_dict_2)
    
    assert chemdict_sc.keys() == chemdict_2.keys()
        
    quantities_sc = np.array(list(chemdict_sc.values()))
    quantities_2 = np.array(list(chemdict_2.values()))
    # Normalise in case one formula is conventional unit cell and the other formula is primitive unit cell.
    norm = np.sum(quantities_sc)/np.sum(quantities_2)
    quantities_2 = quantities_2*norm
    
    # Check differences of elements.
    diffs, reldiffs, totreldiff = calculate_numeric_deviation(quantities_sc, quantities_2)
    
    # First condition
    if totreldiff > total_relcutoff:
        return(False, totreldiff)
    
    # Second condition
    for diff, reldiff in zip(diffs, reldiffs):
        if diff > min_abscutoff and reldiff > max_relcutoff:
            return(False, totreldiff)
        
    return(True, totreldiff)


def calculate_numeric_deviation(nums1, nums2):
    """Calculates absolute and relative differences.
    """
    
    # Make sure nums are numpy arrays.
    nums1, nums2 = np.array(nums1), np.array(nums2)
    
    # Calculate differences.
    diffs = np.abs(nums1 - nums2)
    # Calculate relative differences.
    reldiffs = 2*diffs/(nums1 + nums2)
    # The formula for totreldiff is not the sum of reldiffs but rather the weighted sum of reldiffs. That means elements with low quantities are weighed less than elements with high quantities. This is the right behaviour, because otherwise small changes of doping of small amounts would dominate (e.g. if element1 has quantity 0.1 and element2 has quantity 0.15 they would already make a difference of about 40%, this should be weighed down by other elements with higher quantities.)
    totreldiff = 2*diffs.sum()/(nums1.sum() + nums2.sum())
    return(diffs, reldiffs, totreldiff)


def check_numeric_deviation(nums1, nums2, single_relcutoff, total_relcutoff, single_abscutoff, total_abscutoff):
    """Assumes sorted arrays nums1 and nums2 and checks if the numeric differences  are below a certain threshold.
    """
    diffs, reldiffs, totreldiff = calculate_numeric_deviation(nums1, nums2)
    if max(diffs) > single_abscutoff or max(reldiffs) > single_relcutoff or sum(diffs) > total_abscutoff or totreldiff > total_relcutoff:
        return(False)
    else:
        return(True)  


def bool_sum(bools):
    """Calculate the sum of an iterable of bools.
    
    True adds +1, False adds -1, None/NaN adds 0.
    """        
    result = 0
    for val in bools:
        if val == True:
            result += 1
        elif val == False:
            result += -1
        elif pd.isna(val):
            result += 0
        else:
            raise ValueError(val)
    return(result)

def get_structure_similarity(spacegroup_same, crystalgroup_same, lat_similar, lat_same):
    """Get a score for the similarity of the structures, range: [1,5]. The smaller the score the greater the similarity.
    """
    if (spacegroup_same and lat_similar) or lat_same:
        structure_similarity = 1
    elif spacegroup_same:
        structure_similarity = 2
    elif crystalgroup_same and lat_similar:
        structure_similarity = 3
    elif crystalgroup_same or lat_similar:
        structure_similarity = 4
    else:
        structure_similarity = 5
        
    return(structure_similarity)


def get_formula_similarity(chemdict_sc, chemdict_2, lower_max_relcutoff, lower_total_relcutoff, lower_min_abscutoff, higher_max_relcutoff, higher_total_relcutoff, higher_min_abscutoff):
    """Get a score for the similarity of the chemical formulas.
    """    
    elements_sc = list(chemdict_sc.keys())
    elements_2 = list(chemdict_2.keys())
    
    # Add additonally doped elements from the Supercon formula into the cif formula (i.e. pad chemical formula with zeros).
    assert len(elements_sc) >= len(elements_2)
    assert all([el in elements_sc for el in elements_2])
    if len(elements_sc) > len(elements_2):
        if len(elements_2) == 1:
            # If the cif formula is an element we don't want any doped matches because it's likely that even a bit doping changes the structure a lot.
            formula_similarity = np.nan
            totreldiff = np.nan
            return(formula_similarity, totreldiff)
    chemdict_2 = {el: chemdict_2[el] if el in elements_2 else 0. for el in elements_sc}
    
    # Check if chemical formulas are very close ('similar')
    formulas_similar, totreldiff = similarity_chem_formula(
                                        chem_dict_sc = chemdict_sc,
                                        chem_dict_2 = chemdict_2,
                                        max_relcutoff = lower_max_relcutoff,
                                        total_relcutoff = lower_total_relcutoff,
                                        min_abscutoff = lower_min_abscutoff
                                        )

    if totreldiff == 0:
        # relative formulas are the same.
        formula_similarity = 1
    elif formulas_similar == True:
        formula_similarity = 2
    else:
        formula_similarity = np.nan

    # Check if chemical formulas are relatively close ('doped').
    if pd.isna(formula_similarity): 
        formulas_doped, totreldiff = similarity_chem_formula(
                                        chem_dict_sc = chemdict_sc,
                                        chem_dict_2 = chemdict_2,
                                        max_relcutoff = higher_max_relcutoff,
                                        total_relcutoff = higher_total_relcutoff,
                                        min_abscutoff = higher_min_abscutoff
                                        )
        if formulas_doped == True:
            formula_similarity = 3
        
    return(formula_similarity, totreldiff)


def calculate_similarity(formula_similarity, structure_similarity):
    """Calculates a similarity score ([1,18]) based on several criteria. The smaller the score the better. 
    
    In detail: Determines the overall similarity from the similarity of the chemical formula and the similarity of the structure. This is done in a way so that I thought it would best capture the real similarity: The structure similarity is more important than the similarity of the chemical formula, except if the chemical formula similarity is 3 (doped). Then instead it is preferred to have a a chemical formula similarity of 1(same) but with a structure score that is lower by 1. (I.e. a way to generate the score is to first write all combinations and sorting them first by structure similarity and then by chemical formula similarity. Then all the entries with chemical formula similarity of 3 change position with the entry with chemical formula similarity of 1. Then the scores are just the row numbers each.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                """   
    # Determine similarity.
    if formula_similarity == 1 and structure_similarity == 1:
        similarity = 1
    elif formula_similarity == 1 and structure_similarity == 2:
        similarity = 2
    elif formula_similarity == 1 and structure_similarity == 3:
        similarity = 3    
    elif formula_similarity == 1 and structure_similarity == 4:
        similarity = 4
    elif formula_similarity == 1 and structure_similarity == 5:
        similarity = 5
    elif formula_similarity == 2 and structure_similarity == 1:
        similarity = 6
    elif formula_similarity == 2 and structure_similarity == 2:
        similarity = 7
    elif formula_similarity == 2 and structure_similarity == 3:
        similarity = 8
    elif formula_similarity == 2 and structure_similarity == 4:
        similarity = 9
    elif formula_similarity == 2 and structure_similarity == 5:
        similarity = 10
    elif formula_similarity == 3 and structure_similarity == 1:
        similarity = 11
    elif formula_similarity == 3 and structure_similarity == 2:
        similarity = 12
    elif formula_similarity == 3 and structure_similarity == 3:
        similarity = 13
    elif formula_similarity == 3 and structure_similarity == 4:
        similarity = 14
    elif formula_similarity == 3 and structure_similarity == 5:
        similarity = 15
    else:
        raise Warning("You should not end up here.")
        
    assert similarity == 5*(formula_similarity - 1) + structure_similarity
            
    return(similarity)

# =============================================================================
# # Test calculate_similarity:
# all_variables = ["formulas_same", "O_class", "formulas_similar", "formulas_doped", "spacegroup_same", "crystalgroup_same", "lat_similar", "lat_same"]
# test_df = pd.DataFrame([False, 2, False, True, False, False, False, True], index=all_variables).T
# testdict = {col: [] for col in all_variables}
# for formulas_same in [True, False]:
#     for O_class in [0, 1, 2]:
#         for formulas_similar in [True, False]:
#             for formulas_doped in [True, False]:
#                 for spacegroup_same in [True, np.nan, False]:
#                     for crystalgroup_same in [True, np.nan, False]:
#                         for lat_similar in [True, np.nan, False]:
#                             testdict["formulas_same"].append(formulas_same)
#                             testdict["O_class"].append(O_class)
#                             testdict["formulas_similar"].append(formulas_similar)
#                             testdict["formulas_doped"].append(formulas_doped)
#                             testdict["spacegroup_same"].append(spacegroup_same)
#                             testdict["crystalgroup_same"].append(crystalgroup_same)
#                             testdict["lat_similar"].append(lat_similar)
#                             testdict["lat_same"].append(False)
# 
# test_df = test_df.append(pd.DataFrame(testdict)).sort_values(by=all_variables, ascending=False)
# #%%
# def test_sim(row):
#     formulas_same = row["formulas_same"]
#     formulas_similar = row["formulas_similar"]
#     formulas_doped = row["formulas_doped"]
#     O_class = row["O_class"]
#     spacegroup_same = row["spacegroup_same"]
#     crystalgroup_same = row["crystalgroup_same"]
#     lat_same = row["lat_same"]
#     lat_similar = row["lat_similar"]
#     
#     sim = calculate_similarity(formulas_same, formulas_similar, formulas_doped, O_class, spacegroup_same, crystalgroup_same, lat_same, lat_similar)
#     return(sim)
# 
# test_df["similarity"] = test_df.apply(test_sim, axis=1)
# =============================================================================








if __name__ == "__main__":
    
    data = [
            ['Tc0.9W0.1', 'Tc0.85W0.15', True],
            ["Nb0.9Ca0.1Ge3", "Nb1Ge3", True],
            ["Nb0.7Ge0.3", "Nb0.7Au0.3", False],
            ["Nb0.7Ge0.3", "Nb0.7Ge0.3Au0.3", False],
            ["Nb1Ge0.85As0.1", "Nb1Ge1", True],
            ["Nb1Ge0.9As0.1C1.1N0.9", "Nb1Ge1C1N1", True],
            ["Nb1Ge0.6As0.4C1.4N0.6", "Nb1Ge1C1N1", False],
            ["Nb1GeC1.8N0.2", "Nb1Ge1C1N1", False],
            ["Nb1Ge0.8As0.2C1.2N0.8", "Nb1Ge1C1N1.1", np.nan],
            ["Nb1Ge0.9As0.1", "Nb1Ge1", True],
            ["Nb0.5Ge0.5", "Nb0.45Ge0.55", True],
            ["Nb0.5Ge0.5", "Nb", False],
            ["Nb0.6Ge0.4", "Nb", False],
            ["Nb0.7Ge0.3", "Nb", False],
            ["Nb0.8Ge0.2", "Nb", False],
            ["Nb0.9Ge1.1", "Nb1Ge1", True],
            ["Nb0.9Ge1.1", "Nb1Ge1", True],
            ["In1Te1.002", "In1Te1", True],
            ["Y1Ba2Cu3O6.8 ", "Y1Ba2Cu3O7", True],
            ["Y0.6Ca0.4Ba2Cu3O7 ", "Y1Ba2Cu3O7", False],
            ["Y0.7Ca0.3Ba2Cu3O7", "Y1Ba2Cu3O7", True],
            ["Re2Si1", "Re0.9 Si0.1", False],
            ["Nb0.85Ge0.15", "Nb1.8Ge0.2", True],
            ["Nb0.7Ge0.3", "Nb0.4Ge0.6", False],
            ["Tl1Sr1La1Cu1O5", "Cu La O5 Sr Tl", True],
            ["La2Ca1Cu2O6", "Ca Cu2 La2 O6.037", True],
            ["Li1Ti2O4", "Li0.93 O4 Ti2", True],
            ["Ba0.67Pt3B2", "B2 Ba0.667 Pt3", True],
            ["B3Be1.09", "B3 Be1.0926", True],
            ["La1.85Sr0.15Cu0.9Zn0.1O4", "Cu0.87 La1.85 O3.91 Sr0.15 Zn0.13", True],
            ["La1.5Pb0.35Sr0.15Cu1O4", "Cu La1.58 O4 Pb0.27 Sr0.15", True],
            ["Rh1Ru2U1", "Rh0.99 Ru2.01 U", True],
            ["Re2Si1", "Re0.9 Si0.1", False],
            ["Y1Ba1.9K0.1Cu3O6.9", "Ba1.93 Cu3 K0.07 O7 Y", True],
            ["Tc0.9W0.1", "Tc0.85 W0.15", True],
            ["Y1Pt0.5Ge1.5", "Ge1.305 Pt0.67 Y", True],
            ["Ba8Si43Ge3", "Ba7.71 Ge3.07 Si42.7", True],
            ["Ba2Cu3Pr0.1Y0.9O7", "Ba1.98 Cu3 O7 Pr0.11 Y0.91", True],
            ["Y0.2Ca0.8Ba1.5Nd0.5Cu3O7", "Y1Ba2Cu3O7", False],
            ["Y0.8Ca0.2Ba1.9Nd0.1Cu3O7", "Y1Ba2Cu3O7", True],
            ["Ag3.3Al1", "Ag1.6Al0.4", True],
            ["Cr0.736Ru0.264", "Cr3Ru1", True],
            ["Ga25V75", "Ga1V3", True],
            ["Ga25V70Al5", "Ga1V3", False],
            ["Ga25V75Al15", "Ga1V3", False],
            ["Al0.3B2Mg0.7", "Al0.4B2Mg0.6", True],
            ["Cu0.99La1.8Ni0.01Sr0.2O4", "Cu1.98La3.7Ni0.02Sr0.3O8", True],
            ["Ba2Cu3.85Y1O6.85", "Ba2Cu3Y1O6", True],
            ["Ba3Ca1Cu6.58Fe0.42La2.5Y0.5O16.599", "Ba1.286Ca0.429Cu2.764Fe0.236La1.071Y0.214O7.24", True],
            ["Y1Ba1.4Ca0.6Cu3O7", "Y1Ba2Cu3O7", False],
            ["Y1Ba1.8Ca0.3Cu3O7", "Y1Ba2Cu3O7", True]
            ]
    
    
    df = pd.DataFrame(data, columns=["Supercon", "cif", "Should_match"])
    def test_similarity(formula_sc, formula_2):
        chemdict_sc = get_chem_dict(formula_sc)
        chemdict_2 = get_chem_dict(formula_2)
        try:
            formula_similarity, totreldiff = get_formula_similarity(chemdict_sc, chemdict_2)
        except AssertionError:
            formula_similarity = False
            totreldiff = np.nan
        output = pd.Series([formula_similarity, totreldiff])
        return(output)
    df[["Formula_similarity", "totreldiff"]] = df.apply(lambda row: test_similarity(row["Supercon"], row["cif"]), axis=1)
    
    
    
    
    
    
    
    
    
    
    
    
