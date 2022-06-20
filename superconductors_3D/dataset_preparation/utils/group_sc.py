#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:48:54 2021

@author: timo
This script takes data with a unique mapping from chemical formula to Tc and finds the classes of superconductors. Then it also changes the 3D structures of the files so that the cif files mirror the exact chemical composition of the superconducting entry.
"""






import pandas as pd
import numpy as np
from superconductors_3D.dataset_preparation.utils.check_dataset import get_chem_dict, ALL_ELEMENTS, find_doping_pairs
from copy import deepcopy
from superconductors_3D.machine_learning.own_libraries.own_functions import movecol
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
import chemml.chem.magpie_python as magpie
import itertools
from superconductors_3D.dataset_preparation.utils.calc_similarities import similarity_chem_formula
from superconductors_3D.dataset_preparation.utils.lib_generate_datasets import MAGPIE_features

LANTHANIDES = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
RARE_EARTH = LANTHANIDES + ["Y", "Sc"]
TRANSITION_METALS = ["Ti", "Zr", "Hf", "Rf", "V", "Nb", "Ta", "Db", "Cr", "Mo", "W", "Sg"]
ACTIONOIDES = ["Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
heavy_fermions = ["Ce", "Yt", "U", "Pu", "Pr", "Yb", "Np"]
assert all(el in ALL_ELEMENTS for el in RARE_EARTH)
assert all(el in ALL_ELEMENTS for el in TRANSITION_METALS)

def cluster_by_n_elements(formulas, min_n_el=-np.inf, max_n_el=np.inf):
    """Clusters `formulas` by the number of elements, where all n<=`min_n_el` are thrown together and all n>=`max_n_el` are thrown together.
    """
    n_elements = []
    for formula in formulas:
        chemdict = get_chem_dict(formula)
        n_el = len(chemdict)
        # Restrict n_el to range from min_n_el to max_n_el.
        n_el = np.clip(n_el, min_n_el, max_n_el)
        n_elements.append(n_el)
    return n_elements

def cluster_by_kmeans_with_MAGPIE(formulas, k):
    """Clusters formulas by kmeans clustering using MAGPIE as the features of the formulas.
    """
    magpie_features = MAGPIE_features(formulas)
    clustering = KMeans(n_clusters=k)
    clusters = clustering.fit_predict(magpie_features.to_numpy())
    return clusters

def assign_rough_class(formula, verbose=False):
    """Assigns a class to any superconductor, but these classes are not very sophisticated, only very rough."""
    sc_class = ""
    chemdict = get_chem_dict(formula)
    elements = list(chemdict.keys())
    num_els = len(elements)
    big_doped_els, small_doped_els, rest_els, _ = find_doping_pairs(chemdict, verbose=verbose)
    main_els = big_doped_els + rest_els
    eps = 1e-6
    if num_els >= 3 and "Cu" in elements and "O" in elements:
        sc_class += "Cuprate"
        good = num_els >= 3 and "Cu" in main_els and "O" in main_els
        strict = True
    
    chalco_pnictides = ["N", "P", "As", "Sb", "Bi", "S", "Se", "Te", "Po"]
    if anyin(["Fe", "Ni"], main_els) and anyin(chalco_pnictides, main_els):
        sc_class += "Ferrite"
        Fe_quant = chemdict["Fe"] if "Fe" in elements else chemdict["Ni"]
        other_quants = [chemdict[el] for el in elements if el in chalco_pnictides]
        good = anyin(["Fe", "Ni"], main_els) and anyin(chalco_pnictides, main_els) and Fe_quant > 0.7 and any([quant > 0.4 for quant in other_quants])
        strict = True
    
    if "O" in elements and not("Cuprate" in sc_class or "Ferrite" in sc_class):
        sc_class += "Oxide"
        good = np.nan
        strict = False
    
    if "C" in main_els and chemdict["C"] + eps >= 2*sum([val for el, val in chemdict.items() if not el == "C"]):
        sc_class += "Carbon"
        good = chemdict["C"] + eps >= 2
        strict = False
    
    if not "Cuprate" in sc_class and not "Ferrite" in sc_class and not "Carbon" in sc_class and anyin(heavy_fermions, elements) and any([quant > 0.31 for el, quant in chemdict.items() if el in heavy_fermions]):
        sc_class += "Heavy_fermion"
        good = np.nan
        strict = False
    
    Chevrel_formula = (
            any([rough_frac(chemdict, el1, "S", 6/8) for el1 in TRANSITION_METALS]) or 
            any([rough_frac(chemdict, el1, "Se", 6/8) for el1 in TRANSITION_METALS]) or
            any([rough_frac(chemdict, el1, "Te", 6/8) for el1 in TRANSITION_METALS])
            )
    if Chevrel_formula or ("Mo" in elements and anyin(["S", "Se", "Te"], elements)):
        sc_class += "Chevrel"
        good = Chevrel_formula
        strict = True
    
    # MgB2 = {"Mg": 1, "B": 2}    
    # if "Mg" in elements and "B" in elements:# and is_rough_formula(chemdict, MgB2, 0.7, 2):
    #     sc_class += "MgB2"
    #     spgs = ["P 6/m m m"]
    #     good = np.nan
    #     strict = False

    if sc_class == "":
        sc_class += "Other"
        good = np.nan
        strict = False
    
    output = pd.Series([sc_class, good, strict], ["Class1", "good", "strict"])
    return(output)
            

def anyin(list1, list2):
    result = any([el in list2 for el in list1])
    return(result)

def print_not_allowed_elements(elements, allowed_elements, sc_class, formula):
    not_allowed_elements = [el for el in elements if not el in allowed_elements]
    if not_allowed_elements:
        [print(f"Not allowed in {sc_class}: {', '.join(not_allowed_elements)} in {formula}")]
    return()
        
def is_rough_formula(chemdict, aim_chemdict, max_doping=0.4, max_doped_num=np.inf, normalise=True, allow_dropping=True):
    """Checks if formula from chemdict is roughly like aim_chemdict except for doping.
    
    chemdict: Elements as keys, quantities as values.
    aim_chemdict: Chemical system in format 'H-He-Li' for every allowed element at this position and with quantity as the value. Also accepts 'NaN' as element which means no element (vacancy).
    maxdoping: Maximum deviation of quantities from the aim value.
    max_doped_num: Maximum number of elements that are allowed in chemdict if they don't appear in aim_chemdict while having a quantity of less than maxdoping.
    normalise: If the quantities of the chemdict should be normalised so that it also matches an aim_chemdict with a multiple of quantities.
    allow_dropping: True means that not all of the elements in the aim_chemdict must be in chemdict if their quantity is less than max_doping.
    """
    elements = list(chemdict.keys())
    num_elements = len(elements)
    num_aim_elements = len(aim_chemdict.keys())
    
    # Check if chemical formulas can be similar at all by the elements and the number of elements.
    if allow_dropping:
        not_right_elements = not all([el in elements if quant > max_doping else True for el, quant in aim_chemdict.items()])
    else:
        not_right_elements = not all([el in elements for el, quant in aim_chemdict.items()])    
    
    wrong_num_els = num_elements > num_aim_elements + max_doped_num
    if not_right_elements or wrong_num_els:
        return(False)
    
    # Add doped elements to aim_chemdict.
    aim_chemdict = {el: aim_chemdict[el] if el in aim_chemdict.keys() else 0 for el in elements}
    quantities = np.array(list(chemdict.values()))
    aim_quantities = np.array(list(aim_chemdict.values()))
    
    # Normalise quantities of chemdict if they chose another unit cell.
    norm = np.sum(aim_quantities)/np.sum(quantities) if normalise else 1
    quantities *= norm
    
    # See if the difference of any quantities is too big.
    similar = all(np.abs(quantities - aim_quantities) < max_doping) 
    
    return(similar)



# Test is_rough_formula().
# placeholder_dict = {"Y-Sc-NaN": 1, "H-Ba-He": 2, "Cu-Fe-NaN": 3, "O": 7}
# formulas = {"Y1Ba2Cu3O7": True, "Sc0.9He0.1Ba2.2Cu3O6.3": True, "H2O6.5": True, "Ba1He2Cu3O7": False, "Y1Ba1.9Ca0.1Cu2.9Li0.1O7": False}
# # for formula, match in formulas.items():
# #     chemdict = get_chem_dict(formula)
# #     result = is_rough_formula(chemdict, placeholder_dict, 0.31, 1)
# #     print(f"{formula} should match: {match} and does match: {result}.")
# formulas = {"Ag1Te3": False, "In0.021Sn1Te1": True, "As0.005Sn0.97Te0.995": False}
# for formula, match in formulas.items():
#     chemdict = get_chem_dict(formula)
#     aim_chemdict = {"Sn": 1-0.3, "In": 0.3, "Te": 1}
#     result = is_rough_formula(chemdict, aim_chemdict, 0.31, 0, allow_dropping=True)
#     print(f"{formula} should match: {match} and does match: {result}.")

def calc_rough_frac(num1,  num2, frac, rel_dev):
    """Calculates if two numbers roughly have a given fraction."""
    normed = num1/num2*1/frac
    result = abs(normed- 1) < rel_dev
    return(result)
    
def rough_frac(chemdict, el1, el2, frac, rel_dev=0.33):
    """Calculates if two quantities from the given elements roughly have the given fraction."""
    if el1 in chemdict.keys() and el2 in chemdict.keys():
        num1 = chemdict[el1]
        num2 = chemdict[el2]
        result = calc_rough_frac(num1, num2, frac, rel_dev)
        
    else:
        result = False
    return(result)

def assign_sc_classes(row):
    """Assigns classes of superconductors roughly following Hirsch 2015."""
    formula = row["formula_sc"]
    norm_formula = row["norm_formula_sc"]
    structure = row["str3_sc"]
    spg = row["spacegroup_2"]
    crs = row["crystal_system_2"]
    comment = row["comment_sc"]
    tc = row["tc_sc"]
    max_doping = 0.4
    x = 0.3
    eps = 0.0001
    mark = False
    
    chemdict = get_chem_dict(formula)
    norm_chemdict = get_chem_dict(norm_formula)
    valsorted_chemdict = {el: val for el, val in sorted(chemdict.items(), key=lambda item: item[1])}
    elements = list(chemdict.keys())
    main_els, doped_els, rest_els, _ = find_doping_pairs(chemdict)
    num_els = len(elements)
    num_not_doped_els = len(main_els) + len(rest_els)
    quantities = np.array(list(chemdict.values()))
    valsorted_elements = list(valsorted_chemdict.keys())
    sc_class = ""
    
    
    MgB2 = {"Mg": 1, "B": 2}    
    if "C" in elements and chemdict["C"] >= 60:
        sc_class += "Fullerene"
        spgs = ["F m -3 m"]
        good = spg in spgs
        strict = False
        
    elif "Mg" in elements and "B" in elements and is_rough_formula(chemdict, MgB2, 0.7, 2):
        sc_class += "MgB2"
        spgs = ["P 6/m m m"]
        good = spg in spgs
        strict = True
    
    elif anyin(["Fe", "Ni"], elements) and anyin(["N", "P", "As", "Sb", "Bi", "S", "Se", "Te", "Po"], elements):
        sc_class += "Ferrite"
        good = tc == 0 or crs == "tetragonal"
        strict = True
    
    elif "Cu" in elements and "O" in elements:
        sc_class += "Cuprate"
        spgs = ["I 4/m m m", "P 4/m m m", "P m m m"]
        good = spg in spgs    
        strict = False
    
    elif "C" in elements and chemdict["C"] >= 2 and chemdict["C"] < 60 and chemdict["C"] + eps >= 2*sum([val for el, val in chemdict.items() if not el == "C"]):
        sc_class += "Itc Graphite"
        good = np.nan
        strict = False
    
    elif "B" in elements and anyin(["C", "N"], elements):
        sc_class += "Borocarbide"
        good = spg == "I 4/m m m"
        strict = True
    
    elif "Bi" in elements and "S" in elements:
        sc_class += "Bi-S"
        good = spg in ["I 4/m m m", "I -4 2 m", "P 4/n m m"]
        strict = True
        
    elif structure == "BKBO" or ((rough_frac(chemdict, "O", "Bi", 3) or rough_frac(chemdict, "O", "Sb", 3))):
        sc_class += "Bismuthate"
        good = np.nan
        strict = False
    
    elif anyin(heavy_fermions, elements) and any([quant > 0.31 for el, quant in chemdict.items() if el in heavy_fermions]):
        sc_class += "Heavy_fermion"
        good = np.nan
        strict = False
    
     
    if structure == "Cr3Si(A15)" or (spg == "P m -3 n" and len(elements) in [2,3]):
        max_el = valsorted_elements[-1]
        max_val = chemdict[max_el]
        min_valsum = sum([val for el, val in chemdict.items() if el != max_el])
        rough_fraction = calc_rough_frac(max_val, min_valsum, 3, 0.9)
        min_quant = min(chemdict.values())
        doped_formula = min_quant < 0.5
        if len(elements) == 2 or (len(elements) == 3 and rough_fraction and doped_formula):
            sc_class += "A15"
            min_el_val = min(norm_chemdict.values())
            good = structure == "Cr3Si(A15)" or len(elements) == 2 or calc_rough_frac(max_val, min_valsum, 3, 0.3)
            strict = False
        
    elif num_els <= 2 or (num_els == 3 and min(quantities) < 1):
        sc_class += "Simple"
        good = np.nan
        strict = False
        
    # semiconductors = ["SnTe", "GeTe", "PbTe", "InTe", "SrTiO3", "Ge", "Si", "SiC4", "SiCH6", "C"]
    # semi_chemdicts = [get_chem_dict(semicond) for semicond in semiconductors]
    # doped_semicon = any([min(quantities) < max_doping and len(elements) >= len(semi_chemdict)  and is_rough_formula(chemdict, semi_chemdict, max_doped_num=1, allow_dropping=False) for semi_chemdict in semi_chemdicts])
    # commt_semicon = "semic" in comment or "carrier" in comment
    # if doped_semicon or commt_semicon:
    #     sc_class += "Semicon"
    #     good = np.nan
    #     strict = False
    
    if "N" in elements and any(
            [rough_frac(chemdict, "N", el1, 1) and rough_frac(chemdict, "N", el2, 1) for el1, el2 in itertools.product(["Ti", "Zr", "Hf"], ["Cl", "Br", "I"])]
            ):
        sc_class += "Layered_N"
        good = np.nan
        strict = False
        
    if structure == "Chevrel" or (
            any([rough_frac(chemdict, el1, "S", 6/8) for el1 in TRANSITION_METALS]) or 
            any([rough_frac(chemdict, el1, "Se", 6/8) for el1 in TRANSITION_METALS]) or
            any([rough_frac(chemdict, el1, "Te", 6/8) for el1 in TRANSITION_METALS])
            ):
        sc_class += "Chevrel"
        good = structure == "Chevrel" or "Mo" in elements
        strict = True
    
    if (is_rough_formula(chemdict, {"Cu": x, "Bi": 2, "Se": 3}, 0.4, 0, allow_dropping=False) or
       is_rough_formula(chemdict, {"Sn": 1-x, "In": x, "Te": 1}, 0.4, 0, allow_dropping=False) or
       is_rough_formula(chemdict, {"Pb": 0.5-x/2, "Sn": 0.5-x/2, "In": x, "Te": 1}, 0.4, 0, allow_dropping=False) or
       is_rough_formula(chemdict, {"Cu": x, "Pb": 5, "Se": 23, "Bi": 12}, 0.4, 0, allow_dropping=False)):
        sc_class += "Topological"
        good = np.nan
        strict = False
    
    if "Sr" in elements and "Ru" in elements and "O" in elements:
        sc_class += "Sr-Ru-O"
        spgs = ["I 4/m m m"]
        Sr_ruthenate = {"Sr": 2, "Ru": 1, "O": 4}
        correct_formula = is_rough_formula(chemdict, Sr_ruthenate, max_doping, allow_dropping=False)
        good = spg in spgs and correct_formula
        strict = True
    
    if not anyin(RARE_EARTH + ACTIONOIDES, elements) and num_not_doped_els <= 3:
        sc_class = "Compound"
        good = np.nan
        strict = False
    
    if sc_class == "":
        sc_class = np.nan
        good = np.nan
        strict = True
    
    output = pd.Series([sc_class, good, strict, mark], ["Class", "good", "strict", "mark"])
    return(output)
        
    
    


if __name__ == "__main__":
    
    # This needs to be changed to get the output of the cif manipulating file instead of the output of the selecting file.
    #input_file = "/home/timo/Dokumente/Masterarbeit/Rechnungen/Datasets/Sonstiges/Matched_Supercon_COD_MP.csv"

    important_cols = ['origin_2', 'similarity', 'formula_sc', 'formula_2', 'tc_sc',  'spacegroup_sc', 'spacegroup_2', 'guess_spg_sc', 'crystal_system_2', 'crystal_system_sc', 'form_reldiff', 'lat_reldiff', 'comment_sc', 'comment_2', 'str3_sc', 'strcmt_sc', 'lat_similar', 'lata_2', 'lata_sc', 'latb_2', 'latb_sc', 'latc_2', 'latc_sc',  'name_sc',  'cif_2', 'chemical_composition_sc', 'chemical_composition_2', 'material_id_2', 'file_id_2', 'num_sc', 'original_formula_sc',  'original_formula_2', 'norm_formula_sc', 'tc_old_sc', 'tcn_sc', 'origin_sc', 'MP_theoretical', 'num_elements', 'year_sc']
    
    df = pd.read_csv(input_file, header=1)
    df = df[important_cols]
    
    
    # Assign sc classes.
    df[["Class", "good", "strict", "mark"]] = df.apply(assign_sc_classes, axis=1)
    df = movecol(df, ["Class", "good", "strict", "mark", "comment_sc", "str3_sc", "strcmt_sc"], "tc_sc", "After")
    sc_class = "Compound"
    x = df[df["Class"].str.contains(sc_class, regex=False, na=False)]
    num_good_points = sum(x["good"])
    print(f"Number of good points for {sc_class}: {num_good_points}")
    
    # TODO: Exclude if good=False and strict=True except if tc=0.
    
    
    
    
    
    