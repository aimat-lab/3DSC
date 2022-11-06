#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:05:00 2020

@author: timo
This is a helper script that includes a lot of functions particulary useful for cleaning datasets.
"""
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import hashlib
from copy import deepcopy
import gemmi
import warnings
from collections import Counter
from pymatgen.core.composition import CompositionError, Composition
from superconductors_3D.machine_learning.own_libraries.own_functions import isfloat



O_PATTERN = "O[\-\+\.\=A-Za-rt-z0-9]*$"

ALL_ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]


    
def chem_dict_from_pymatgen(string):
    '''Get dictionary with chemical composition from pymatgen. Ignore weird behaviour of pymatgen when dealing with non existing elements.'''
    chem_dict = dict(Composition(string, strict=True).as_dict())
    # Fix weird behaviour of pymatgen
    for el in list(chem_dict.keys()):
        if "0+" in el:
            val = chem_dict[el]
            del chem_dict[el]
            el = el.replace("0+", "")
            chem_dict[el] = val
    return(chem_dict)

def get_chem_dict(string, weird_oxygen=False):
    """Get a sorted dictionary of the elements in the string and the corresponding values.
    
    Can deal with unspecified oxygen quantitites if they follow the pattern O_PATTERN. If oxygen is not specified at all (e.g. Oz) it's value is np.nan, if it's value is something like O7+z it's new value is 7. The sorting is alphabetically for the keys, except that oxygen is always last.
    """
    
    # Stupid exception in case someone used Y as unknown value for oxygen.
    exception_pattern = "OY$"
    if re.search(exception_pattern, string) != None:
        O_value = "Y"
        string = re.sub(exception_pattern, "", string)
    
    try:
        chem_dict = chem_dict_from_pymatgen(string)
    except (CompositionError, ValueError) as error: 
        # Search for weird oxygen strings in the Supercon.
        if weird_oxygen == True and re.search(O_PATTERN, string) != None:
            split_idx = re.search(O_PATTERN, string).start()
            O_string = string[split_idx:]
            O_value = O_string.split("O")[-1]
            if O_value == "nan":
                O_value = np.nan
            string = string[:split_idx]
        elif "Onan" in string:
            # If value of oxygen is np.nan.
            O_value = np.nan
            string = string.replace("Onan", "")
        try:
            chem_dict = chem_dict_from_pymatgen(string)
        except (CompositionError, ValueError) as error:
            # print("Bad formula, returned empty chemdict: {}".format(string))
            chem_dict = {}
            return(chem_dict)
    
    # Get oxygen value if no weird string is present.
    try:
        O_value
    except NameError:
        if "O" in chem_dict.keys():
            O_value = chem_dict["O"]
            del chem_dict["O"]
        else:
            O_value = None

    assert all([isfloat(chem_dict[key]) for key in chem_dict.keys()]), "In the chem_dict we found an element with not float value: {}".format(chem_dict)
    
    # Sort alphabetically, except oxygen comes last.
    sorted_elements = sorted(chem_dict.keys())
    sorted_chem_dict = {el: chem_dict[el] for el in sorted_elements}
    if O_value != None:
        sorted_chem_dict["O"] = O_value
    return(sorted_chem_dict)

# Test get_chem_dict
# l = ["SmFeAsO0.8H0.2", "CuBa0.2Oz", "CuIr0.2Onan", "Bi1.6Pb0.3Sb0.1Sr2Ca2Cu3OZ", "Bi1Sr1Cu1OY", "Eu1.45Pr0.05Ce0.5Sr2Cu2Nb1O10=z"]
# for string in l:
#     chem_dict = get_chem_dict(string, weird_oxygen=True)
#     print(string, chem_dict)


def get_x_dict(formula_string):
    """Get a dictionary with the chemical composition, and also work with values like 'x' or '2-x'."""
    elements = {}
    while formula_string.find("x") != -1:
        idx = formula_string.find("x")
        for el_idx in range(0, idx+1)[::-1]:
            one_char = formula_string[el_idx:el_idx+1]
            two_chars = formula_string[el_idx:el_idx+2]
            if two_chars in ALL_ELEMENTS:
                el = two_chars
                value = formula_string[el_idx+2:idx+1]
                break
            elif one_char in ALL_ELEMENTS:
                el = one_char
                value = formula_string[el_idx+1:idx+1]
                break
            else:
                continue
        elements[el] = value
        formula_string = formula_string.replace(el+value, "", 1)
    
    rest_elements = chem_dict_from_pymatgen(formula_string)
    for el in rest_elements.keys():
        value = rest_elements[el]
        elements[el] = value
    return(elements)


def normalise_chemdict(to_normalise, by):
    """Normalises one chemical dictionary by something, either the sum of values of a given dictionary or a float.."""
    quantities_to_normalise = np.array(list(to_normalise.values()))
    if isinstance(by, dict):
        by_quantities = np.array(list(by.values()))
        norm = sum(by_quantities)/sum(quantities_to_normalise)
    elif isfloat(by):
        norm = by
    to_normalise = {el: val*norm for el, val in to_normalise.items()}
    return(to_normalise)


def filter_entries(excl_condition, reason, df_correct, df_excluded=pd.DataFrame(), verbose=True, reason_col_name="Reason for exclusion"):
        '''Gets a condition (negative, i.e. if the condition is True than the entry gets filtered out) in form of a boolean array and returns matched and not matched data in the corresponding dataframes. The reason_string explains why this entry was excluded.'''
        df_ex = df_correct[excl_condition]
        df_ex[reason_col_name] = reason
        df_excluded = df_excluded.append(df_ex)
        df_correct = df_correct[~ excl_condition]
        if verbose == True:
            if type(reason) == str:
                all_reasons = reason
            elif type(reason) == pd.core.series.Series:
                all_reasons = ", ".join(reason.unique().tolist())
            else:
                all_reasons = "Warning: Reason not recognisable"
            print("Excluded {} from {} entries because: {}".format(len(df_ex), len(df_ex) + len(df_correct), all_reasons))
        return(df_correct, df_excluded)
    
def exclude_row(df_excluded, row, col_value, col_name="Reason for exclusion"):
    """Add one row to a dataframe and also add one column with value. Used for appending rows to exclude to the dataframe containing these rows.
    """
    row[col_name] = col_value
    df_excluded = df_excluded.append(row)
    return(df_excluded)


def get_chemical_composition(formula):
    '''This function takes a chemical formula and returns the chemical composition as string (only the elements sorted, not the quantities)'''
    
    # Get sorted dictionary of chemical elements.
    chemdict = get_chem_dict(formula)
    
    all_elements = list(chemdict.keys())
    all_elements_string = "-".join(all_elements)
    return(all_elements_string)


def normalise_quantities(chemdict, total=100):
    """Normalises the quantities in the chemical formula dictionary so that the new values are in percent of the sum of quantities. Can deal with np.nan as quantity of an element.
    """
    if "O" in chemdict.keys():
        # Oxygen value may either be number or np.nan
        assert isfloat(chemdict["O"]), chemdict
    quantities = np.array(list(chemdict.values()))
    total_quantities = np.nansum(quantities)
    norm_factor = total/total_quantities
    chemdict = {el: norm_factor*chemdict[el] for el in chemdict}
    return(chemdict)
        
def chemdict_to_formula(chem_dict):
    """Generates the chemical formula from a chemical dictionary."""
    chem_dict = deepcopy(chem_dict)
    if "O" in chem_dict.keys():
        O_value = chem_dict["O"]
        del chem_dict["O"]
    else:
        O_value = ""
    
    chem_list = [el + str(round(chem_dict[el], 3)) for el in chem_dict.keys()]
    if O_value != "":
        chem_list.append("O{}".format(O_value))
    chem_list = [re.sub("\.0$", "", el) for el in chem_list]
    sorted_string = "".join(chem_list)
    return(sorted_string)


def standardise_chem_formula(string, normalise=False):
    '''Does some normalising and sorting, so that chemical formulas have the same string when they are chemically identical.'''
    string = string.replace(" ", "")
    chem_dict = get_chem_dict(string)
    if normalise == True:
        chem_dict = normalise_quantities(chem_dict)
            
    if "O" in chem_dict.keys():
        O_value = chem_dict["O"]
        del chem_dict["O"]
    else:
        O_value = ""
    
    chem_list = [el + str(round(chem_dict[el], 3)) for el in chem_dict.keys()]
    if O_value != "":
        chem_list.append("O{}".format(O_value))
    chem_list = [re.sub("\.0$", "", el) for el in chem_list]
    sorted_string = "".join(chem_list)
    return(sorted_string)

# formulas = ["Y0.975Ca0.025Ba2Cu2.975Al0.025O6.89"]
# for formula in formulas:
#     print(standardise_chem_formula(formula))


def if_valid_formula(string):
    '''Tests if all chemical elements exist and have float values > 0.'''
    string0 = string
    if not isinstance(string, str):
        # print("Excluded: ", string0)
        return(False)
    if string == "":
        # print("Excluded: ", string0)
        return(False)
    string = string.replace(" ", "")
    chem_dict = get_chem_dict(string)
    if chem_dict == {}:
        # print("Excluded: ", string0)
        return(False)
    for el in chem_dict.keys():
        value = chem_dict[el]
        # Exclude every non-standard element like e.g. Deuterium
        if el not in ALL_ELEMENTS:
            # print("Excluded: ", string0)
            return(False)
        if not isfloat(value):
            # print("Excluded: ", string0)
            return(False)
        if value <= 0:
            # print("Excluded: ", string0)
            return(False)
    return(True)


def set_column_type(col):
    '''Sets column dtype to the dtype of the rest of the values (ignoring nans).'''
    unique_dtypes = col.dropna().map(type).unique()
    if str in unique_dtypes or len(unique_dtypes)==0:
        newcol = col.fillna("").astype("object")
    elif unique_dtypes == [float]:
        newcol = col.astype(float)
    elif unique_dtypes == [int]:
        newcol = col.astype(int)
    elif unique_dtypes == [bool]:
        newcol = col.astype(bool)
    else:
        raise Exception("Column type is being a weirdo.")
    # print(newcol.name, newcol.dtype)
    return(newcol)


def extract_float_values(val, correct_pattern):
    '''Extracts float value from a string with uncertainty as value in brackets at the end or other ocurring strings.'''
    if val == "":
        val = np.nan
    elif type(val) == str and correct_pattern == True:
        # Strip strings that indicate uncertainy or similar and only leave float value.
        val0 = val
        pattern_front = "\[?\'?"
        pattern_back = "\s?\(([0-9]\.?[0-9]*)*\).*\]?\'?$"
        pattern_temperature = "([0-9]\.?[0-9]*)+\s?K"
        val = re.sub(pattern_front, "", val)
        val = re.sub(pattern_back, "", val)
        val = re.sub(pattern_temperature, "", val)
        if re.search("[Rr]oom\s?[Tt]emp", val):
            val = 295
        try:
            val = float(val)
        except ValueError:
            print("Weird value found and nan assigned: {}".format(val))
            val = np.nan
    # All of the numbers that this function is applied to are bound to be positive.
    if val < 0:
        val = np.nan
    assert isfloat(val)
    return(val)

def normalise_AFLOW_prototype_spacegroups(spg_string):
    """Tries to recognise spacegroup in the format of the AFLOW prototype database."""
    
    if pd.isna(spg_string):
        return(spg_string)
    
    # Replace weird minus indicators by better ones.
    minus_pattern = re.compile("(\d)\¯")
    matches = minus_pattern.findall(spg_string)
    for number in matches:
        to_replace = f"{number}¯"
        spg_string = spg_string.replace(f"{number}¯", f"-{number}")
    
    # Return normalised spacegroup.
    spg = gemmi.find_spacegroup_by_name(spg_string)
    if spg != None:
        spg_string = get_normalised_spg(spg).hm
        return(spg_string)
    else:
        raise Warning()


def normalise_pymatgen_spacegroups(spg_string):
    '''Tries to recognise spacegroup and to return the Hermann Maguin form. If it doesn't recognise the spacegroup it sets it "".'''
    if pd.isna(spg_string):
        return(np.nan)
    
    spg = gemmi.find_spacegroup_by_name(spg_string)
    if spg != None:
        spg_string = get_normalised_spg(spg).hm
        return(spg_string)

    if spg_string == "":
        return(spg_string)
    
    pattern = "_[0-9]"
    if re.search(pattern, spg_string) != None:
        spg_string = re.sub("_", "", spg_string)
    
    spg = gemmi.find_spacegroup_by_name(spg_string)
    if spg == None:
        print("Unknown spacegroup found: {}".format(spg_string))
        spg_string = ""
        return(spg_string)
    else:
        spg_string = get_normalised_spg(spg).hm
        return(spg_string)
        

def get_normalised_spg(spg):
    '''Normalises the spacegroup because apparently there are several ones with the same IT number, I see the number as the important one.'''
    number = spg.number
    spacegroup = gemmi.find_spacegroup_by_number(number)
    # if spg != spacegroup:
        # print("Normalised spacegroup: H-m: {} vs {}; Number: {} vs {}".format(spg, spacegroup, spg.number, spacegroup.number))
    return(spacegroup)


def check_and_complement_structure(df, sp_name, crys_name, num_name):
    '''Use spacegroup, spacegroup number and crystal system. Check if they are consistent, make an empty row if they are not consistent and imply one from the other if possible.'''
    df[[sp_name, crys_name, num_name]] = df.apply(
                                                consistent_spacegroup_and_crystal_system, 
                                                axis="columns", 
                                                args=(sp_name, crys_name, num_name)
                                                )

    # Final consistency check of the structure, because the function before is a bit complex.
    print("Check if the structure features really are consistent.")
    # Assert that spacegroup and crystal system is consistent.
    df.apply(assert_structure_consistent, 
                     axis="columns",
                     args=(sp_name, crys_name, num_name))
    print("Spacegroups and crystal systems are consistent!")
    return(df)


def consistent_spacegroup_and_crystal_system(row, spg_name, crg_name, num_name):
    '''Check spacegroup, IT number and crystal system for consistency. If something is inconsistent, return empty for everything. Else imply crystalgroup from spacegroup (and spacegroup/ IT_number from the other one) or if everything is consistent, return as given.'''
    spacegroup = row[spg_name]
    if pd.isna(spacegroup):
        spacegroup == ""
    crystalgroup = row[crg_name]
    
    IT_number = row[num_name]
    # Should not happen but just in case.
    if IT_number == 0:
        IT_number = np.nan
    # The function of gemmi needs an int, not float.
    elif not pd.isna(IT_number):
        IT_number = int(IT_number)
        assert IT_number <= 230, row
    
    # Either complement spacegroup/ IT_number if other is given or return if none is given.
    if spacegroup == "" and not pd.isna(IT_number):
        spacegroup = gemmi.find_spacegroup_by_number(IT_number).hm
        
    elif spacegroup != "" and pd.isna(IT_number):
        IT_number = gemmi.find_spacegroup_by_name(spacegroup).number
        
    elif spacegroup == "" and pd.isna(IT_number):
        output = pd.Series(["", crystalgroup, np.nan], [spg_name, crg_name, num_name])  
        return(output)
    
    # Here both spacegroups existed from the beginning on.
    elif spacegroup != "" and not pd.isna(IT_number):
        spg_from_number = gemmi.find_spacegroup_by_number(IT_number).hm
        
        if spacegroup == spg_from_number:
            crg_from_spg = gemmi.find_spacegroup_by_name(spacegroup).crystal_system_str()
            
            # All consistent, return.
            if crystalgroup == crg_from_spg:
                output = pd.Series([spacegroup, crystalgroup, IT_number], [spg_name, crg_name, num_name])
                assert_structure_consistent(output, spg_name, crg_name, num_name)
                return(output)
            
            # Crystalgroup not given, imply from spacegroup.
            elif crystalgroup == "":
                crystalgroup = crg_from_spg
                output = pd.Series([spacegroup, crystalgroup, IT_number], [spg_name, crg_name, num_name])
                assert_structure_consistent(output, spg_name, crg_name, num_name)
                return(output)
            
            # Crystalgroup and spacegroup not consistent, return empty.
            elif crystalgroup != crg_from_spg:
                output = pd.Series(["", "", np.nan], [spg_name, crg_name, num_name])
                return(output)
        
        # Spacegroups not consistent, return empty.
        elif spacegroup != spg_from_number:
            output = pd.Series(["", "", np.nan], [spg_name, crg_name, num_name])  
            return(output)
        
    # If from the beginning on there was only one of spacegroup or IT_number given, but now the other one was complemented.
    crg_from_spg = gemmi.find_spacegroup_by_name(spacegroup).crystal_system_str()
    # Imply missing crystalgroup from spacegroup.
    if crystalgroup == "":
        crystalgroup = crg_from_spg
        output = pd.Series([spacegroup, crystalgroup, IT_number], [spg_name, crg_name, num_name])  
        assert_structure_consistent(output, spg_name, crg_name, num_name)
        return(output)
    
    # Crystalgroup is given.
    elif crystalgroup != "":
        
        # Crystalgroup consistent with spacegroup, return.
        if crystalgroup == crg_from_spg:
            output = pd.Series([spacegroup, crystalgroup, IT_number], [spg_name, crg_name, num_name])
            assert_structure_consistent(output, spg_name, crg_name, num_name)
            return(output)
        
        # Crystalgroup not consistent with spacegroup,return empty.
        elif crystalgroup != crg_from_spg:
            output = pd.Series(["", "", np.nan], [spg_name, crg_name, num_name])  
            return(output)
        
    # If all is good you should not end up here.
    raise Warning("Why am I here???") 


def assert_structure_consistent(row, spg_name, crg_name, num_name):
    '''Check if spacegroup, IT_number and crystalgroup are consistent. If this check is passed, each row is consistent, and either the whole row could be missing or the crystalgroup could be given but the spacegroup (and IT_number) is empty.'''
    spacegroup = row[spg_name]
    crystalgroup = row[crg_name]
    IT_number = row[num_name]
    if spacegroup != "":
        assert crystalgroup != "", row
        assert not pd.isna(IT_number)
        spg1 = gemmi.find_spacegroup_by_name(spacegroup) 
        spg2 = gemmi.find_spacegroup_by_number(int(IT_number))
        assert spg1 == spg2, row
        assert spg1.crystal_system_str() == crystalgroup, row
    else:
        assert pd.isna(IT_number), row
        


def correct_symmetry_cell_setting(col):
    '''Replaces a lot of wrong written crystal symmetries with the right ones. If mistakenly the spacegroup was written here, implies the crystal system from the spacegroup. Also checks that in the end only the 7 valid crystal systems remain.'''
    
    valid_values = ["", "monoclinic", "orthorhombic", "triclinic", "tetragonal", "hexagonal", "cubic", "trigonal"]
    
    # Try to get crystal system if mistakenly spacegroup was recorded.
    def get_crystal_system(string):
        sp = gemmi.find_spacegroup_by_name(string)
        if sp != None:
            crystal_system = sp.crystal_system_str()
            # print("Recognised crystal system from spacegroup: {} --> {}".format(sp.hm, crystal_system))
            return(crystal_system)
        else:
            return(string)
    newcol = col.apply(get_crystal_system)
    
    replace_symmetry_cell_setting = {
        np.nan: "",
        'rhombohedral': "trigonal",
        "Monoclinic": "monoclinic",
        '?': "",
        'Orthorhomic': "orthorhombic",
        "Monoclinic'": "monoclinic",
        'P-1': "",
        'orthogonal': "", 
        "Orthorhombic'": "orthorhombic", 
        'Triclinic': "triclinic",
        'i': "", 
        'Pnma': "",
        '11': "", 
        'trigonal (hexagonal)': "trigonal",
        "Triclinic'": "triclinic",
        '??monoclinic??': "",
        'rhombohedral (on hexagonal axes)': 'trigonal',
        'monoclinic, No. 14': "monoclinic",
        'monoclinic (b axis)': "monoclinic",
        'rhombohedral (hexagonal setting)': 'trigonal',
        'Rhombohedral (hexagonal axes)': 'trigonal',
        'Triclinic twin': "", 
        'P21/n': "",
        'P21/m': "",
        'P21/c': "",
        '2': "",
        'monoclin': "monoclinic",
        'triclin': "triclinic",
        '1': "",
        'tatragonal': "tetragonal", 
        'Rhombic': "",
        'monoclilic': "monoclinic", 
        'momoclinic': "monoclinic", 
        'mooclinic': "monoclinic",
        'triclininc': "triclinic",
        'Triclini': "triclinic",
        'Pna2(1)': "", 
        'monoclinic?': "",
        'P2(1)/N': "",
        'standard': "",
        'Tetrahedral': "",
        'triclinic, twinned via 2[100]': "",
        'triclinic (monoclinic cell)': "",
        'P -1': "",
        'P 2(1)2(1)2(1) ': "",
        '-R 3 2': "",
        'monoclinic, twinned via 2(100)': "",
        'rhombohedral (hexagonal axes)': "trigonal",
        'p 2ac 2ab': "",
         'P m -3 n': "",
         'F d -3 m': "",
         '-P1': "",
         'anorthic': "",
         '146h': "",
        '14': "",
         'triclinic?': "",
         "Tetragonal'": "tetragonal",
         'P212121': "",
         '&#x00A1;&#x00AF;monoclinic&#x00A1;&#x00AF;': "",
         'Trigonal': 'trigonal',
         'Monoclinic': "monoclinic",
         'tericlinic': 'triclinic',
         'TETRAGONAL': 'tetragonal',
         'Orthorhombic': "orthorhombic",
         'c-centered monoclinic': "monoclinic",
         'tetrahedral': "",
         'Hexagonal': "hexagonal",
         '-1': "",
         'general': "",
         'primitive': "",
         'Pca21': "",
         'rhombohedral, hexagonal': "",
         'rhombohedral on R axes': 'trigonal',
         'trigonal (hexagonal setting)': "",
         'monoclinic second setting': "monoclinic",
         'Orthorgonal': "",
         'Teternal': "",
         'orthorhombic,': "orthorhombic",
         'P21/a': "",
         'C-centered monoclinic': "monoclinic",
         '-P 1': "",
         '-P2yn': "",
         'first setting, choice 1': "", 
        'Teragonal': "tetragonal", 
        'Moniclinic': "monoclinic",
        'orthorhombic?': "", 
        'Orthorombic': "orthorhombic", 
        'Orthohombic': "orthorhombic", 
        'R': "", 
        'P1': "",
        'Trigonal (hexagonal)': "", 
        ' Rhombohedral on Hexagonal axes ': "trigonal", 
        'P2/k': "",
        '? tetragonal': "", 
        'Triclinic, twinned via 2[001]*': "",
        '&#x00A1;&#x00AE;orthorhombic&#x00A1;&#x00AF;': "",
        'Least-squares, 28 randomly selected refls': "", 
        'Monoclinic?': "",
        'P2(1)/c': "", 
        'P-3c1': "", 
        'Cmcm': "", 
        'Monoclinic twin': "",
        'Monoclinic, twinned via 2[001]': "", 
        'Orthogonal': "", 
        'Trigonal?': "",
        'Triclinic, twinned via 2[100]': "", 
        "orthorhombic'": "orthorhombic", 
        "triclinic'": "triclinic",
        'monoclnic': "monoclinic", 
        'orthorhomic': "orthorhombic", 
        'Mnonclinic': "monoclinic", 
        'multi-scan': "",
        'Triiclinic': "triclinic", 
        'Monoclinic, twinned': "", 
        'triclinil': "triclinic",
        'rhombohedral - hexagonal axes': "trigonal", 
        'Monclinic': "monoclinic", 
        '? monoclinic': "",
        'Triclic': "triclinic", 
        'tricrinic': "triclinic", 
        "Cubic'": "cubic", 
        'tetagonal': "tetragonal", 
        'Tticlinic': "triclinic"
        }
    newcol = newcol.replace(to_replace=replace_symmetry_cell_setting)
    assert all(newcol.isin(valid_values))
    return(newcol)

def entries_to_check():
    """Returns a dataframe with entries Supercon entries and cod entries and if they should match. This comes from what I thought once and was revised because some superconducting entries were excluded from Clean_Supercon.py.
    """
    test_sc_entries = ["Pb2Sr2Nd1Ce1Cu3O10+Y",
 "Tl1Sr1La1Cu1O5",
 "Hg0.7V0.3Ba1Sr1Cu1O4.6+z",
 "La2Ca1Cu2O6",
 "Li1Ti2O4",
 "Ba0.67Pt3B2",
 "B3Be1.09",
 "La1.85Sr0.15Cu0.9Zn0.1O4",
 "La1.5Pb0.35Sr0.15Cu1Oz",
 "Rh1Ru2U1",
 "Re2Si1",
 "Y1Re0.1Ba2Cu2.9O7.08",
 "Nd1Ba2Cu3O6.93",
 "Nd1.05Ba1.95Cu3Oz",
 "Nd1.05Ba1.95Cu3O6.68",
 "Nd1.05Ba1.95Cu3O7+z",
 "Y1Ba1.9K0.1Cu3O6.9",
 "Y1Ba1.8K0.2Cu3O6.57",
 "Y1Ba1.95Na0.05Cu3Oz",
 "Tc0.9W0.1",
 "Y1Pt0.5Ge1.5",
 "Hg1Ba2Ca2Cu3O8.3",
 "La1.85Sr0.12Ca0.016Ba0.014Cu1O4",
 "La1.85Sr0.12Ca0.016Ba0.014Cu1O3.99",
 "La1.7Nd0.15Ca0.056Ba0.094Cu1O4.02",
 "Ba8Si43Ge3",
 "Li0.84H1Fe1.14Se1O1"]
    test_cod_entries = ["Ce Cu3 Nd O10.2 Pb2 Sr2",
 "Cu La O5 Sr Tl",
 "Ba Cu Hg0.67 O4.66 Sr V0.33",
 "Ca Cu2 La2 O6.037",
 "Li0.93 O4 Ti2",
 "B2 Ba0.667 Pt3",
 "B3 Be1.0926",
 "Cu0.87 La1.85 O3.91 Sr0.15 Zn0.13",
 "Cu La1.58 O4 Pb0.27 Sr0.15",
 "Rh0.99 Ru2.01 U",
 "Re0.9 Si0.1",
 "Ba2 Cu2.9 O7.049 Re0.1 Y",
 "Ba2 Cu3 Nd O6.92",
 "Ba1.95 Cu3 Nd1.05 O6.95",
 "Ba1.95 Cu3 Nd1.05 O6.95",
 "Ba1.95 Cu3 Nd1.05 O6.95",
 "Ba1.93 Cu3 K0.07 O7 Y",
 "Ba1.82 Cu3 K0.18 O7 Y",
 "Ba1.95 Cu3 Na0.05 O7 Y",
 "Tc0.85 W0.15",
 "Ge1.305 Pt0.67 Y",
 "Ba2 Ca2 Cu3 Hg O8.35",
 "Ba0.014 Ca0.016 Cu La1.85 O4 Sr0.12",
 "Ba0.014 Ca0.016 Cu La1.85 O4 Sr0.12",
 "Ba0.094 Ca0.056 Cu La1.7 Nd0.15 O4",
 "Ba7.71 Ge3.07 Si42.7",
 "Fe1.14 H Li0.83 O Se"]
    test_if_match = [False, True, False, False, True, True, True, 
                     False, False, True, False, True, True, False, False, False, True, True, False, 
                     True, True, True, False , False, False, True, True]
    
    df_test =  pd.DataFrame(data=[test_sc_entries, test_cod_entries, test_if_match], index=["original_formula_sc", "original_formula_2", "if_match_wanted"]).T
    
    return(df_test)

def find_doping_pairs(chemdict, verbose=True):
    """Finds which element was doped by which other element and returns two lists of main element, doping element, rest of elements.
    """
    elements = np.array(list(chemdict.keys()))
    quantities = np.array(list(chemdict.values()))
    
    # Exclude elements that are not doped.
    doped = quantities != np.round(quantities)
    quantities = quantities[doped]
    rest_els = list(elements[~doped])
    elements = list(elements[doped])
    
    # Find doping pairs where the two added give a round number.
    pairsum = quantities[:,None] + quantities
    doping_pair = pairsum == np.round(pairsum)
    for i in range(len(doping_pair)):
        # In case two matches both have 0.5 the trace must be False so that they are not confused with not unique matches just because they match themselves.
        doping_pair[i,i] = False
    
    # If there are not unique matches notify and put them in rest_els.
    multiple_match_idx = []
    multiple_match_els = []
    doubled_elements = False
    for i, el1 in enumerate(elements):
        if sum(doping_pair[i,:]) > 1:
            doubled_elements = True
            # If more than one match exclude the element from the row that has moe than one match and from all colums where these doubled matches are.
            if el1 not in multiple_match_els:
                multiple_match_idx.append(i)
                multiple_match_els.append(el1)
            for j, el2 in enumerate(elements):
                if doping_pair[i,j] == True and el2 not in multiple_match_els:
                    multiple_match_idx.append(j)
                    multiple_match_els.append(el2)
    rest_els = rest_els + multiple_match_els
    
    # For each match that is unique write the element with the higher quantity in main_els and the one with the lower quantity in doped_els.
    main_els, doped_els = [], []
    for i, (el1, quant1) in enumerate(zip(elements, quantities)):
        if el1 in multiple_match_els:
                continue
        for j, (el2, quant2) in enumerate(zip(elements, quantities)):
            if i >= j or el2 in multiple_match_els or doping_pair[i,j] == False:
                continue
            if quant1 >= quant2:
                main_els.append(el1)
                doped_els.append(el2)
            else:
                main_els.append(el2)
                doped_els.append(el1)
    
    # If any element has no match at all just add it to rest_els.
    rest_els = rest_els + [el for el in chemdict.keys() if el not in rest_els+doped_els+main_els]
    
    # Assert uniqueness and same number of elements as in the beginning.
    all_els = rest_els + main_els + doped_els
    assert len(set(all_els)) == len(all_els), (all_els, chemdict)
    assert len(all_els) == len(chemdict), (all_els, chemdict)
    
    if verbose and doubled_elements:
        warnings.warn(f"Doping element doubled in chemdict {chemdict}.")
    
    return(main_els, doped_els, rest_els, doubled_elements)

# Test find_doping_pairs
# formulas = ["Cu0.5Au0.3Gd0.2", "Ce0.5Nd0.5O0.5F0.5Bi0.3S1.8U0.2Ag0.3Au1.7", "Ba3.2Cu6La0.8Y2O14.2", "Te0.7As0.3", "Cu2Gd1.2Pu0.8", "Cu1.8Au0.2O0.8", "Cu0.5Au0.5", "Bi4Ca3Cu3.88Fe0.12Sr3Onan", "Ba0.8Cu1Y1.2O4", "Bi2Ca0.46Cu2Pr0.54Sr2Onan", "Ba4Cu7Pr2O15", "Ba2Cu3Y1O6.86"]
# for formula in formulas:
#     chemdict = get_chem_dict(formula)
#     el_pairs = find_doping_pairs(chemdict)
#     print(f"Formula: {formula}: Doping pairs: {el_pairs}")

def prepare_df_2(df_2, rename_df_2):
        '''Prepares and cleans up the dataframe of the crystal database to make it more general.'''

        df_2 = df_2.rename(rename_df_2, axis="columns")
        
        # Check if all entries are good, but this should be the case.
        assert all(df_2["formula"].apply(if_valid_formula)), "Invalid entries found, see above."
        
        # Remove leading underscores because df.itertuples() doesn't play well with this.
        rename_underscores = {col: col.lstrip("_") for col in df_2.columns if col.startswith("_")}
        if len(rename_underscores) > 0:
            df_2 = df_2.rename(columns=rename_underscores)
            print("Removed leading underscores from column names in df_2 because this leads to errors.")        
        assert not any(df_2.columns.duplicated()), "Duplicates in {df_sc_columns}"
        
        #  Rename columns to show to which dataset they belong.
        df_2 = df_2.rename({col: col+"_2" for col in df_2.columns}, axis="columns")
        print("Prepared dataframe.")
        return(df_2)
    

def prepare_df_sc(df_sc, rename_sc):
    '''Prepares a df of the Supercon database to be usable to merge with a df of the crystal databases.'''
    
    df_sc = df_sc.rename(rename_sc, axis="columns")
    
    # All formulae should be valid
    if not all(df_sc["formula"].apply(if_valid_formula)):
        warnings.warn("Invalid formula encountered, see above.")
    
    # Remove leading underscores because df.itertuples() doesn't play well with this.
    rename_underscores = {col: col.lstrip("_") for col in df_sc.columns if col.startswith("_")}
    if len(rename_underscores) > 0:
        df_sc = df_sc.rename(columns=rename_underscores)
        print("Removed leading underscores from column names in df_sc because this leads to errors.")        
    assert not any(df_sc.columns.duplicated()), "Duplicates in {df_sc_columns}"
    
    #  Add suffix to columns to show to which dataset they belong.
    df_sc = df_sc.rename({col: col+"_sc" for col in df_sc.columns}, axis="columns")
    print("Prepared Supercon dataframe.")
    return(df_sc)

    

    
    
    
    
    

