#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:10:05 2021

@author: timo
This script checks whether the conditions for synthetic doping are met and if yes, performs synthetic doping. Matches where the conditions are not met are excluded and saved in an extra df.
"""
import argparse
from superconductors_3D.utils.projectpaths import projectpath
import os
import pandas as pd
import numpy as np
import pymatgen as pmg
import tempfile
import gemmi
from copy import deepcopy
from pymatgen.analysis.structure_matcher import StructureMatcher
import warnings
from joblib import Parallel, delayed, cpu_count
import os
from collections import defaultdict 
from superconductors_3D.machine_learning.own_libraries.own_functions import flatten, only_unique_elements, intersection, write_to_csv, isfloat, movecol
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from superconductors_3D.dataset_preparation.utils.check_dataset import find_doping_pairs, get_chem_dict, standardise_chem_formula, normalise_chemdict, chemdict_to_formula, get_normalised_spg, exclude_row, normalise_pymatgen_spacegroups

def quantity_fracs(dict_to_normalise, dict_normalise, elements):
    """Calculates the fraction of the quantities of two chemdict of the provided elements.
    """
    quantities_sc_fixed = np.array([dict_to_normalise[el] for el in elements])
    quantities_struct_fixed = np.array([dict_normalise[el] for el in elements])
    norms = quantities_struct_fixed/quantities_sc_fixed
    return(norms)

def get_oxi_states(structure):
    """Returns a dict with all oxidation states per site and species.
    """
    oxi_states = {}
    for site_idx, site in enumerate(structure.sites):
        oxi_states[site_idx] = {}
        species = site.as_dict()['species']
        for specie in species:
            el = specie['element']
            oxi_state = specie['oxidation_state']
            oxi_states[site_idx][el] = oxi_state
    return(oxi_states)
        
def add_oxi_states(structure, oxi_states):
    """Adds oxidation states to a structure without oxidation states.
    """
    structure = deepcopy(structure)
    replace_dict = {}
    for site_idx, site in enumerate(structure.sites):
        replace_dict[site_idx] = {}
        species = site.as_dict()['species']
        for specie in species:
            el = specie['element']
            occu = specie['occu']
            try:
                oxi_state = oxi_states[site_idx][el]
            except KeyError:
                # Doping element gets the same oxidation state as host element if it has not been there before.
                host_el = list(oxi_states[site_idx].keys())[0]
                oxi_state = oxi_states[site_idx][host_el]
            oxi_species = pmg.core.periodic_table.Species(el, oxi_state)
            replace_dict[site_idx][oxi_species] = occu
    el_to_species = ReplaceSiteSpeciesTransformation(replace_dict)
    structure = el_to_species.apply_transformation(structure)
    return(structure)

def get_symm_equiv_sites(symm_struct):
    """Finds which sites in a structure are symmetrically equivalent. Additional to the symetrically equivalent structures that pymatgen recognizes because of the spacegroup disordered sites with the same chemical formula are recognized as symetrically equivalent as well because obviously they behave identical under doping (because they probably have the same chemical environment).
    
    Returns: 
        symm_sites: List of symmetrically equivalent sites, always the first site of each symmetrically equivalent group is returned.
        equiv_indices: List of list of indices of groups of symmetrically equivalent sites.
    """
    pseudo_equiv_sites = symm_struct.equivalent_sites
    pseudo_equiv_indices = symm_struct.equivalent_indices
    
    # Make sure every symmetrically equivalent sites have the same formula and get these formulae.
    pseudo_equiv_formulae = []
    for indices, sites in zip(pseudo_equiv_indices, pseudo_equiv_sites):
        formulae = [site.species.formula for site in sites]
        assert len(set(formulae)) == 1, formulae
        formula = formulae[0]
        pseudo_equiv_formulae.append(formula)
                    
    # Get only first site of each symmetrically equivalent group.
    pseudo_symm_sites = [sites[0] for sites in pseudo_equiv_sites]
    
    # If there are two sites that are not symmetrically invariant according to pymatgen but that are disordered and have the same chemical formula see these sites as symmetrically equivalent as well. This is because even though these sites are not equivalent under spatial transformations they obviously have the same energy and probably chemical environment. This is especially important because the 3D structures of many cuprates of the COD like YBCO have this issue.
    symm_sites, equiv_indices, equiv_formulae = [], [], []
    for formula, indices, site in zip(pseudo_equiv_formulae, pseudo_equiv_indices, pseudo_symm_sites):
        disordered_site = not site.is_ordered
        other_site_same_formula = equiv_formulae.count(formula) > 0
        if not (disordered_site and other_site_same_formula):
            symm_sites.append(site)
            equiv_indices.append(indices)
            equiv_formulae.append(formula)
        else:
            idx_same_formula = equiv_formulae.index(formula)
            equiv_indices[idx_same_formula] = equiv_indices[idx_same_formula] + indices
    
    assert len(symm_sites) == len(equiv_indices)
    all_equiv_indices = flatten(equiv_indices)
    all_indices = list(range(0, len(all_equiv_indices)))
    assert sorted(all_equiv_indices) == all_indices
    return(symm_sites, equiv_indices)

def free_and_fixed_sites(symm_sites, equiv_indices, structure):
    """Get indices of free and fixed sites for the normal structure and the symmetrically equivalent structure. A symmetrically equivalent site is fixed if it is ordered and it is not the only site with this element or if it is not ordered but there are other not ordered sites with the same elements. Also gets a dictionary with the elements that occur at free sites and the symmetrical sites at which they occur.
    """
    elements_per_symm_site = [list(get_chem_dict(site.species.formula)) for site in symm_sites]
    sites_ordered = [site.is_ordered for site in symm_sites]
        
    # Get indices of free and fixed sites.
    fixed_indices = []
    fixed_symm_indices = []
    for idx, (ordered, site) in enumerate(zip(sites_ordered, symm_sites)):
        formula = site.species.formula
        site_els = list(get_chem_dict(formula).keys())
        same_chem_sys = any([els == site_els for i, (els, is_ordered) in enumerate(zip(elements_per_symm_site, sites_ordered)) if not is_ordered and idx != i])
        if ordered:
            site_el = site_els[0]
            element_count = [el for site in elements_per_symm_site for el in site].count(site_el)
            # If there is another site with the same element.
            if element_count > 1:
                fixed_indices = fixed_indices + equiv_indices[idx]
                fixed_symm_indices.append(idx)
        elif same_chem_sys:
            fixed_indices = fixed_indices + equiv_indices[idx]
            fixed_symm_indices.append(idx)
            
            
    num_sites = len(structure)
    num_symm_sites = len(symm_sites)
    free_indices = [idx for idx in range(num_sites) if not idx in fixed_indices]
    free_symm_indices = [idx for idx in range(num_symm_sites) if not idx in fixed_symm_indices]
    
    free_symm_elements_and_sites = {el: idx for idx in free_symm_indices for el in elements_per_symm_site[idx]}
    
    all_indices = list(range(num_sites))
    all_symm_indices = list(range(num_symm_sites))
    assert sorted(free_indices + fixed_indices) == all_indices
    assert sorted(free_symm_indices + fixed_symm_indices) == all_symm_indices
    return(fixed_indices, free_indices, fixed_symm_indices, free_symm_indices, free_symm_elements_and_sites)

def elements_all_ordered(structure, elements_struct):
    """Returns a list of elements which sites are all ordered."""
    num_sites = len(structure)
    elements_per_site = [list(get_chem_dict(struct.species.formula)) for struct in structure]
    not_ordered_indices = [idx for idx in range(num_sites) if not structure[idx].is_ordered]
    all_not_ordered_els = list(set([el for idx in not_ordered_indices for el in elements_per_site[idx]]))
    els_all_ordered = [el for el in elements_struct if not el in all_not_ordered_els]
    return(els_all_ordered)

def formula_scaling(els_all_fixed, chemdict_sc, chemdict_struct, els_all_ordered):
    """Calculates the normalisation by wich the Supercon formula should be scaled to look close to the structure formula. If possible the scaling is done according to the fixed els and if these have not all the same, exclude_row is set because then the formula can't be modified and the row must be excluded. If there are no fixed elements the scaling will be according to the intersection of elements with round values in the Supercon formula and ordered elements in the structure. If this is empty too only the ordered elements will be taken for scaling.
    """
    row_excluded = False
    if len(els_all_fixed) > 0:
        # Scale the chemical formula of the Supercon by the fixed elements of the structure.
        norms = quantity_fracs(chemdict_sc, chemdict_struct, els_all_fixed)
        norms_unique = np.unique(norms)
        if len(norms_unique) > 1:
            row_excluded = True
            norm = None
            return(norm, row_excluded)
        else:
            norm = norms_unique[0]
    else:
        # Scale the quantities by comparing the quantities of all elements that are ordered in the structure and that are round in the Supercon formula because this should be the most natural.
        els_with_round_quants = [el for el, quant in chemdict_sc.items() if quant == round(quant)]
        elements_struct = list(chemdict_struct.keys())
        els_to_compare = [el for el in elements_struct if el in els_all_ordered and el in els_with_round_quants]    
    
        if els_to_compare:
            norms = quantity_fracs(chemdict_sc, chemdict_struct, els_to_compare)
            norm = pd.Series(norms).mode().iloc[0]
            # Just checking, this should occur very rarely if at all.
            # norms_unique = np.unique(norms)
            # if len(norms_unique) > 1:
            #     print(f"Different scaling found for sc chemdict {chemdict_sc}: {norms}.")
        else:
            # If there is no obvious scaling by individual elements possible just scale by sum of quantities.
            quant_sc = np.array(list(chemdict_sc.values()))
            quant_struct = np.array(list(chemdict_struct.values()))
            norm = sum(quant_struct)/sum(quant_sc)
        
            
    return(norm, row_excluded)

def insert_additional_elements(big_doped_els, small_doped_els, els_all_fixed, chemdict_sc, chemdict_struct, free_symm_elements_and_sites, symm_sites, equiv_indices, structure, elements_sc, elements_struct):
    """Replace doping quantities in the structure like given in the Supercon formula.
    """
    exclude_reason = None
    
    # Update chemdict_struct with elements with quantity zero if this element is as dopant in the Supercon formula.
    chemdict_struct = {el: chemdict_struct[el] if el in elements_struct else 0 for el in elements_sc}
    assert sorted(chemdict_sc.keys()) == sorted(chemdict_struct.keys())
    
    for el1, el2 in zip(big_doped_els, small_doped_els):
        # If any of the doping elements is a fixed element in the structure (i.e. it would not be clear which site to dope) exclude this entry.
        if el1 in els_all_fixed or el2 in els_all_fixed:
            exclude_reason = "Doping not unique"
            return(structure, exclude_reason)
            
        # Find which element is the host and which the dopant. The dopant replaces the host when going from the formula of the structure to the formula of the superconductor.
        quant1_sc, quant1_struct = chemdict_sc[el1], chemdict_struct[el1]
        quant2_sc, quant2_struct = chemdict_sc[el2], chemdict_struct[el2]
        diff1 = round(quant1_struct - quant1_sc, 6)
        diff2 = round(quant2_struct - quant2_sc, 6)
        host = el1 if diff1 > 0 else el2
        dopant = el1 if diff2 > 0 else el2
        if diff1 != - diff2:
            # Definitely no doping happening here.
            continue
        diff = abs(diff1)
        if diff == 0:
            # No change needed here.
            continue
        if el1 == host and el2 == dopant:
            host_quant_sc, dopant_quant_sc = quant1_sc, quant2_sc
            host_quant_struct, dopant_quant_struct = quant1_struct, quant2_struct
        elif el2 == host and el1 == dopant:
            host_quant_sc, dopant_quant_sc = quant2_sc, quant1_sc
            host_quant_struct, dopant_quant_struct = quant2_struct, quant1_struct
        else:
            exclude_reason = "No dopant/ host recognized"
        
        # Get the index of the symmetrical site of host to know which site to manipulate.
        host_site = free_symm_elements_and_sites[host]
        site_els = list(get_chem_dict(symm_sites[host_site].species.formula).keys())
        other_el_present = any([not el in [host, dopant] for el in site_els])
        if other_el_present:
            exclude_reason = "Other el at doping site present"
    
        old_host_quant = symm_sites[host_site][host]
        old_host_occu = get_chem_dict(symm_sites[host_site].species.formula)
        old_dopant_quant = old_host_occu[dopant] if dopant in old_host_occu.keys() else 0
        all_host_site_indices = equiv_indices[host_site]
        
        # Scale the difference of the chemical formula by how many sites will be modified so that there is no doubling.
        num_symm_sites = len(all_host_site_indices)
        diff /= num_symm_sites
        
        if old_host_quant >= diff:
            # Replace the host by the dopant at all symmetrically equivalent sides.
            new_host_quant = old_host_quant - diff
            new_dopant_quant = old_dopant_quant + diff
            for site_idx in all_host_site_indices:
                replace_dict = {site_idx: {host: new_host_quant, dopant: new_dopant_quant}}
                replace_host_by_dopant = ReplaceSiteSpeciesTransformation(replace_dict)
                structure = replace_host_by_dopant.apply_transformation(structure)
        else:
            exclude_reason = "Doping too much for one site"
            return(structure, exclude_reason)
    return(structure, exclude_reason)

def modify_occupancies(free_symm_elements_and_sites, chemdict_sc, chemdict_struct, equiv_indices, structure, symm_sites):
    """Changes quantities for not doping sites like given in the Supercon formula.
    """    
    exclude_reason = np.nan
    replace_dict = {}
    for el, symm_site_idx in free_symm_elements_and_sites.items():
        quant_struct = chemdict_struct[el]
        quant_sc = chemdict_sc[el]
        diff = round(quant_struct - quant_sc, 10)
        if diff != 0:
            site_indices = equiv_indices[symm_site_idx]
            num_symm_sites = len(site_indices)
            diff /= num_symm_sites
            old_site_quant = get_chem_dict(symm_sites[symm_site_idx].species.formula)[el]
            new_site_quant = old_site_quant - diff
            if new_site_quant > 0 and new_site_quant < 1:
                for site_idx in site_indices:
                    try:
                        replace_dict[site_idx][el] = new_site_quant
                    except KeyError:
                        replace_dict[site_idx] = {el: new_site_quant}
            else:
                exclude_reason = "Change too much for one site"
                return(structure, exclude_reason)
    modify_quantity = ReplaceSiteSpeciesTransformation(replace_dict)
    try:
        structure = modify_quantity.apply_transformation(structure)
    except ValueError:
        exclude_reason = "Species occupancies sum to more than 1"
        return(structure, exclude_reason)
    return(structure, exclude_reason)

def get_cif_file(rel_cif_path):
    """Returns cif file as string.
    """
    cifpath = projectpath(rel_cif_path)
    with open(cifpath, 'r') as file:
        cif = file.read()
    return(cif)

def check_spacegroup(symm_analyzer, spg_2):
    """Check if spacegroup is recognized correctly.
    """
    spg_struct = normalise_pymatgen_spacegroups(symm_analyzer.get_space_group_symbol())
    spg_struct = get_normalised_spg(gemmi.find_spacegroup_by_name(spg_struct)).hm
    if spg_struct != spg_2:
        # print(f"Spacegroup of structure not spacegroup 2: {spg_struct} vs {spg_2}.")
        return(False)
    else:
        return(True)

def cif_will_be_read_in_correctly(structure, symprec):
    """Tests whether a structure will be read in again exactly as it is now when saving and reloading it from a cif file. This is is necessary e.g. because doped hydrogen is not recognized by pymatgen when reading from a cif.
    """
    with tempfile.NamedTemporaryFile() as tmp:
        name = tmp.name + '.cif'
        try:
            save_structure_to_cif(structure, name, symprec)
        except ZeroDivisionError:
            # Happens sometimes, don't know why.
            same_structs = False
            return same_structs
    new_structure = Structure.from_file(name)
    same_structs = structure == new_structure
    return same_structs

def save_structure_to_cif(structure, output_path, symprec):
    cifwriter = CifWriter(structure, symprec=symprec, refine_struct=False)
    cifwriter.write_file(output_path)
    return
    
def synthetic_doping(row, database, symprec, output_cif_dir):
    """Check whether a match can and must be synthetically doped. If no, exclude it. Otherwise, perform synthetic doping.
    """

    formula_sc = row["formula_sc"]
    formula_2 = row["formula_2"]
    formula_sc_old = formula_sc
    
    cif = get_cif_file(rel_cif_path=row['cif_2'])
    
    # Get structure from cif.
    structure = Structure.from_str(cif, fmt="cif")
    formula_struct = structure.formula
    norm_formula_struct = standardise_chem_formula(formula_struct, normalise=True)
    norm_formula_2 = standardise_chem_formula(formula_2, normalise=True)
    norm_formula_sc = standardise_chem_formula(formula_sc, normalise=True)
    
    # The ICSD has oxidation states for each element on each site. Remove oxidation states because this script was written without this in mind and otherwise bugs happen. Keep the oxidation states to add them again after the synthetic doping.
    old_structure = deepcopy(structure)
    if database == 'ICSD':
        oxi_states = get_oxi_states(structure)
        structure.remove_oxidation_states()
    
    # Get chemdicts and list of the elements.
    chemdict_sc = get_chem_dict(formula_sc)
    chemdict_struct = get_chem_dict(formula_struct)
    elements_sc = list(chemdict_sc.keys())
    elements_struct = list(chemdict_struct.keys())
    assert all(el in elements_sc for el in elements_struct)
    
    # Find pairs of dopant and host for later.
    big_doped_els, small_doped_els, *_, = find_doping_pairs(chemdict_sc, verbose=False)
    
    if not norm_formula_sc == norm_formula_struct:
        manipulated = True
        symm_analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        try:
            symm_struct = symm_analyzer.get_symmetrized_structure()
            spg_recognized = check_spacegroup(symm_analyzer, row['spacegroup_2'])
        except TypeError:
            spg_recognized = False
        
        # Exclude entry if spacegroup is not recognized correctly because this is dodgy.
        if not spg_recognized:
            reason = "Inconsistent spacegroups."
            row["Reason for exclusion"] = reason
            return(row.to_dict())
    
        # Check if symmetrical sites really have same elements and occupancy and get only one species from each symmetrical group.
        symm_sites, equiv_indices = get_symm_equiv_sites(symm_struct)
        
        # Get all indices in which the site/ species must be fixed. A site is fixed if it is ordered and it is not the only site with this element.
        ## I know that if an element has more than one not equivalent side I cannot touch it anyway. That means if all these elements do not have the right fraction to each other I must dismiss the structure anyway. Therefore get the elements that needs to be fixed and scale the whole chemical formula by them.
        fixed_indices, free_indices, fixed_symm_indices, free_symm_indices, free_symm_elements_and_sites = free_and_fixed_sites(symm_sites, equiv_indices, structure)
        unique_free_el_mapping = set(free_symm_elements_and_sites.values()) == set(free_symm_indices)
        # print(fixed_symm_indices)
                                               
        if not unique_free_el_mapping:
            reason = "Mapping of free elements and sites not unique"
            row["Reason for exclusion"] = reason
            return(row.to_dict())
        
        # Assert that if a site is free it's elements don't occur at another site to guarantee a unique mapping when manipulating these sites.
        free_symm_elements = list(free_symm_elements_and_sites.keys())
        assert only_unique_elements(free_symm_elements), "Free elements do not have a unique site."
        
        # Scale Supercon formula to look close to the structure formula. Also check if relative quantities of fixed sites are the same and exclude if not because then I couldn't manipulate the structure to mirror the formula of the superconducting entry.
        els_all_fixed = [el for el in elements_struct if not el in free_symm_elements]
        els_all_ordered = elements_all_ordered(structure, elements_struct)
        norm, row_excluded = formula_scaling(els_all_fixed, chemdict_sc, chemdict_struct, els_all_ordered)
        if row_excluded:
            reason = "Different scaling of fixed sites"
            row["Reason for exclusion"] = reason
            return(row.to_dict())
        chemdict_sc = normalise_chemdict(chemdict_sc, norm)
        formula_sc = chemdict_to_formula(chemdict_sc)        
        
        # Replace doped elements.
        structure, exclude_reason = insert_additional_elements(big_doped_els,
                                                        small_doped_els,
                                                        els_all_fixed,
                                                        chemdict_sc,
                                                        chemdict_struct,
                                                        free_symm_elements_and_sites,
                                                        symm_sites,
                                                        equiv_indices,
                                                        structure,
                                                        elements_sc,
                                                        elements_struct
                                                        )
        if pd.notna(exclude_reason):
            row["Reason for exclusion"] = exclude_reason
            return(row.to_dict())
        formula_struct = structure.formula
        chemdict_struct = get_chem_dict(formula_struct)
        
        # Add or subtract quantity for not doped sites.
        structure, exclude_reason = modify_occupancies(free_symm_elements_and_sites,
                                                       chemdict_sc,
                                                       chemdict_struct,
                                                       equiv_indices,
                                                       structure,
                                                       symm_sites
                                                       )
        if pd.notna(exclude_reason):
            row["Reason for exclusion"] = exclude_reason
            return(row.to_dict())
        formula_struct = standardise_chem_formula(structure.formula)
        
        if formula_sc != formula_struct:
            reason = "Formulas still not equal."
            row["Reason for exclusion"] = reason
            return(row.to_dict())
        
        # Test if file will correctly be read in after saving.
        read_in_correctly = cif_will_be_read_in_correctly(structure, symprec)
        if not read_in_correctly:
            reason = 'Structure will not be read in correctly.'
            row["Reason for exclusion"] = reason
            return(row.to_dict())
        
    else:
        manipulated = False
    
    formula_struct = standardise_chem_formula(structure.formula)
    norm_formula_struct = standardise_chem_formula(formula_struct, normalise=True)
    norm_formula_sc = standardise_chem_formula(formula_sc, normalise=True)
    
    # Add previously removed oxidation states.
    if database == 'ICSD':
        structure = add_oxi_states(structure, oxi_states)
    # print('Old structure:\n', old_structure)
    # print('Structure:\n', structure)
    if not manipulated:
        if not structure == old_structure:
            warnings.warn('Old structure != new structure.')
            # print('Old structure:\n', old_structure)
            # print('New_structure:\n', structure)
        # structure = old_structure
    
    # Save cif.
    # print(structure)
    formula = standardise_chem_formula(structure.formula)
    # Keep path to old cif for safety and later double checking.
    row['cif_before_synthetic_doping'] = row['cif_2']
    # New cif name is unique combination from old cif name and new chemical formula because only of the two is not unique. Also mention if this structure was synthetically doped.
    add_string = '-synth_doped' if manipulated else ''
    filename = f'{row["formula_sc"]}-{row["database_id_2"]}{add_string}.cif'
    rel_cif_path = os.path.join(output_cif_dir, filename)
    row['cif_2'] = rel_cif_path    
    # Save new cif.
    output_path = projectpath(rel_cif_path)
    # Set symprec for cif saving tight because otherwise the space group differs from the original file.
    save_structure_to_cif(structure, output_path, symprec=0.01)

    
    # Make dictionary with data from this row for the df later.
    row['formula'] = formula
    row['synth_doped'] = manipulated
    return(row.to_dict())

def perform_synthetic_doping(input_file, output_cif_dir, output_file, output_file_excluded, database, n_cpus, symprec):
        
    print(f'Start artificial doping for database {database} with {n_cpus} cpus.')
    
    df = pd.read_csv(input_file, header=1)
    
    # Test if this combination is unique so that we can use it to name the cif files.
    duplicates = df[["formula_sc", "database_id_2"]][df.duplicated(subset=['formula_sc', 'database_id_2'])]
    assert len(duplicates) == 0, 'Not only unique combinations, cif files would be overwritten. Issues: {duplicates}'
    
    # test_formulas = ["Ba1.975Cu3La0.025Y1O6.95", "Ba2Co0.7Cu2.3Y1O7.13", "Ba2Cu3Sm1O7.01", "Ba2Cu2.6Fe0.4Y1O7", "As1F0.1Fe1La0.7Y0.3O0.9", "Bi3.7Cs1Pb0.3Te6", "Ga4Mn0.006Mo0.994", "Ga0.2Nb0.8", "N0.99Ti1"]
    # df = df[df['formula_sc'].isin(test_formulas)]
    
    print(f'Doing artificial doping for {len(df)} entries.')
    with Parallel(n_jobs=n_cpus, verbose=1) as parallel:
        data = parallel(delayed(synthetic_doping)(row, database, symprec, output_cif_dir) for _, row in df.iterrows())
    
    
    # Save dataframes.
    df_results = pd.DataFrame(data)
    df_results = df_results.rename(columns={'formula_2': 'orig_formula_cif'})
    df_results = movecol(df_results, ['formula', 'orig_formula_cif'], to='formula_sc')
    success = df_results['formula'].notna()
    df_good = df_results[success]
    df_excluded = df_results[~success]
    
    # Assert no confusion between good and bad dataframe.
    assert df_good[['formula', 'synth_doped', 'cif_before_synthetic_doping']].notna().all().all()
    assert df_good['Reason for exclusion'].isna().all()
    assert df_excluded[['formula', 'synth_doped', 'cif_before_synthetic_doping']].isna().all().all()
    assert df_excluded['Reason for exclusion'].notna().all()
    
    num_sc_before = df_results['formula_sc'].nunique()
    num_sc_after = df_good['formula_sc'].nunique()
    print(f'Number of unique superconductors before artifial doping: {num_sc_before}')
    print(f'Number of unique superconductors after artificial doping: {num_sc_after}')
    print(f'Number of lost unique superconductors: {num_sc_before - num_sc_after}')
    
    
    comment = f"The matched entries from SuperCon and the crystal structure database were synthetic doping was performed successfully or was not needed in the first place."
    comment_excluded = f"The matched entries from SuperCon and the crystal structure database were synthetic doping could NOT be performed."
    write_to_csv(df_good, output_file, comment)
    write_to_csv(df_excluded, output_file_excluded, comment_excluded)
    if not len(df_good) == len(os.listdir(projectpath(output_cif_dir))):
        print('WARNING: Not all cif files seem to be in the folder.')
    
    
        
        
if __name__ == '__main__':
    
    database = 'ICSD'         # change this or parse from cmd

    n_cpus = 1             # change this or parse from cmd
    
    
    
    # Parse input and overwrite.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    args = parser.parse_args()
    
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    
    symprec = 0.1 if database == 'MP' else 0.01     # for symmetry recognition
    
    input_file = projectpath('data', 'intermediate', database, f'3_SC_{database}_matches.csv')
    
    output_cif_dir = os.path.join('data', 'final', database, 'cifs')
    
    output_file = projectpath('data', 'intermediate', database, f'4_SC_{database}_synthetically_doped.csv')
    output_file_excluded = projectpath('data', 'intermediate', database, f'excluded_4_SC_{database}_synthetically_doped.csv')
    
    synthetic_doping(input_file, output_cif_dir, output_file, output_file_excluded, n_cpus, symprec)
            
            
