#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:37:03 2021

@author: timo
This script cleans cif files by reading them in a pymatgen structure before writing them to a cif file again. Each crystal dataframe must have
- column `cif` containing a relative projectpath to the cif
- column `database_id` containing the id of this entry like '{database_name}-{id}'
"""
import os
from joblib import Parallel, delayed, cpu_count
from superconductors_3D.utils.projectpaths import projectpath
import datetime
import pandas as pd
import io
import warnings
import pathlib
import signal
import pymatgen
import gemmi
from pymatgen.io.cif import CifParser
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
import numpy
import copy
from superconductors_3D.dataset_preparation.utils.check_dataset import if_valid_formula, set_column_type, correct_symmetry_cell_setting, extract_float_values, consistent_spacegroup_and_crystal_system, assert_structure_consistent, get_normalised_spg, check_and_complement_structure, get_chemical_composition, filter_entries, normalise_pymatgen_spacegroups
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv


def get_cif_as_string(path_to_cif):
    """ Import cif as string."""
    string = pathlib.Path(path_to_cif).read_text()
    return(string)


def invalid_cif_output(reason):
    # Invalid cif found.
    valid_cif = False
    spacegroup = np.nan
    cif_pymatgen = np.nan
    pymatgen_formula = np.nan
    crystal_system = np.nan
    lata = np.nan
    latb = np.nan
    latc = np.nan
    new_cif = np.nan
    output = (valid_cif, spacegroup, crystal_system, pymatgen_formula, lata, latb, latc, new_cif, reason)
    return(output)

def check_cif_structure(input_cif_path, verbose):
    """If rigid == False, checks using pymatgen if cif is valid and if possible tries to correct little mistakes, otherwise returns False. Also returns the calculated spacegroup and the normalised cif file by pymatgen. If rigid == True it is checked that there are no issues (except rounding of floats) at all."""
    
    # Catch warnings of pymatgen correcting little mistakes.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF.*")
        warnings.filterwarnings("ignore", message="No _symmetry_equiv_pos_as_xyz type key found.*")
        warnings.filterwarnings("ignore", message="Some occupancies.*")
        warnings.filterwarnings("ignore", message="No structure parsed for 1 structure in CIF.*")
        warnings.filterwarnings("ignore", message="data_.*")
        warnings.filterwarnings("ignore", message="Error is Species occupancies sum to more than 1.*")    
        warnings.filterwarnings("ignore", message=".*Some fractional co-ordinates rounded to ideal values to avoid issues with finite precision.*")
        warnings.filterwarnings("ignore", message=".*No electronegativity for.*")
        
        
        try:
            cif_parser = CifParser(input_cif_path, occupancy_tolerance=1.1, site_tolerance=0.001)
            structures = cif_parser.get_structures()    # primitive unit cell
            
            # I would not know what to do with such structures.
            if len(structures) > 1:
                if verbose:
                    print("Too many structures.")
                raise ValueError()
            structure = structures[0]
            
        except TimeoutError:
            # If function took too long because the structure was too big.
            reason = "cif took too long"
            output = invalid_cif_output(reason)
            return(output)
        except Exception as e:   
            # cif seems to be bad and will be excluded.
            if verbose:
                print(f"Error in cif found: {e}")
            reason = "Invalid cif"
            output = invalid_cif_output(reason)
            return(output)        
          
        try:
            # Get cif file and structure data of corrected structure.
            new_cif = CifWriter(structure, symprec=0.1, refine_struct=False)
            symmetries = SpacegroupAnalyzer(structure, symprec=0.1)
            spacegroup = symmetries.get_space_group_symbol()
            spacegroup = normalise_pymatgen_spacegroups(spacegroup)
            crystal_system = symmetries.get_crystal_system()
        except TimeoutError:
            # If function took too long because the structure was too big.
            reason = "cif took too long"
            output = invalid_cif_output(reason)
            return(output)
        except:
            # If symmetry analyzer fails, just write the cif without symmetry information and leave spacegroup/ crystal system empty.
            new_cif = CifWriter(structure, refine_struct=False)
            spacegroup = np.nan
            crystal_system = np.nan
        
        try:
            # Test if newly generated cif file is really good (for some reason sometimes it is not!)
            io_file = io.StringIO()
            print(new_cif, file=io_file)
            new_cif_string = io_file.getvalue()
            new_cif_parser = CifParser.from_string(new_cif_string)
            # Structure should already be primitive
            new_structure = new_cif_parser.get_structures(primitive=False)[0]
            same_struct = StructureMatcher().fit(structure, new_structure, symmetric=True)
            if not same_struct:
                if verbose:
                    print("pymatgen cif has become different")
                reason = "pymatgen cif has become different"
                output = invalid_cif_output(reason)
                return(output)
        except TimeoutError:
            # If function took too long because the structure was too big.
            reason = "cif took too long"
            output = invalid_cif_output(reason)
            return(output)
        except:
            # Sometimes the cif cannot be read in after being written by pymatgen even if it was good before. Still this is unhelpful, therefore this file will be excluded.
            if verbose:
                print("pymatgen cif file still has errors")
            reason = "Invalid pymatgen cif"
            output = invalid_cif_output(reason)
            return(output)
            
        # cif is good, collect important structure data for the dataframe.
        lat = structure.lattice
        lata = lat.a
        latb = lat.b
        latc = lat.c        
        valid_cif = True     
        pymatgen_formula = structure.formula
        reason = np.nan
        output = (valid_cif, spacegroup, crystal_system, pymatgen_formula, lata, latb, latc, new_cif, reason)
        
        return(output)

def handler(signum, frame):
    """This hander says what to do when a function takes too long time to run."""
    print("Function took too long!")
    raise TimeoutError("End of time")

def check_and_correct(row, output_dir_cleaned_cifs, df, timeout_min, verbose):
    """Checks cif files and saves corrected cif files. Also returns pymatgen crystal structure data.
    """
    raw_cif_path = projectpath(row['cif'])
    
    # If the function takes too long to finish (because of huge cifs e.g.) then interrrupt it and handle it as an invalid structure.
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_min*60)
    try:
        # Generate new pymatgen cif and structure data from the original one.
        valid_cif, spacegroup_pymatgen, crystal_system_pymatgen, formula_pymatgen, lata, latb, latc, new_cif, reason = check_cif_structure(raw_cif_path, verbose)
        signal.alarm(0)
    except TimeoutError:
        valid_cif = False
        spacegroup_pymatgen = np.nan
        crystal_system_pymatgen = np.nan
        formula_pymatgen = np.nan
        lata = np.nan
        latb = np.nan
        latc = np.nan
        new_cif = np.nan
        reason = "cif took too long"
  
    # Write all data to dict for the dataframe later.
    datadict = {}
    for col in df.columns:
        datadict[col] = row.loc[col]
    datadict["valid_cif"] = valid_cif
    datadict["formula_pymatgen"] = formula_pymatgen
    datadict["spacegroup_pymatgen"] = spacegroup_pymatgen
    datadict["crystal_system_pymatgen"] = crystal_system_pymatgen
    datadict["lata_pymatgen"] = lata
    datadict["latb_pymatgen"] = latb
    datadict["latc_pymatgen"] = latc
    datadict["Reason for exclusion"] = reason
                
    if valid_cif == True:            
        # Write new cif file with corrected little mistakes.
        pymatgen_cif_name = row['database_id'] + '.cif'
        pymatgen_cif_path = os.path.join(output_dir_cleaned_cifs, pymatgen_cif_name)
        new_cif.write_file(projectpath(pymatgen_cif_path))
        datadict["cif_pymatgen_path"] = pymatgen_cif_path
    else:
        datadict["cif_pymatgen_path"] = np.nan
    return(datadict)

def round_up_int(x):
    result = int(np.ceil(x))
    return(result)

def clean_cifs(input_raw_csv_data, output_csv_cleaned_with_pymatgen, output_excluded, output_dir_cleaned_cifs, comment, comment_excluded, database, n_cpus, timeout_min, crystal_db_frac=1, verbose=True):
        
    df = pd.read_csv(input_raw_csv_data, header=1)
    
    # For debugging.
    if crystal_db_frac != 1:
        print(f'Downsampling crystal database for debugging to a fraction of {crystal_db_frac}.')
        df = df.sample(frac=crystal_db_frac)
    
    # My raw ICSD dataframe is not yet perfectly formatted.
    if database == 'ICSD':
        # Doesn't have database_id in the raw data yet.
        df['database_id'] = database + '-' + df['_database_code_icsd'].astype(str)
        assert not any(df.duplicated('database_id'))
        # In the raw data the cifs have an absolute path.
        cif_dir = os.path.join('data', 'source', 'ICSD', 'raw', 'cifs')
        cif_names = df['file_id'].str.split('/').str[-1].astype(str)
        df['cif'] = cif_names.apply(lambda cifname: os.path.join(cif_dir, cifname))
    
    print(f'Number of cpus: {n_cpus}')
    print(f'Number of lines in csv: {len(df)}')
        
    batch_size = round_up_int(len(df) / n_cpus)
    with Parallel(n_jobs=n_cpus, verbose=1, batch_size=batch_size, pre_dispatch='all') as parallel:
        datadict_list = parallel(delayed(check_and_correct)(row, output_dir_cleaned_cifs, df, timeout_min, verbose) for _, row in df.iterrows())
    
    print("Save data with good cifs and with bad cifs in two different dataframes.")     
    df = pd.DataFrame(datadict_list)
    
    # Exclude entries that had not correctable errors in their cif.
    df_excluded = pd.DataFrame()
    excl_condition = ~ df["valid_cif"]
    excl_reason = df["Reason for exclusion"].dropna()
    df_correct, df_excluded = filter_entries(excl_condition, excl_reason, df, df_excluded)

    print("Save good data chunk.")
    write_to_csv(df_correct, output_csv_cleaned_with_pymatgen, comment) 
    print("Save excluded data chunk.")
    write_to_csv(df_excluded, output_excluded, comment_excluded)
    
    
    

if __name__ == "__main__":    
    
    database = 'ICSD'     # Only for simple change of input paths.

    input_raw_csv_data = projectpath('data', 'source', database, 'raw', f'0_all_data_{database}.csv')
    output_csv_cleaned_with_pymatgen = projectpath('data', 'source', database, 'cleaned', f'1_all_data_{database}_cifs_normalized.csv')
    output_excluded = projectpath('data', 'source', database, 'cleaned', f'excluded_1_all_data_{database}_cifs_normalized.csv')
    output_dir_cleaned_cifs = os.path.join('data', 'source', database, 'cleaned', 'cifs')
    
    comment = f"All the data from {input_raw_csv_data}, but the cif files are cleaned/normalised by reading them in a pymatgen structure and writing them in a cif file again. Additionally chemical formula, spacegroup and crystal system are calculated by pymatgen and written in own columns. If a cif file has too bad errors the entry is excluded."
    comment_excluded = f"All the data that was not included in {output_csv_cleaned_with_pymatgen} because the cif file was too bad."
    
    n_cpus = 3
    timeout_min = 2        # timeout per structure (in minutes)
    
    print(f'Database: {database}')
    clean_cifs(input_raw_csv_data, output_csv_cleaned_with_pymatgen, output_excluded, output_dir_cleaned_cifs, comment, comment_excluded, database, n_cpus, timeout_min)

    
    








