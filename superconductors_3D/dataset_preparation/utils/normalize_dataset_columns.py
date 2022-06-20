#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:13:14 2021

@author: timo
This script is a library for functions that transform different cif datasets like the Materials Projekt, the COD and the ICSD into the same shape (i.e. the same column names.)
"""

rename_to_COD = {
                    '_space_group_name_h-m_alt': '_symmetry_space_group_name_H-M',
                    '_symmetry_space_group_name_h-m': '_symmetry_space_group_name_H-M',
                    '_space_group_it_number': '_space_group_IT_number'
                    }

rename_to_Sc = {
                    '_chemical_formula_sum': 'original_formula',
                    'cif': 'original_cif',
                    'cif_pymatgen_path': 'cif',
                    'full_formula': 'original_formula',
                    'formula_pymatgen': 'formula',
                    'spacegroup_pymatgen': 'spacegroup',
                    'crystal_system_pymatgen': 'crystal_system',
                    'lata_pymatgen': 'lata',
                    'latb_pymatgen': 'latb',
                    'latc_pymatgen': 'latc',
                    'volume': 'cell_volume',
                    'tags': 'comment',
                    'commt': 'comment',
                    '_database_code_icsd': 'database_code_icsd',
                    '_cod_database_code': 'database_code_cod',
                    '_diffrn_ambient_pressure': 'pressure',
                    '_cell_measurement_temperature': 'crystal_temp'
                    }

rename_MP = {
                    'spacegroup_pymatgen': 'spacegroup',
                    'crystal_system_pymatgen': 'crystal_system',
                    'lata_pymatgen': 'lata',
                    'latb_pymatgen': 'latb',
                    'latc_pymatgen': 'latc'
                    }
