#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:51:03 2021

@author: Timo Sommer

Utils for dealing with crystals.
"""
import pandas as pd
import gemmi

state_features = ['crystal_temp_2'] + ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal'] + ['primitive', 'base-centered', 'body-centered', 'face-centered']

# In the order of Wikipedia
crys_sys_dict = {
    'cubic': {'23': 1, 'm-3': 2, '432': 4, '-43m': 5, 'm-3m': 6},
    'hexagonal': {'6': 1, '-6': 2, '6/m': 3, '622': 4, '6mm': 5, '-62m': 6, '6/mmm': 7},
    'trigonal': {'3': 1, '-3': 2, '32': 4, '3m': 5, '-3m': 6},
    'tetragonal': {'4': 1, '-4': 2, '4/m': 3, '422': 4, '4mm': 5, '-42m': 6, '4/mmm': 7},
    'orthorhombic': {'222': 4, 'mm2': 6, 'mmm': 7},
    'monoclinic': {'2': 1, '2/m': 3, 'm': 5},
    'triclinic': {'1': 1, '-1': 2}
    }
all_crys_sys = sorted(crys_sys_dict.keys())

all_bravais_centrings = ['primitive', 'base-centered', 'body-centered', 'face-centered']

def One_Hot_crystal_system_and_point_group(crystal_system, point_group):
    """One Hot encodes crystal systems with their corresponding point groups so that each crystal group has one feature (like in One Hot) but instead of only a 1 in this feature the point group is encoded as an integer.
    """
    One_Hot = {cry: crys_sys_dict[crystal_system][point_group] if crystal_system == cry else 0 for cry in all_crys_sys}
    assert len(One_Hot) == 7
    One_Hot = pd.Series(One_Hot)
    return(One_Hot)

def point_group_from_space_group(spg):
    spg = gemmi.SpaceGroup(spg)
    point_group = spg.point_group_hm()
    return point_group

def One_Hot_bravais_centring(spg):
    """One Hot encodes the bravais centring (primitive, body-centered, face-centered, base-centered) given the space group name.
    """
    bravais_centring = bravais_centring_from_spacegroup(spg)
    One_Hot = {br: 1 if bravais_centring == br else 0 for br in all_bravais_centrings}
    One_Hot = pd.Series(One_Hot)
    return One_Hot

def bravais_centring_from_spacegroup(spg):
    name = gemmi.SpaceGroup(spg).hm
    centring = name[0]
    centring_to_bravais = {'P': 'primitive', 'I': 'body-centered', 'F': 'face-centered', 'A': 'base-centered', 'B': 'base-centered', 'C': 'base-centered', 'R': 'primitive'}
    bravais = centring_to_bravais[centring]
    return bravais

