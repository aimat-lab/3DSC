#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:48:23 2021

@author: Timo Sommer

This script contains implementations of scalers useful for me.
"""
import numpy as np


class Arcsinh_Scaler():
    
    def __init__(self, x_min=np.arcsinh(0), x_max=np.arcsinh(200/2)):
        self.x_min = x_min
        self.x_max=x_max
        assert np.isclose(self.inv_scaler(self.scaler(23)), 23)
    
    def scaler(self, x):
        return restricted_arcsinh(x)
    
    def inv_scaler(self, x):
        return restricted_sinh(x, x_min=self.x_min, x_max=self.x_max)

def restricted_sinh(x, x_min=np.arcsinh(0), x_max=np.arcsinh(200/2)):
    """Returns sinh(x)*2 if x is between x_min and x_max and otherwise returns x_min/ x_max.
    """
    # Revert norm of restricted_arcsinh.
    norm = np.arcsinh(1/2) *10
    y = x * norm
    # Clip to avoid inf for outliers because of exponential scaling.
    y = np.clip(y, x_min, x_max)
    # Scale.
    y = np.sinh(y) * 2
    return(y)

def restricted_arcsinh(x):
    """Just arcsinh(x/2) but named so for consistency with restricted_sinh(). The norm is so that restricted_arcsinh(0) = 0 and restricted_arcsinh(1) = 1 so that one can just leave it for classification. The factor 2 is for making it closer to the logarithm log(x).
    """
    # Normalize it so that restricted_arcsinh(0) = 0 and restricted_arcsinh(1) = 1 so that one can just leave it for classification.
    norm = np.arcsinh(1/2) *10  # factor 10 to make Tc roughly in between 0 and 1
    return np.arcsinh(x/2) / norm

def restricted_log(x, eps=1e-6):
    norm = np.log((1+eps)/eps)
    return np.log((x + eps)/eps) / norm

def restricted_exp(x, Tc_min=0, Tc_max=200, eps=1e-6):
    x_min = restricted_log(Tc_min)
    x_max = restricted_log(Tc_max)
    x = np.clip(x, x_min, x_max)
    Tc = eps*np.exp(x * np.log((1+eps)/eps)) - eps
    return(Tc)