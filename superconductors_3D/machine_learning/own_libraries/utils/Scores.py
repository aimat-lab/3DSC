#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:15:04 2021

@author: Timo Sommer

This script contains custom scores for the ML models.
"""
import numpy as np
import warnings



def SMAPE(ytrue, ypred, sample_weight=None, multioutput='uniform_average'):
    """Calculates the symmetrical mean absolute percentage error.
    """
    norm = (ytrue + ypred)
    diff = np.abs(np.abs(ytrue) - np.abs(ypred))
    score = diff / norm
    # If ytrue and ypred are both zero the score is NaN because norm is 0 but should be 0.
    good_zero = (ytrue == 0) & (ypred == 0)
    score = np.where(good_zero, 0, score)
    
    assert not any(np.isnan(score))
    
    if multioutput == 'raw_values':
        # Return individual scores per data point.
        return score
    
    if len(score) > 0:
        score = np.average(score, weights=sample_weight)
    else:
        warnings.warn('The SMAPE can not be calculated, setting to 0.')
        score = 0
    return score