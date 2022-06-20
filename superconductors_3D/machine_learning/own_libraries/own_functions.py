#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:40:19 2021

@author: timo
This is a collection of useful little self written functions.
"""

import pandas as pd
import os
from collections.abc import Iterable
import os
from datetime import datetime
import inspect


DATE = datetime.now()

def write_to_csv(df, output_path, comment):
    """Write df to csv with a comment in the first line."""
    # Get comment with execution date and filename of original script.
    caller_filename = os.path.basename(inspect.stack()[1].filename)
    date_comment=f" Generated on {DATE} by the script {caller_filename}."
    comment = comment + date_comment
    
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "a") as file:
        file.write('# "'+comment+'"\n')
        df.to_csv(file, index=False)
        print(f"Saved {len(df)} entries to {os.path.basename(output_path)}.")
    return()

def first_line(filename, strip=False):
    """Returns the first line of `filename`."""
    with open(filename) as f:
        line = f.readline()
    if strip:
        line = line.strip()
    return(line)

def insert_element(list0, el, after):
    """Return list with element inserted after another element."""
    idx = list0.index(after) + 1
    list0.insert(idx, el)
    return(list0)
# Test insert_element
# l = [1,2,4,5]
# l = insert_element(l, el=3, after=2)
# print(l)

def only_unique_elements(l):
    """Tests if a list or tuple has only unique elements."""
    is_unique = len(l) == len(set(l))
    return(is_unique)

def intersection(l1, l2):
    """Returns the intersection of two lists as list."""
    intersect = [el for el in l1 if el in l2]
    return(intersect)

def isfloat(value):
    """"Check if variable is float."""
    try:
      float(value)
      return True
    except ValueError:
      return False

def is_int(value):
    """Checks if value is round float."""
    try:
        if float(value) == round(float(value)):
            return(True)
        else:
            return(False)
    except (ValueError, TypeError):
        return(False)

def movecol(df, cols, to, place='After'):
    """Shifts columns in a dataframe, either 'After' or 'Before' the to."""
    cols_before = df.columns.tolist()
    if place == 'After':
        seg1 = cols_before[:list(cols_before).index(to) + 1]
        seg2 = cols
    if place == 'Before':
        seg1 = cols_before[:list(cols_before).index(to)]
        seg2 = cols + [to]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols_before if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])

def frequentvals(series, n, mode="exact"):
    """Gets all values in series which appear exactly/more/less than n times."""
    counts = series.value_counts()
    if mode == "exact":
        result = series[series.isin(counts.index[counts.eq(n)])]
    elif mode == "more":
        result = series[series.isin(counts.index[counts.gt(n)])]
    elif mode == "less":
        result = series[series.isin(counts.index[counts.lt(n)])]
    else:
        raise ValueError("mode")
    return(result)

def diff_rows(df1, df2):
    """Get the rows which are different between df1 and df2, NaNs are ignored."""
    if_diff = (df1 != df2) | ((df1.isna()) & (df2.isna()))
    if_diff_rows = if_diff.any(axis=1)
    if_diff_cols = if_diff.any(axis=0)
    diff_cols = df1.columns[if_diff_cols].tolist()
    print(f"Different columns (df1): {diff_cols}")
    diff_rows = df1[if_diff_rows]
    return(diff_rows)
    
def flatten(x):
    """Flattens a list/tuple."""
    result = []
    for el in x:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

import functools
import inspect

def enforce_types(f):
    sig = inspect.signature(f)
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        args = bound.arguments
        for param in sig.parameters.values():
            if param.annotation is not param.empty and not isinstance(args[param.name], param.annotation):
                raise TypeError("Parameter '%s' must be an instance of %s" % (param.name, param.annotation))
        result = f(*bound.args, **bound.kwargs)
        if sig.return_annotation is not sig.empty and not isinstance(result, sig.return_annotation):
            raise TypeError("Returning value of function '%s' must be an instance of %s" % (f.__name__, sig.return_annotation))
        return result
    return wrapper
    
    
