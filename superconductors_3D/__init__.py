#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ignore all warnings. I know of two warnings that sometimes appear at the moment:
    # /Users/timosommer/opt/anaconda3/envs/3DSC7/lib/python3.9/site-packages/spglib/spglib.py:115: DeprecationWarning: dict interface (SpglibDataset['wyckoffs']) is deprecated.Use attribute interface ({self.__class__.__name__}.{key}) instead
    # /Users/timosommer/academic_projects/3DSC_paper/3DSC/superconductors_3D/dataset_preparation/utils/check_dataset.py:145: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    #   df_excluded = df_excluded.append(df_ex)
# These should at some point be addressed to make the code more robust, and then the corresponding pins in the requirements.txt could be relaxed.
import warnings
warnings.filterwarnings("ignore")
