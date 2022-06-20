#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:59:22 2021

@author: Timo Sommer

This script contains a class for checking if the output of the Machine_Learning() class is the same as in one reference directory. It is particular useful for automatically checking if the code still does the same after refactoring.
"""
import os
import superconductors_3D.machine_learning.Custom_Machine_Learning_v1_3 as ML
import pandas as pd
import joblib
import pickle
import torch
import filecmp
import warnings
import tempfile

class Refactoring():
    """This class is used for refactoring. It checks if the files in one directory match the files in another comparison directory, except for some things that must change like clocktime.
    """
    def __init__(self, cmp_dir):
        self.cmp_dir = os.path.abspath(cmp_dir)
        
        self.expected_files = ['All_scores.csv', 'All_values_and_predictions.csv', 'Domain_statistics.csv', 'Numerical results', 'hparams.yml', 'arguments']
        self.expected_dirs = ['models', 'plots']
        
        self.out_filename = 'Refacturing_result.txt'
        self.skip_files = [self.out_filename]
    
    def check_if_content_as_expected(self):
        """Checks if all of the expected files and directories in the comparison directory are there and if there are no unexpected files. Otherwise just prints messages.
        """
        contents = sorted(os.listdir(self.cmp_dir))
        expected_contents = sorted(self.expected_files + self.expected_dirs)
        too_few = [name for name in expected_contents if not name in contents]
        too_much = [name for name in contents if not name in expected_contents]
        for content in too_few:
            if content in self.skip_files:
                continue
            self.print(f'Expected but not found: {content}')
        for content in too_much:
            if content in self.skip_files:
                continue
            self.print(f'Unexpectedly found: {content}')
        return()
    
    def indent(self, text):
        space = '    '
        text = space + text.replace('\n', '\n' +  space)
        return(text)
        
    def print(self, text, to_console=True):
        """Prints text to output file. If to_console=True, also prints to console.
        """
        outpath = os.path.join(self.new_dir, self.out_filename)
        with open(outpath, 'a') as f:
            f.write('\n' + text)
        if to_console:
            print(text)
        return()
    
    def check_if_df_is_close(self, filename, d1, d2):
        """Checks if a df is at least close to another df.
        """
        f1 = os.path.join(d1, filename)
        f2 = os.path.join(d2, filename)
        df1, _ = ML.load_df_and_metadata(f1)
        df2, _ = ML.load_df_and_metadata(f2)
        try:
            pd.testing.assert_frame_equal(df1, df2)
            is_close = True
            error = ''
        except AssertionError as e:
            is_close = False
            error = str(e)
        return(is_close, error)
        
    def cmp_Numerical_results(self, filename1, filename2):
        """Compares the files 'Numerical results' in the two directories and returns True if they are equal except for the time.
        """
        with open(filename1) as f1:
            with open(filename2) as f2:
                # Exclude the first line that includes the time of the run.
                relevant1 = f1.readlines()[1:]
                relevant2 = f2.readlines()[1:]
                equal = relevant1 == relevant2
        return(equal)
    
    def cmp_pytorch_NN(self, NN1, NN2):
        """If the model is a pytorch model, there will be non deterministic behaviour. Therefore only compare the state_dicts in this case.
        """
        params1 = list(NN1.state_dict().values())
        params2 = list(NN2.state_dict().values())
        same_n_weights = len(params1) == len(params2)
        equal = same_n_weights and all([torch.equal(w1, w2) for w1, w2 in zip(params1, params2)])
        return equal
    
    def cmp_GP_on_NN_featurizer(self, regr1, regr2):
        """If the model is a GP model that loads a pytorch NN, the attribute that is a path that points to the pytorch model changes between runs. Therefore before comparing these models overwrite these attributes.
        """
        # Check NNs of GP_on_NN for equality.
        NN1 = regr1.regressor_['model'].NN_featurizer
        NN2 = regr2.regressor_['model'].NN_featurizer
        equal_NNs = self.cmp_pytorch_NN(NN1, NN2)
        
        # Overwrite NN_stuff of GP_on_NN.
        regr1.regressor_['model'].NN_path =  ''
        regr2.regressor_['model'].NN_path = ''
        regr1.regressor_['model'].NN_featurizer = ''
        regr2.regressor_['model'].NN_featurizer = ''
        regr1.regressor['model'].NN_path = ''
        regr2.regressor['model'].NN_path = ''   
        
         # Check GP with overriden NN stuff for equality.
        with tempfile.NamedTemporaryFile() as tmp_file1:
            with tempfile.NamedTemporaryFile() as tmp_file2:    
                pickle.dump(regr1, open(tmp_file1.name, 'wb'))
                pickle.dump(regr2, open(tmp_file2.name, 'wb'))
                equal_GPs = filecmp.cmp(tmp_file1.name, tmp_file2.name, shallow=True)
                
        equal = equal_NNs and equal_GPs
        return equal
                
    def cmp_joblib(self, filename1, filename2):
        """Compare joblib files.
        """
        regr1 = joblib.load(filename1)
        regr2 = joblib.load(filename2)
        try:
            NN1 = regr1.regressor_['model'].trainer
            NN2 = regr2.regressor_['model'].trainer
            try:
                # pytorch model
                equal = self.cmp_pytorch_NN(NN1, NN2)
            except AttributeError:
                # pytorch lightning model
                NN1 = NN1.model
                NN2 = NN2.model
                equal = self.cmp_pytorch_NN(NN1, NN2)

        except AttributeError:
            try:
                equal = self.cmp_GP_on_NN_featurizer(regr1, regr2)
            except AttributeError:
                equal = filecmp.cmp(filename1, filename2, shallow=True)
        return(equal)
    
    def cmp_files(self, filename, d1, d2):
        """Compares if two files are the same. In some cases, it is clear that files cannot be the same after refacturing, in this case this function compares only what can be compared.
        """
        pkl_files = ['.pkl', '.joblib']
        f1 = os.path.join(d1, filename)
        f2 = os.path.join(d2, filename)      
        if filename == 'Numerical results':
            equal = self.cmp_Numerical_results(f1, f2)
        elif any([filename.endswith(pkl_file) for pkl_file in pkl_files]):
            equal = self.cmp_joblib(f1, f2)
        else:
            equal = filecmp.cmp(f1, f2, shallow=False)
        return(equal)
    
    def not_equal(self, filename, new_path, cmp_path):
        """Executed if files are not equal. Prints the issues.
        """
        self.print(f'\nFile not equal: {filename}')
        # If file is df, check if dfs are at least close.
        if filename.endswith('.csv'):
            is_close, error = self.check_if_df_is_close(filename, new_path, cmp_path)
            if is_close:
                self.print('--> But dfs are close.')
            else:
                self.print('--> Dataframes are not even close.')
                print(f'--> See file {self.out_filename} for further information.')
                error = self.indent(error)
                self.print(f'--> Error:\n{error}', to_console=False)            
    
    def both_exist(self, filename, d1, d2):
        f1 = os.path.join(d1, filename)
        f2 = os.path.join(d2, filename)
        exist = True
        for f in (f1, f2):
            if not os.path.exists(f):
                self.print(f'File {f} doesn\'t exist.')
                exist = False
        return(exist)
                
    def check(self, new_dir):
        """Checks if the original `cmp_dir` matches the new directory `new_dir` in all important files. If yes, refacturing was successfull.
        """
        self.new_dir = os.path.abspath(new_dir)
        
        self.print(f'\nCheck refactoring against comparison directory {self.cmp_dir}.')
        if not os.path.exists(self.cmp_dir):
            warnings.warn(f'Comparison directory {self.cmp_dir} doesn\'t exist!\n Exiting without check.')
            return(False)
        
        self.check_if_content_as_expected()
        
        # Check every file if they are equal and otherwise print message.
        dirs_equal = True
        for new_path, _, new_files in os.walk(self.new_dir):
            # New path including subdirectories of the cmp directory.
            cmp_path = new_path.replace(self.new_dir, self.cmp_dir)
            for filename in new_files:
                if filename in self.skip_files or not self.both_exist(filename, new_path, cmp_path):
                    continue
                equal = self.cmp_files(filename, new_path, cmp_path)
                if equal and filename in self.expected_files:
                    print(f'File equal: {filename}')
                if not equal:
                    dirs_equal = False
                    self.not_equal(filename, new_path, cmp_path)                        
                    
        if dirs_equal:
            self.print('\nAll relevant files of the directories are equal.\nRefactoring successfull!\n')
        else:
            self.print('\nThere are some relevant files that are not equal.\nRefactoring not successfull!\n')
        return(dirs_equal)