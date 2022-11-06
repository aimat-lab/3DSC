#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:12:31 2021

@author: Timo Sommer

This script is a collection of classes for saving and loading models of the ML script.
"""
from megnet.models import MEGNetModel
import os
import pickle
import io

def get_modelpath(outdir, modelname, repetition):
    """Returns path without extension where a specific model would be saved. Used internally and externally. Outdir is the  directory of the whole run with a subdirectory 'models'.
    """
    filename = f'{modelname}_{repetition}'
    path = os.path.join(outdir, 'models', filename)
    return(path)

def regressor_from_pipeline(pipe):
    """Returns the ML model from a given sklearn Pipeline or TransformedTargetRegressor.
    """
    return pipe.regressor_['model']

class Pickle_tf():
    """Use this class to pickle tensorflow models that cannot be pickled normally because of some ThreadLock error.
    """
    def __init__(self):
        pass
    
    def model_is_MEGNet(self, modelpath):
        """Checks by the modelpath if the model is a MEGNET model.
        """
        tf_filename = self.get_tf_filepath(modelpath)
        try:
            MEGNetModel.from_file(tf_filename)
            is_MEGNet = True
        except FileNotFoundError:
            is_MEGNet = False
            
        return is_MEGNet
    
    def get_tf_filepath(self, modelpath):
        """Turns the pickle path into the tf model path.
        """
        return os.path.splitext(modelpath)[0] + '.hdf5'
    
    def save(self, regr, modelpath):
        """Saves a tf or MEGNet model. Part of the sklearn pipeline is saved as pickle and the not-pickable tf part is saved as tf model.
        """
        tf_filename = self.get_tf_filepath(modelpath)
        try:
            # MEGNet
            tf_model = regressor_from_pipeline(regr).model
            # Save non-pickable part as hd5.
            tf_model.save_model(filename=tf_filename)
        except AttributeError as e:
            # Other tf models
            raise NotImplementedError(f'Not implemented for other model than MEGNet. Error message: {e}')
            
        # Remove non-pickable part from regressor.
        regressor_from_pipeline(regr).model = None
        
        # Pickle regressor.
        with open(modelpath, 'wb') as f:
            pickle.dump(regr, f)
        
        # Put non-pickable model back into pipeline to not change regr from the outside. Somehow using deepcopy doesn't work.
        regressor_from_pipeline(regr).model = tf_model
    
    def load(self, modelpath):
        """Loads a tf model composed of pickled regressor/pipeline and tf_model saved as hdf5.
        """
        regr = Models().load_pickle(modelpath)
        tf_filepath = self.get_tf_filepath(modelpath)
        try:
            # MEGNet            
            tf_model = MEGNetModel.from_file(tf_filepath)
            regressor_from_pipeline(regr).model = tf_model
        except AttributeError as e:
            # Other tf models
            raise NotImplementedError(f'Not implemented for other model than MEGNet. Error message: {e}')
        
        return regr


class Models():
    """Class for saving and loading ML models.
    """
    
    def __init__(self):
        pass
        
    def save(self, regr, rundir, modelname, repetition, save_torch_statedict=False):
        """Saves model as pickle and if possible as pytorch models.
        """
        filename = get_modelpath(rundir, modelname, repetition)
        outpath = filename + '.pkl'
        try:
            with open(outpath, 'wb') as f:
                pickle.dump(regr, f)
                
        except (TypeError, AttributeError) as e:
            # tf or keras models must be saved seperately because they cannot be pickled.
            if 'pickle' in str(e):
                Pickle_tf().save(regr=regr, modelpath=outpath)                
            else:
                raise TypeError(e)
        
        if save_torch_statedict:
            outpath = filename + '.pt'
            try:
                model = regressor_from_pipeline(regr)
                import torch
                torch.save(model.trainer.state_dict(), outpath)
            except AttributeError:
                pass
        return()
    
    def load_pickle(self, modelpath):
        """Loads a pickled model from the path.
        """
        with open(modelpath, 'rb') as f:
            model = CPU_Unpickler(f).load()
        
        return model
    
    def load_from_path(self, modelpath):
        """Loads a model from the given path with .pkl extension and returns it. Can also load tensorflow models that were not simply pickled.
        """
        model_is_MEGNet = Pickle_tf().model_is_MEGNet(modelpath)
        if model_is_MEGNet:
            model = Pickle_tf().load(modelpath)
        else:
            model = self.load_pickle(modelpath)

        return model
        
    
    def load(self, modelname: str, repetition: int, rundir: str, regressor: bool=False):
        """Loads a model with given name and repetition from rundir. If `regressor==True` the returned model is not the whole sklearn pipeline/regressor but only the fitted ML model.
        """
        modelpath = get_modelpath(rundir, modelname, repetition) + '.pkl'

        model = self.load_from_path(modelpath)
        
        if regressor:
            # Get regressor model from pipeline
            model = regressor_from_pipeline(model)
            
        return(model)

class CPU_Unpickler(pickle.Unpickler):
    """Use this for unpickling instead of pickle.load(f) to be able to load a model even if it was saved on a gpu.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)