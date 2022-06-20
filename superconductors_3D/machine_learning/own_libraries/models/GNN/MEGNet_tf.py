#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:12:44 2021

@author: Timo Sommer

This script contains an implementation of MEGNet from the original authors based on tensorflow with an sklearn API.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph, CrystalGraphDisordered, _AtomEmbeddingMap
import json
import numpy as np
import os
import sklearn
import megnet
import tensorflow as tf
import keras.backend as K
import networkx as nx
import matplotlib.pyplot as plt
from superconductors_3D.machine_learning.own_libraries.utils.Models import Models, regressor_from_pipeline, Pickle_tf



import warnings
warnings.filterwarnings('ignore', '.*The `lr` argument is deprecated, use `learning_rate` instead.*')


def read_json_file(path_to_file):
    try:
        with open(path_to_file) as f:
            return json.load(f)
    except Exception as e:
        print(f'File {path_to_file} throws an error:')
        raise Exception(e)
    
# TODO
    # Early Stopping with grouped validation set.
    # Loss function with only positive values and with Huber Loss.

def sc_huber(ytrue, ypred, min_value=0):
    """Computes the huber loss and also sets all negative predicted y values to 0 beforehand. Useful for superconductors.
    """
    ypred_min = tf.maximum(ypred, min_value)
    loss = tf.keras.losses.huber(ytrue, ypred_min)
    return loss

# def train_test_split(X, y, sample_weight, test_size, groups):
#     """Splits data into train and test. If groups is given, a GroupShuffleSplit is used instead of a usual ShuffleSplit.
#     """
#     if groups is None:
#         CV = sklearn.model_selection.ShuffleSplit(test_size=test_size, random_state=1)   # TODO
#         train_inds, test_inds = next(CV.split(X, y, sample_weight))
#     else:
#         CV = sklearn.model_selection.GroupShuffleSplit(test_size=test_size)
#         train_inds, test_inds = next(CV.split(X, y, groups))
    
#     X_train, X_test, y_train, y_test = list(np.array(X)[train_inds]), list(np.array(X)[test_inds]), y[train_inds], y[test_inds]
        
#     return X_train, X_test, y_train, y_test



class MEGNet_tf(BaseEstimator, RegressorMixin):
    def __init__(self, use_learnt_elemental_embedding, epochs, batch_size, patience, lr, l2_coef, dropout, optimizer_kwargs, nblocks, n1, n2, n3, r_cutoff=4, n_feat_bond=100, early_stopping=False, validation_frac=None, save_checkpoint=False, prev_model=None, loss='mse', domain_col=None, tensorboard=False, lr_exp_decay=None, act='softplus2', npass=3):
        """
        Args:
        graphs_path: (str) path to a json file of all graphs
        kwargs:
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        lr: (float) learning rate
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        is_classification: (bool) whether it is a classification task
        loss: (object or str) loss function
        metrics: (list or dict) List or dictionary of Keras metrics to be evaluated by the model during training
            and testing
        dropout: (float) dropout rate
        graph_converter: (object) object that exposes a "convert" method for structure to graph conversion
        target_scaler: (object) object that exposes a "transform" and "inverse_transform" methods for transforming
            the target values
        optimizer_kwargs (dict): extra keywords for optimizer, for example clipnorm and clipvalue
        sample_weight_mode (str): sample weight mode for compilation
        kwargs (dict): in the case where bond inputs are pure distances (not the expanded distances nor integers
            for embedding, i.e., nfeat_edge=None and bond_embedding_dim=None),
            kwargs can take additional inputs for expand the distance using Gaussian basis.
            centers (np.ndarray): array for defining the Gaussian expansion centers
            width (float): width for the Gaussian basis
        """
        self.use_learnt_elemental_embedding = use_learnt_elemental_embedding
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.r_cutoff = r_cutoff
        self.n_feat_bond = n_feat_bond
        self.early_stopping = early_stopping
        self.validation_frac = validation_frac
        self.save_checkpoint = save_checkpoint
        self.prev_model = prev_model
        self.loss = loss
        self.domain_col = domain_col
        self.tensorboard = tensorboard
        self.lr = lr
        self.lr_exp_decay = lr_exp_decay
        self.l2_coef = l2_coef
        self.dropout = dropout
        self.optimizer_kwargs = optimizer_kwargs
        self.nblocks = nblocks
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.act = act
        self.npass = npass
        
    def get_loss_fn(self):
        """Returns the loss function.
        """
        if self.loss == 'mse':
            loss_function = 'mse'
        elif self.loss == 'huber':
            loss_function = 'huber'
        elif self.loss == 'sc_huber':
            loss_function = sc_huber
        else:
            raise ValueError(f'Loss function {self.loss} of MEGNet model not recognized.')
        
        return loss_function
    
    def get_act_fn(self):
        """Returns the activation function.
        """
        if self.act == 'softplus2' or self.act == 'softplus':
            act_function = megnet.activations.softplus2
        elif self.act == 'swish':
            act_function = megnet.activations.swish
        elif self.act == 'relu':
            act_function = tf.keras.activations.relu
        else:
            raise ValueError(f'Activation function {self.act} of MEGNet model not recognized.')
            
        return act_function
    
    def get_paths(self, X):
        assert X.shape[1] == 1
        paths = [os.path.expanduser(path) for path in X[:,0]]
        return paths
    
    def get_graphs(self, X):
        """Returns the graphs of each input. X must be a list of paths to these graphs.
        """        
        paths = self.get_paths(X)
        graphs = [read_json_file(path) for path in paths]
        
        # Get already learnt elemental embedding features.
        if self.use_learnt_elemental_embedding:
            print('Convert atom features to learnt embedding.')
            embedding = _AtomEmbeddingMap()
            for i in range(len(graphs)):
                graphs[i]['atom'] = embedding.convert(graphs[i]['atom'])
                
        return graphs
    
    def get_prev_model_path(self):
        """Saves only the MEGNet part of a previously trained model in a temporary file and returns this filename.
        """
        # megnet_path = Pickle_tf.get_tf_filepath(self.prev_model)
        return self.prev_model
    
    def get_feature_dimensions(self, example_graph):
        """Returns the input dimensions of the features in the correct `encoding`.
        """
        # No encoding here, just use the plain provided features.
        nfeat_state = len(example_graph['state'][0])
        
        # Get `encoding` of correct input of atom.
        Z_with_embedding = all([isinstance(feat, int) for feat in example_graph['atom']]) 
        embedding_from_disordered_dict =  all([isinstance(feat, dict) for feat in example_graph['atom']])
        if Z_with_embedding:
            print('Use atom number Z with embedding as atom features.')
            nfeat_node = None
        elif embedding_from_disordered_dict:
            print('Use already learnt embedding of 16 features per element.')
            nfeat_node = 16
        else:
            print('Use plain features as provided for the atoms.')
            nfeat_node = len(example_graph['atom'][0])
        
        # Expand distance in gaussian basis
        nfeat_edge = None
        
        return nfeat_node, nfeat_edge, nfeat_state
    
    def exp_decay_scheduler(self, epoch, lr):
        """This learning rate scheduler reduces the learning rate exponentially.
        """
        return lr * self.lr_exp_decay
    
    def init_callbacks(self):
        """Add callbacks.
        """
        callbacks = []
        
        if self.early_stopping:
            assert not self.validation_frac is None, 'Early stopping needs validation data!'
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0, 
                    patience=self.patience, 
                    verbose=0,
                    mode='min', 
                    restore_best_weights=True
                ))
            # Early stopping with monitoring anything else than val_mae only works if not saving checkpoints, MEGNet is a bit weird there.
            self.save_checkpoint = False
        
        if self.tensorboard:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir='tensorboard', 
                    write_graph=False,
                ))
        
        if not self.lr_exp_decay is None:
            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(
                    schedule=self.exp_decay_scheduler
                ))
            
        return callbacks

    def train_val_split(self, graphs, y, sample_weight, groups):
        """Splits data into train and validation data. If groups is given, this gives the groups that will be kept together in either train or validation set. This is useful e.g. for superconductors, where all datapoints with the same chemical composition should be kept together.
        """
        if not groups is None:
            raise NotImplementedError('groups is not implemented because the train_test_split with the groups can not deal with sample_weight yet.')
            
        if self.validation_frac == None:
            train_graphs = graphs
            val_graphs = None
            train_y = y
            val_y = None
            train_w = sample_weight
            val_w = None
        else:
            assert 0 < self.validation_frac < 1
            
            if sample_weight is None:
                train_graphs, val_graphs, train_y, val_y = sklearn.model_selection.train_test_split(graphs, y, test_size=self.validation_frac)#, groups=groups)
                train_w = None
                val_w = None
            else:
                train_graphs, val_graphs, train_y, val_y, train_w, val_w = sklearn.model_selection.train_test_split(graphs, y, sample_weight, test_size=self.validation_frac)#, groups=groups)
                
        return train_graphs, val_graphs, train_y, val_y, train_w, val_w
        
    def sanitize_input(self, d_train):
        """Sanitize input.
        """
        # We want either None or several groups, therefore set to None if there is only one group. This is the encoding of the ML script which is a bit unfortunate.
        if len(np.unique(d_train)) == 1:
            d_train = None
        
        if self.validation_frac == 0:
            self.validation_frac = None
        
        if self.lr_exp_decay == 1:
            self.lr_exp_decay = None
        
        if not self.lr_exp_decay is None:
            if not 0 < self.lr_exp_decay < 1:
                raise ValueError('`lr_exp_decay` must be a float between 0 and 1.')
            if self.lr_exp_decay < 0.95:
                warnings.warn('`lr_exp_decay` is very small, should usually be very close to 1.')
        
        return d_train        
        
    def fit(self, X, y, d_train=None, sample_weight=None):
        """The input X must be a list with paths to json files of graph structures.
        """
        d_train = self.sanitize_input(d_train)
        
        self.metrics = ['mse']
        
        # Load graphs
        graphs = self.get_graphs(X)
        
        # Define model
        # Dimensions
        example_graph = graphs[0]
        self.nfeat_node, self.nfeat_edge, self.nfeat_state = self.get_feature_dimensions(example_graph)
        self.n_target = y.shape[1] if y.ndim == 2 else 1
        
        # Expansion of bond features for continous representation
        gaussian_centers = np.linspace(0, self.r_cutoff + 1, self.n_feat_bond)
        gaussian_width = 0.5
        
        self.graph_converter = CrystalGraph(cutoff=self.r_cutoff)
            
        self.model = FixedMEGNetModel(
                        nfeat_node=self.nfeat_node,
                        nfeat_global=self.nfeat_state,
                        nfeat_edge=self.nfeat_edge,
                        graph_converter=self.graph_converter,
                        centers=gaussian_centers,
                        width=gaussian_width,
                        ntarget=self.n_target,
                        loss=self.get_loss_fn(),
                        metrics=self.metrics,
                        lr=self.lr,
                        l2_coef=self.l2_coef,
                        dropout=self.dropout,
                        optimizer_kwargs=self.optimizer_kwargs,
                        nblocks=self.nblocks,
                        n1=self.n1,
                        n2=self.n2,
                        n3=self.n3,
                        act=self.get_act_fn(),
                        npass=self.npass,
                        )
        
        train_graphs, val_graphs, train_y, val_y, train_w, val_w = self.train_val_split(graphs, y, sample_weight, d_train)
        
        # if not self.prev_model is None:
        #     prev_model = self.get_prev_model_path()
            
        self.callbacks = self.init_callbacks()
        model = self.model.train_from_graphs(train_graphs=train_graphs,
                                     train_targets=train_y,
                                     validation_graphs=val_graphs,
                                     validation_targets=val_y,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     patience=self.patience,
                                     callbacks=self.callbacks,
                                     save_checkpoint=self.save_checkpoint,
                                     prev_model=self.prev_model,
                                     sample_weights=train_w,
                                     val_sample_weights=val_w,
                                     )
        
        # Get loss curve
        self.loss_curve_ = {}
        self.loss_curve_['train'] = model.history.history['loss']
        if not self.validation_frac is None:
            self.loss_curve_['valid'] = model.history.history['val_loss']
        for metric in self.metrics:
            self.loss_curve_[f'{metric} (train)'] = model.history.history[metric]
            if not self.validation_frac is None:
                self.loss_curve_[f'{metric} (valid)'] = model.history.history[f'val_{metric}']
        return self
    
    def predict(self, X):
        """The input X must be a list with paths to json files of graph structures.
        """
        graphs = self.get_graphs(X)
        y_pred = self.model.predict_graphs(graphs)
        return y_pred
        








from typing import Dict, List, Union
from tensorflow.keras.callbacks import Callback
from megnet.callbacks import ModelCheckpointMAE, ManualStop, ReduceLRUponNan

class FixedMEGNetModel(MEGNetModel):
    """Added validation sample weights.
    """
    
    def train_from_graphs(
        self,
        train_graphs: List[Dict],
        train_targets: List[float],
        validation_graphs: List[Dict] = None,
        validation_targets: List[float] = None,
        sample_weights: List[float] = None,
        epochs: int = 1000,
        batch_size: int = 128,
        verbose: int = 1,
        callbacks: List[Callback] = None,
        prev_model: str = None,
        lr_scaling_factor: float = 0.5,
        patience: int = 500,
        save_checkpoint: bool = True,
        automatic_correction: bool = False,
        dirname: str = "callback",
        val_sample_weights: List[float] = None,     # Added this line
        **kwargs,
    ):
        """
        Args:
            train_graphs: (list) list of graph dictionaries
            train_targets: (list) list of target values
            validation_graphs: (list) list of graphs as validation
            validation_targets: (list) list of validation targets
            sample_weights: (list) list of sample weights
            epochs: (int) number of epochs
            batch_size: (int) training batch size
            verbose: (int) keras fit verbose, 0 no progress bar, 1 only at the epoch end and 2 every batch
            callbacks: (list) megnet or keras callback functions for training
            prev_model: (str) file name for previously saved model
            lr_scaling_factor: (float, less than 1) scale the learning rate down when nan loss encountered
            patience: (int) patience for early stopping
            save_checkpoint: (bool) whether to save checkpoint
            automatic_correction: (bool) correct nan errors
            dirname: (str) the directory in which to save checkpoints, if `save_checkpoint=True`
            **kwargs:
        """
        # load from saved model
        if prev_model:
            self.load_weights(prev_model)
        is_classification = "entropy" in str(self.model.loss)
        monitor = "val_acc" if is_classification else "val_mae"
        mode = "max" if is_classification else "min"
        has_sample_weights = sample_weights is not None
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if callbacks is None:
            # with this call back you can stop the model training by `touch STOP`
            callbacks = [ManualStop()]
        train_nb_atoms = [len(i["atom"]) for i in train_graphs]
        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(train_targets, train_nb_atoms)]
        if (validation_graphs is not None) and (validation_targets is not None):
            filepath = os.path.join(dirname, "%s_{epoch:05d}_{%s:.6f}.hdf5" % (monitor, monitor))
            val_nb_atoms = [len(i["atom"]) for i in validation_graphs]
            validation_targets = [self.target_scaler.transform(i, j) for i, j in zip(validation_targets, val_nb_atoms)]
            val_inputs = self.graph_converter.get_flat_data(validation_graphs, validation_targets)

            val_generator = self._create_generator(*val_inputs, sample_weights=val_sample_weights, batch_size=batch_size)   # Fixed this line.
            steps_per_val = int(np.ceil(len(validation_graphs) / batch_size))
            if save_checkpoint:
                callbacks.extend(
                    [
                        ModelCheckpointMAE(
                            filepath=filepath,
                            monitor=monitor,
                            mode=mode,
                            save_best_only=True,
                            save_weights_only=False,
                            val_gen=val_generator,
                            steps_per_val=steps_per_val,
                            target_scaler=self.target_scaler,
                        )
                    ]
                )
                # avoid running validation twice in an epoch
                val_generator = None  # type: ignore
                steps_per_val = None  # type: ignore

            if automatic_correction:
                callbacks.extend(
                    [
                        ReduceLRUponNan(
                            filepath=filepath,
                            monitor=monitor,
                            mode=mode,
                            factor=lr_scaling_factor,
                            patience=patience,
                            has_sample_weights=has_sample_weights,
                        )
                    ]
                )
        else:
            val_generator = None  # type: ignore
            steps_per_val = None  # type: ignore

        train_inputs = self.graph_converter.get_flat_data(train_graphs, train_targets)
        # check dimension match
        self.check_dimension(train_graphs[0])
        train_generator = self._create_generator(*train_inputs, sample_weights=sample_weights, batch_size=batch_size)
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))
        self.fit(
            train_generator,
            steps_per_epoch=steps_per_train,
            validation_data=val_generator,
            validation_steps=steps_per_val,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs,
        )
        return self
    















        