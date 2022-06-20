#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:36:50 2021

@author: Timo Sommer

This script includes a standard Neural Network based on pytorch lightning.
"""
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

def get_activation_fn(activation: str):
    """Returns torch activation function based on string activation.
    """
    if activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'logistic':
        activation_fn = nn.Sigmoid()
    elif activation == 'tanh':
        activation_fn = nn.Tanh()
    else:
        raise ValueError(f'Activation function {activation} not recognized. Activation functions are lowercase always.')
    return(activation_fn)

def get_sequential_NN(input_layer_size: int, hidden_layer_sizes: list, activation: str):
    """Returns a sequential (Feed Forward) NN. `last_linear` means if the last layer should be linear or with activation function.
    """        
    activation_fn = get_activation_fn(activation)
    layers = []
    num_layers = len(hidden_layer_sizes)
    for i in range(num_layers):
        if i == 0:
            in_size = input_layer_size
        else:
            in_size = hidden_layer_sizes[i-1]
        out_size = hidden_layer_sizes[i]
        layers.append(nn.Linear(in_size, out_size))
        
        last_layer = i == num_layers - 1
        if not last_layer:
            layers.append(activation_fn)
            
    layers = tuple(layers)
    network = nn.Sequential(*layers)
    return(network)


class DataModule(LightningDataModule):
    def __init__(self, X, y, batch_size, validation_fraction):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        
    def prepare_data(self):
        # called only on 1 GPU
        return

    def setup(self, stage):
        # called on every GPU
        X_train, X_val, y_train, y_val = train_test_split(
                                        self.X,
                                        self.y,
                                        test_size=self.validation_fraction
                                        )
        self.train_data = TrainDataset(X_train, y_train)
        self.val_data = TrainDataset(X_val, y_val)


    def train_dataloader(self):
        train_dataloader = DataLoader(
                                        self.train_data,
                                        batch_size=self.batch_size,
                                        shuffle=True
                                      )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
                                        self.val_data,
                                        batch_size=self.batch_size
                                        )
        return val_dataloader

    
    

class TrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


class LightningMLP(LightningModule):
    def __init__(self,
                 solver,
                 learning_rate,
                 alpha,
                 n_features,
                 n_targets,
                 hidden_layer_sizes,
                 activation
                 ):
        super().__init__()
        self.solver = solver
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.loss_func = self.get_loss_func()
        self.loss_curve_ = {'train': [], 'valid': []}
        
        self.make_architecture()
        return
    
    def configure_optimizers(self):
        if self.solver == 'adam':
            optimizer = torch.optim.Adam(
                                            self.parameters(),
                                            lr=self.learning_rate,
                                            weight_decay=self.alpha
                                    )
        else:
            raise NotImplementedError(f'Optimizer {self.solver} not found.')
        return optimizer
    
    def make_architecture(self):
        """Generates `self.network`.
        """
        layer_sizes = list(self.hidden_layer_sizes) + [self.n_targets]
        self.network = get_sequential_NN(
                                        input_layer_size=self.n_features,
                                        hidden_layer_sizes=layer_sizes,
                                        activation=self.activation
                                        )
        return

    def get_loss_func(self):
        loss_func = nn.MSELoss()
        return loss_func
    
    def get_loss(self, y_pred, y_true):
        loss = self.loss_func(y_pred, y_true)
        return loss
    
    def forward(self, X):
        """Used for inference.
        """
        y_pred = self.network(X)

        return y_pred
    
    def training_step(self, batch, batch_idx):
        """Used for training.
        """
        X, y_true = batch
        y_pred = self(X)
        loss = self.get_loss(y_pred, y_true)
        
        # log
        self.log('train_loss', loss, on_epoch=True)
        self.loss_curve_['train'].append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Used for validation.
        """
        X, y_true = batch
        y_pred = self(X)
        loss = self.get_loss(y_pred, y_true)
        
        # log
        self.log("val_loss", loss)
        self.loss_curve_['valid'].append(loss.item())
        
        return loss
    
    
    
    
class MLP():
    def __init__(self, 
                 hidden_layer_sizes: tuple=(100),
                 activation: str='relu',
                 solver: str='adam',
                 n_epochs: int=200,
                 batch_size: int=200,
                 learning_rate: float=1e-3,
                 alpha: float=1e-4,
                 validation_fraction: (int, type(None))=None,
                 patience: int=10,
                 # clip_grad: float=np.inf,
                 # log_metrics: bool=True,
                 # random_seed: (int, type(None))=None,
                 # use_tensorboard: bool=True
                 **kwargs
                 ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.kwargs = kwargs
    
    def standardize_input(self, X, y):
        """Standardizes input parameters.
        """
        # Module parameters
        self.n_epochs = int(self.n_epochs)
        self.batch_size = int(self.batch_size)
        self.learning_rate = float(self.learning_rate)
        self.alpha = float(self.alpha)
        if self.validation_fraction is not None:
            self.early_stopping = True
            self.validation_fraction = float(self.validation_fraction)
            assert (0 < self.validation_fraction < 1)
        else:
            self.early_stopping = False
            self.validation_fraction = 0
        self.patience = int(self.patience)
        
        # Input arrrays
        self.n_features = X.shape[1]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.targets_1D = True
        else:
            self.targets_1D = False
        self.n_targets = y.shape[1]
        
        return X, y
    
    def configure_callbacks(self):
        """Configure all callbacks here.
        """
        callbacks = []
        
        # Early stopping
        if self.early_stopping:
            early_stopping = EarlyStopping(
                                            monitor="val_loss", 
                                            patience=self.patience
                                            )
            callbacks.append(early_stopping)
        
        return callbacks
        
    
    def fit(self, X, y):
        # TODO
            # Add repeatibility (https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html)
            
        X, y = self.standardize_input(X, y)
        self.callbacks = self.configure_callbacks()
        
        data = DataModule(X, y, self.batch_size, self.validation_fraction)
        model = LightningMLP(
                                    solver=self.solver,
                                    learning_rate=self.learning_rate,
                                    alpha=self.alpha,
                                    n_features = self.n_features,
                                    n_targets=self.n_targets,
                                    hidden_layer_sizes=self.hidden_layer_sizes,
                                    activation=self.activation
                                    )
        
        self.trainer = pl.Trainer(
                                    max_epochs=self.n_epochs,
                                    callbacks=self.callbacks,
                                    enable_progress_bar=False,
                                    logger=False,
                                    **self.kwargs
                                    )
        self.trainer.fit(model=model, datamodule=data)
        
        self.loss_curve_ = self.trainer.model.loss_curve_
        
        return        
        
    def convert_predictions_to_numpy(self, y_pred):
        """Sanitizes the output of `self.trainer.predict` to make a numpy array with the same number of dimensions as the input targets.
        """
        y_pred = np.array(y_pred[0].tolist()).reshape(-1, self.n_targets)
        if self.targets_1D:
            y_pred = y_pred.ravel()
            
        return y_pred
        
    def predict(self, X):
        
        dataset = TestDataset(X)
        dataloader = DataLoader(dataset, batch_size=len(X))
        # model_callback
        y_pred = self.trainer.predict(
                                        model=self.trainer.model,
                                        dataloaders=dataloader,
                                        ckpt_path='best'
                                        )
        y_pred = self.convert_predictions_to_numpy(y_pred)
        
        return y_pred
        
        
        
        
        

        
        
    
    
        
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        