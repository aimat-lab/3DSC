#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:24:04 2021

@author: timo
This script contains my wrapper of the RGM of 2020 Jin to go with the standard sklearn API.
"""

import os
import numpy as np
from superconductors_3D.machine_learning.Algorithms.RGM_Jin import RGM as RGM_Jin
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
import warnings
from torch.utils.data import Dataset, DataLoader
from chemprop.nn_utils import initialize_weights
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
import mlflow
import time
# import torchviz
from superconductors_3D.machine_learning.own_libraries.own_functions import enforce_types


def log_metrics(metrics: dict, step: float = None, max_tries: float = 10, if_log_metrics: bool = True) -> None:
    """Logs metrics with mlflow but also checks for errors and tries again up to max_tries times if there is a connection error. Logs only if if_log_metrics = True.
    """
    if if_log_metrics:
        i = 0
        while i < max_tries:
            try:
                mlflow.log_metrics(metrics, step=step)
                i = max_tries
            except ConnectionError:
                warnings.warn('Connection Error, try again.')
                time.sleep(0.1)
                i += 1
    return()


def get_domain_batch(x_data, y_data, w_data, d_data, domain):
    """Returns the parts of x_data, y_data, w_data that are from domain as defined in d_data.
    """
    is_domain = d_data == domain
    x = x_data[is_domain]
    y = y_data[is_domain]
    d = d_data[is_domain]
    w = w_data[is_domain]
    return(x, y, d, w)


def get_all_domain_batches(x_data, y_data, w_data, d_data, all_domains):
    """Returns a list of batches where each entry in the list is a batch of a certain domain.
    """
    batches = []
    for i, domain in enumerate(all_domains):
        batch = get_domain_batch(x_data, y_data, w_data, d_data, domain)
        batches.append(batch)
    return(batches)


    
class DomainDataset(Dataset):

    def __init__(self, x, y, d, w, device):
        self.x = x
        self.y = y
        self.d = d
        self.w = w
        self.device = device
        
        # Assert same lengths
        length = len(self.x)
        assert all([length == len(self.y), length == len(self.d), length == len(self.w)]), 'At least one of features, targets, domains and weights has different length.'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        d = self.d[index]
        w = self.w[index]
        return x, y, d, w
        
    def collate_fn(self, batch):
        x, y, d, w = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)
        d = torch.stack(d)
        w = torch.stack(w)
        return x, y, d, w
    
    
    
class EarlyStopping:
    """Stops early if validation loss doesn't improve after a given patience."""
    def __init__(self, n_iter_no_change, delta=0):
        """
        Args:
            n_iter_no_change (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0    
        """
        self.n_iter_no_change = n_iter_no_change
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.valid_loss_min = np.Inf
        self.delta = delta
    def check_loss(self, valid_loss, model, epoch):

        score = -valid_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.n_iter_no_change:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(valid_loss, model, epoch)
            self.counter = 0    

    def save_checkpoint(self, valid_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        # Deepcopy is important, otherwise only reference is passed.
        self.best_model_state_dict = deepcopy(model.state_dict())
        self.valid_loss_min = valid_loss
        self.best_epoch = epoch




class RGM_sklearn(nn.Module):
    """A wrapper around the RGM implementation of 2020 Jin to be able to be called like a standard sklearn module.
    """
    @enforce_types
    def __init__(self, solver:str, max_iter: int, batch_size: int, learning_rate_init: (float, int), featurizer, classifier, batch_mode: str, weight_decay: (float, int), rgm_e: (float, int), erm_e: (float, int), holdout_e: (float, int), detach_classifier: bool, oracle: bool, ensemble_pred: bool, validation_fraction: float, early_stopping: str, n_iter_no_change: int, num_train_domains: int, max_n_classifiers: int=999999999, clip_grad: (float, int)=np.inf, if_log_metrics: bool=True, random_seed: (int, type(None))=None, coeff_lr_classifier: (float, int)=1., reduce_lr_factor: float=1., use_tensorboard: bool=True):
        super(RGM_sklearn, self).__init__()
        self.max_epochs = max_iter
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.coeff_lr_classifier = coeff_lr_classifier
        self.batch_mode = batch_mode    # Conserve_ratio'
        self.rgm_e = rgm_e
        self.erm_e = erm_e
        self.holdout_e = holdout_e
        self.detach_classifier = detach_classifier
        self.oracle = oracle
        self.seed = random_seed  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')
        self.loss_func = nn.MSELoss(reduction='none')
        self.featurizer = deepcopy(featurizer)
        self.classifier = deepcopy(classifier)
        self.solver = solver
        self.weight_decay = weight_decay
        self.ensemble_pred = ensemble_pred
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.clip_grad = clip_grad
        self.if_log_metrics = if_log_metrics
        self.num_train_domains = num_train_domains
        self.max_n_classifiers = max_n_classifiers
        self.use_tensorboard = use_tensorboard
        self.reduce_lr_factor = reduce_lr_factor
        if early_stopping != 'False':
            assert n_iter_no_change != None
        self.n_iter_no_change = n_iter_no_change
                
        self.input_args = deepcopy(locals())
        self.input_args['estimator_name'] = type(self).__name__
        del self.input_args['self']
        del self.input_args['featurizer']
        del self.input_args['classifier']
        
        self.backward_graphs = []

    
    def get_optimizer(self, solver, parameters, lr, weight_decay):
        """Returns correct optimizer based on `solver`."""
        if solver == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        elif solver == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif solver == 'adamw':
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif solver == 'rmsprop':
            optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise Warning('Solver "f{solver} not known. Solvers are lowercase always.')
        return(optimizer)
    
    
    def calc_params_norm(self, parameters, norm_type=2):
        """Calculates parameter norm as in torch.nn.utils.clip_grad. Returns torch.tensor().
        """
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return(total_norm)
    
    
    # def make_backwards_graph(self, loss, batch_idx):
    #     """"Make graph of backward pass."""
    #     if self.epoch == 0 and batch_idx == 0:
    #         backward_graph = \
    #             torchviz.make_dot(loss, dict(self.trainer.named_parameters()))
    #         self.backward_graphs.append(backward_graph)
    #     return()
    
    
    def initialize_model_weights(self, model):
        """Initializes all weights of the model."""
        model.apply(self.weight_initialization)
        return()
    
    
    def weight_initialization(self, m):
        """Initializes the layer m with a uniform xavier distribution if it is linear."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
        return()

    
    def make_consistent_torch_tensors(self, x_train, y_train, d_train, w_train):
        """Make data to torch tensors for consistency. If weights are not given give it the neutral value 1 and make it a consistent tensor.
        """
        if w_train is None:
            shape = y_train.shape
            w_train = torch.ones(shape, dtype=torch.float, device=self.device)
        else:
            w_train = torch.tensor(w_train, dtype=torch.float, device=self.device)
            
        x_train = torch.tensor(x_train, dtype=torch.float, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)    # fix sklearn inconsistency.
        y_train = torch.tensor(y_train, dtype=torch.float, device=self.device)
        
        try:
            if len(d_train) == 0:
                num_datapoints = len(x_train)
                d_train = torch.zeros(num_datapoints, dtype=torch.float, device=self.device)
            else:
                d_train = torch.tensor(d_train, dtype=torch.float, device=self.device)
        except TypeError:
            print('Internally converting string domain labels to integers.')
            labels, counts = np.unique(d_train, return_counts=True)
            sort_idc = np.argsort(counts)
            sorted_labels = labels[sort_idc]
            domain_dict = {label: idx for idx, label in enumerate(sorted_labels)}
            d_train = np.array([domain_dict[d] for d in d_train])
            d_train = torch.tensor(d_train, dtype=torch.float, device=self.device)
        
        assert all([length == len(x_train) for length in (len(y_train), len(d_train), len(w_train))])
        return(x_train, y_train, d_train, w_train)
    
    
    def get_validation_data(self, x_train, y_train, d_train, w_train):
        """Splits data in train and validation data."""
        assert isinstance(self.validation_fraction, float), 'validation_fraction must be float.'
        train_indices, valid_indices = next(StratifiedShuffleSplit(n_splits=1, test_size=self.validation_fraction).split(x_train.cpu(), d_train.cpu()))
        x_train, x_valid = x_train[train_indices], x_train[valid_indices]
        y_train, y_valid = y_train[train_indices], y_train[valid_indices]
        d_train, d_valid = d_train[train_indices], d_train[valid_indices]
        w_train, w_valid = w_train[train_indices], w_train[valid_indices]
        x_valid = x_valid.to(self.device)
        y_valid = y_valid.to(self.device)
        d_valid = d_valid.to(self.device)
        w_valid = w_valid.to(self.device)
        return(x_train, x_valid, y_train, y_valid, d_train, d_valid, w_train, w_valid)    
    

    def clip_grad_norm(self):
        """Clips the gradient norm, returns gradient norm before and after and logs them.
        """
        grad_norm_before = torch.nn.utils.clip_grad_norm_(self.trainer.parameters(), self.clip_grad).item()
        grad_norm_clipped = self.calc_params_norm(self.trainer.parameters()).item()    
        # self.writer.add_scalar('grad_norm_before', grad_norm_before)
        # self.writer.add_scalar('grad_norm_clipped', grad_norm_clipped)
        return(grad_norm_before, grad_norm_clipped)
    
    def weighted_average(self, y, weights):
        y = y * weights
        norm = weights.sum()
        return y.sum() / norm
        
    def loss_forward(self, y_pred, y_true, weights):
        # TODO
        assert (y_pred >= 0).all()
        pred_losses = self.loss_func(y_pred, y_true)
        pred_loss = self.weighted_average(pred_losses, weights)
        return pred_loss
        
    
    def get_domain_data(self, x_data, y_data, w_data, d_data, domain):
        """Returns the dataset for a single domain.
        
        x_data, y_data, w_data, d_data are the datasets with all domains together. domain is the domain label.
        """
        x, y, d, w = get_domain_batch(x_data, y_data, w_data, d_data, domain)
        data = DomainDataset(x, y, d, w, self.device)
        return(data)
    
    
    def determine_batch_ratios(self, mode):
        """This function determines the batch sizes per domain ('batch_ratios') based on `mode`.
        """
        if mode == 'Conserve_ratio':
        # Each domain will have a batch size according to its ratio in the total dataset.
            ratios = self.all_domain_counts / sum(self.all_domain_counts)
            batch_ratios = self.batch_size * ratios
            batch_ratios = torch.round(batch_ratios.float()).int()
            drop_last = True
        else:
            raise ValueError(f'Unknown batch_mode "{mode}".')
        if any(batch_ratios == 0):
            warnings.warn('One of the domains has very few examples and should have zero examples per batch acccording to batch_size. Setting minimum number of examples to 1.')
            min_batch_ratios = torch.ones_like(batch_ratios)
            batch_ratios = torch.maximum(batch_ratios, min_batch_ratios)
        elif any(batch_ratios > self.all_domain_counts):
            warnings.warn('For at least one domain the batchsize is greater than the number of samples, unpredictable behaviour.')
        return(batch_ratios, drop_last)
        
    
    def get_batches(self, x_data, y_data, d_data, w_data, mode):
        """Generates a list of batches for each domain.
        """
        batch_ratios, drop_last = self.determine_batch_ratios(mode)
        # Get data loaders for each domain that return minibatches scaled like the ratio in the total dataset. 
        data_loaders = {}
        for domain, batch_ratio in zip(self.all_domains, batch_ratios):
            batch_ratio = batch_ratio.item()
            domain = domain.item()
            data = self.get_domain_data(x_data, y_data, w_data, d_data, domain)
            data_loaders[domain] = iter(DataLoader(data, batch_size=batch_ratio, drop_last=drop_last, shuffle=True, collate_fn=data.collate_fn))
        
        # Go through batches of each domain. If a domain has no examples left in the dataset, resample it and continue until all domains are through at least once.
        num_batches = [len(loader) for loader in data_loaders.values()]
        # print('The number of batches per domain is', num_batches)
        num_iters = max(num_batches)
        for batch_idx in range(num_iters):
            batches = []
            for domain, batch_ratio in zip(self.all_domains, batch_ratios):
                batch_ratio = batch_ratio.item()
                domain = domain.item()
                d_loader = data_loaders[domain]
                d_batch = next(d_loader, None)
                # If one domain has no data points left reload data points.
                if d_batch == None:
                    assert self.num_domains > 1
                    data = self.get_domain_data(x_data, y_data, w_data, d_data, domain)
                    d_loader = iter(DataLoader(data, batch_size=batch_ratio, drop_last=drop_last, shuffle=True, collate_fn=data.collate_fn))
                    d_batch = next(d_loader, None)
                    data_loaders[domain] = d_loader
                    assert not d_batch == None
                batches.append(d_batch)
            yield(batches)
         
            
    def prepare_trainer(self):
        """Builds up the Neural Network trainer with all it's layers, it's optimizer etc.
        """
        
        trainer = RGM_Jin(featurizer=self.featurizer,
                          classifier=self.classifier,
                          num_domains=self.num_domains,
                          rgm_e=self.rgm_e,
                          erm_e=self.erm_e,
                          holdout_e=self.holdout_e,
                          detach_classifier=self.detach_classifier,
                          oracle=self.oracle,
                          loss_forward=self.loss_forward,
                          num_train_domains=self.num_train_domains,
                          max_n_classifiers=self.max_n_classifiers
                          )
        # Initialize weights new. Very important because otherwise all classifiers have the same initialization!
        initialize_weights(trainer)
        # self.initialize_model_weights(trainer)
        trainer = trainer.to(self.device)
        
        # Get parameter lists for classifier and representation separately for different learning rates.
        classifier_params = []
        for classifier in (trainer.classifier, trainer.f_k, trainer.g_k):
            classifier_params.extend(list(classifier.parameters()))
        assert len(list(trainer.parameters())) == len(classifier_params) + len(list(trainer.copy_f_k.parameters())) + len(list(trainer.featurizer.parameters()))
        learning_rate_classifier = self.coeff_lr_classifier * self.learning_rate_init
        parameters = [
                      {'params': trainer.featurizer.parameters()},
                      {'params': classifier_params, 'lr': learning_rate_classifier}
                      ]
        
        trainer.optimizer = self.get_optimizer(solver=self.solver,
                                               parameters=parameters,
                                               lr=self.learning_rate_init,
                                               weight_decay=self.weight_decay
                                               )
        trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=self.reduce_lr_factor)
        return(trainer)
    
    
    def validate(self, x_train, x_valid, y_train, y_valid, w_train, w_valid):
        """Calculates train and validation loss."""
        y_train_pred = self.evaluate(x_train)
        y_valid_pred = self.evaluate(x_valid)
        
        train_loss = self.loss_forward(y_pred=y_train_pred, y_true=y_train, weights=w_train).item()
        valid_loss = self.loss_forward(y_pred=y_valid_pred, y_true=y_valid, weights=w_valid).item()
        return(train_loss, valid_loss)
        
    
    def train_epoch(self, x_train, y_train, d_train, w_train):     # d_train: domain labels
        "Train one epoch."
        self.trainer.train()
        
        all_batches = self.get_batches(x_train, y_train, d_train, w_train, mode=self.batch_mode)
        # Record all losses and the gradient.
        losses = {sub_loss: 0 for sub_loss in self.trainer.loss_curve_iter.keys()}
        losses['grad_norm_before'] = 0
        losses['grad_norm_clipped'] = 0
        for batch_idx, d_batches in enumerate(all_batches):          
            self.trainer.optimizer.zero_grad()        
            loss = self.trainer(d_batches)        
            # self.make_backwards_graph(loss, batch_idx)
            loss.backward()
            grad_norm_before, grad_norm_clipped = self.clip_grad_norm()            
            self.trainer.optimizer.step()
            
            for sub_loss in self.trainer.loss_curve_iter.keys():
                losses[sub_loss] += self.trainer.loss_curve_iter[sub_loss][-1]
            losses['grad_norm_before'] += grad_norm_before
            losses['grad_norm_clipped'] += grad_norm_clipped
            
        for sub_loss, loss_value in losses.items():
            self.loss_curve_[sub_loss].append(loss_value)
                
            
    def train_until_stop(self, x_train, y_train, d_train, w_train, x_valid, y_valid, w_valid):
        """Trains the model until either all epochs are trained or early_stopping criterion is reached. Label is a string to differentiate the run to train the representation and the run to train the classifier if these are seperated.
        """
        if self.early_stopping != 'False':
            stop = EarlyStopping(self.n_iter_no_change)
        
        self.loss_curve_['train'] = []
        self.loss_curve_['valid'] = []       
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.train_epoch(x_train, y_train, d_train, w_train)
            
            # Validation on validation set.
            train_loss, valid_loss = self.validate(x_train, x_valid, y_train, y_valid, w_train, w_valid)
            self.loss_curve_['train'].append(train_loss)
            self.loss_curve_['valid'].append(valid_loss)
            # metrics = {'train_loss': train_loss, 'valid_loss': valid_loss}
            # log_metrics(metrics, step=epoch, if_log_metrics=self.if_log_metrics)
            # self.writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, global_step=epoch)
            
            self.trainer.scheduler.step()
            
            # Early stopping, but only after min 20% of epochs.
            if self.early_stopping != 'False' and epoch > self.max_epochs / 5:
                valid_score = self.loss_curve_[self.early_stopping][-1]
                stop.check_loss(valid_score, self.trainer, epoch)
                if stop.early_stop == True:
                    print(f'Stopped early in epoch {epoch}.')
                    break
        
        if self.early_stopping != 'False' and epoch > self.max_epochs / 5:
            self.trainer.load_state_dict(stop.best_model_state_dict)
            self.best_epoch = stop.best_epoch
            print(f'Best model loaded from epoch {self.best_epoch}.')
        return()
    
    
    def fit(self, x_train, y_train, d_train=[], sample_weight=[]):
        """Train the model for all epochs.
        """
        if self.seed != None:
            warnings.warn('torch.manual_seed and random.seed behave a bit dangerous, different than np.random_seed. If the torch seed is set it not only is set for afterwards in this run, but it is set even if the variables are cleaned! Only a kernel restart also unsets the torch.manual_seed. I don\'t know if random.seed also persists after cleaning variables but I think so and it definitely persists across modules.')
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        x_train, y_train, d_train, w_train = self.make_consistent_torch_tensors(x_train, y_train, d_train, sample_weight)
                
        # Get validation data.
        if not self.validation_fraction == None:
            x_train, x_valid, y_train, y_valid, d_train, d_valid, w_train, w_valid = \
                self.get_validation_data(x_train, y_train, d_train, w_train)
        else:
            raise Warning('Code currently asserts validation set.')
        
        # Initialize variables.
        self.num_domains = len(torch.unique(d_train))
        self.output_size = y_train.shape[1] if len(y_train.shape) == 2 else 1
        self.n_train_samples = len(y_train)
        if self.batch_size > len(x_train):
            warnings.warn('Got batch size larger than sample size. It is going to be clipped.')
            self.batch_size = len(x_train)
        self.all_domains, self.all_domain_counts = torch.unique(d_train, return_counts=True)
        self.use_classifier = False if self.ensemble_pred else True
        
        self.trainer = self.prepare_trainer()
        
        self.loss_curve_ = {sub_loss: [] for sub_loss in self.trainer.loss_curve_iter.keys()}
        self.loss_curve_['grad_norm_before'] = []
        self.loss_curve_['grad_norm_clipped'] = []
        
        self.train_until_stop(x_train, y_train, d_train, w_train, x_valid, y_valid, w_valid)
            
        # self.trainer.log_losses = False
        # batches_data = get_all_domain_batches(x_train, y_train, w_train, d_train, self.all_domains)
        # self.writer.add_graph(self.trainer, [batches_data])
        # self.writer.add_embedding(x_train, y_train, tag='data')
        # self.writer.flush()
        # torch.onnx.export(self.trainer, (batches_data,), self.outpath, export_params=True, opset_version=11, input_names=['input'], output_names=['output'], example_outputs=torch.tensor(1))
        # torch.save(self.trainer, self.outpath)
        
        # for loss_name, loss_curve in self.trainer.loss_curve_iter.items():
        #     self.loss_curve_[loss_name] = loss_curve
        
        # Push model to cpu after training. Could be useful sometimes but not really needed.
        self = self.to('cpu')
        self.device = 'cpu'
    
    
    def evaluate(self, x_test):
        """Forward evaluates the network. Can either use the classifier or an ensemble of all extrapolator classifiers.
        """
        if self.use_classifier:
            model = self.trainer.network
            model.eval()
            with torch.no_grad():
                y_pred = model(x_test)
        else:
            shape = (self.trainer.n_classifiers, len(x_test), self.output_size)
            all_y_pred = torch.zeros(shape, device=self.device)
            for k, model in enumerate(self.trainer.ensemble):
                model.eval()
                with torch.no_grad():
                    all_y_pred[k, :, :] = model(x_test)
            y_pred = torch.quantile(all_y_pred, 0.5, dim=0)   # median
        return(y_pred)
        
    
    def predict(self, x_test):
        """Infers data points and outputs a numpy array with the same dtype as the input array.
        """
        input_dtype = x_test.dtype
        
        x_test = torch.from_numpy(x_test).float().to(self.device)
        self.use_classifier = True if not self.ensemble_pred else False
        y_pred = self.evaluate(x_test)

        y_pred = np.array(y_pred.cpu(), dtype=input_dtype)
        return(y_pred)
        
        
        
        
        
        