#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:13:25 2021

@author: Timo Sommer

This script contains an implementation of a Gaussian Process that works with the GPflow models.
"""
import torch
from sklearn.base import RegressorMixin, BaseEstimator
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gpflow
import pickle
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import os
import warnings
from gpflow.utilities import print_summary, set_trainable
from gpflow.optimizers import NaturalGradient



class GPflow_GP(RegressorMixin, BaseEstimator):
    
    def __init__(self, model, kernel, alpha, mean_function=None, batch_size=100, epochs=1000, learning_rate=0.1, n_inducing_points=100, NN_path=None, standard_scale=False, natgrad=False, diff_std_for_sc_and_non_sc=False, train_noise=True, train_noise_scale=False, predict_y=False):
        """alpha: Estimated std of data. float for constant noise, array of floats for these noises as absolute std. Pass function to compute noise std from input y. 
        `natgrad`: If the variational parameters should be trained using Natural Gradients instead of Adam.
        """
        self.model = model
        self.kernel = kernel
        self.alpha = alpha
        self.mean_function = mean_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_inducing_points = n_inducing_points
        self.NN_path = NN_path
        self.standard_scale = standard_scale
        self.natgrad = natgrad
        self.diff_std_for_sc_and_non_sc = diff_std_for_sc_and_non_sc
        self.train_noise = train_noise
        self.train_noise_scale = train_noise_scale
        self.predict_y = predict_y
    
    def get_array_noise_var(self, alpha, y):
        try:
            # If alpha is function.
            noise_std = alpha(y)
            print('Using specified function for noise.')
        except TypeError:
            try:
                # If alpha is float.
                alpha = float(alpha)
                noise_std = np.full(y.shape, alpha)
                print('Using single float for noise, but in likelihood.')
            except TypeError:
                # If alpha is array.
                noise_std = np.asarray(alpha)
                print('Using specified array for alpha')
        noise_std = noise_std.reshape(y.shape)
        noise_var = noise_std**2
        return noise_var
    
    def get_noise_var(self, y):
        """Implement variance per data point if model allows it, otherwise set the same variance per data point.
        """
        if self.model in self.homogeneous_noise_models:
            
            # GPflow model only takes a single float for noise as `noise_var`.
            try:
                noise_variance = {'noise_variance': float(self.alpha)**2}
                print('Using single float as noise.')
            except TypeError:
                raise ValueError('Parameter `alpha` must be float for this model.')
                
        else:            
            if y.shape[1] == 1:
                
                if self.diff_std_for_sc_and_non_sc:
                    # Allow different variances for all data points with Tc=0 and all other data points.
                    print('Using different noise for Tc=0 and Tc>0.')
                    likelihood = gpflow.likelihoods.SwitchedLikelihood(
                        [gpflow.likelihoods.Gaussian(variance=self.alpha**2),
                          gpflow.likelihoods.Gaussian(variance=self.alpha**2)]
                        )
                    non_sc_tc = min(y)
                    groups = np.where(y > non_sc_tc, 0, 1)
                    y = np.hstack([y, groups])
                    
                elif self.train_noise_scale:
                    # Train a separarate GP to learn the scale of the noise.
                    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
                        distribution_class=tfp.distributions.Normal,
                        scale_transform=tfp.bijectors.Softplus()
                        )
                    self.kernel = gpflow.kernels.SeparateIndependent([
                                        self.kernel,  # This is k1, the kernel of f1
                                        self.kernel,  # this is k2, the kernel of f2
                                        ])
                    
                else:                    
                    
                    try:
                        # If alpha is float.
                        print('Using single float as noise.')
                        likelihood = gpflow.likelihoods.Gaussian(variance=float(self.alpha)**2)
                        
                    except TypeError:    
                        # If alpha is iterable or function.
                        print('Alpha is iterable or function.')
                        noise_var = self.get_array_noise_var(self.alpha, y)
                        likelihood = HeteroskedasticGaussian()
                        y = np.hstack([y, noise_var])
                
                noise_variance = {'likelihood': likelihood}
            else:
                raise NotImplementedError('Expression above not yet working with multiple y (see num_latent_GPs of HeteroskedasticGaussian()).')
            
        return noise_variance, y
    
    def get_inducing_variable(self, X):
        """Choose `M` inducing points for sparse model from data `X`.
        """
        if self.model in self.sparse_models:
            assert not (self.n_inducing_points is None)
            M = self.n_inducing_points
            Z = X[:M, :].copy()
            if self.train_noise_scale:
                Z = gpflow.inducing_variables.SeparateIndependentInducingVariables(
                            [
                                gpflow.inducing_variables.InducingPoints(Z),
                                gpflow.inducing_variables.InducingPoints(Z),
                            ]
                            )
            inducing_variable = {'inducing_variable': Z}
        else:
            inducing_variable = {}
        return inducing_variable
    
    def run_adam(self, X, y):
        """Utility function running the Adam optimizer.
        """
        assert not (self.batch_size is None) and not (self.epochs is None)
        # Commented out because seems wrong though it was in tutorial.
        # try:
        #     gpflow.set_trainable(self.gp.inducing_variable, False)            
        # except AttributeError:
        #     pass
    
        # Train the variational parameters using natural gradients.
        if self.natgrad:
            set_trainable(self.gp.q_mu, False)
            set_trainable(self.gp.q_sqrt, False)
            natgrad = NaturalGradient(gamma=1)
            variational_params = [(self.gp.q_mu, self.gp.q_sqrt)]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).repeat().shuffle(len(X))            
        train_iter = iter(train_dataset.batch(self.batch_size))
            
        # Create an Adam Optimizer action
        training_loss = self.gp.training_loss_closure(train_iter, compile=True)
        adam = tf.optimizers.Adam(self.learning_rate)

    
        @tf.function
        def optimization_step():
            adam.minimize(training_loss, self.gp.trainable_variables)
            if self.natgrad:
                natgrad.minimize(training_loss, var_list=variational_params)
        
        self.loss_curve_ = []
        for step in range(self.epochs):
            optimization_step()
            self.loss_curve_.append(training_loss().numpy())
            
        return 
        
    def transform_X_with_NN(self, X):
        """Load pytorch NN and use it to transform X.
        """
        if not hasattr(self, 'NN_featurizer'):            
            if os.path.exists(self.NN_path):            
                with open(self.NN_path, 'rb') as f:
                    NN = pickle.load(f)
                    self.NN_featurizer = NN.regressor['model'].featurizer            
                    
            else:
                warnings.warn('Model {self.NN_path} not found. Continue training on original input features.')                    
                return(X)
            
            print(f'Training GP on output of {self.NN_path}.')
            
        else:
            print(f'Evaluating GP on output of {self.NN_path}.')
        
        X = deepcopy(X)
        X = torch.tensor(X, dtype=torch.float)
        X = self.NN_featurizer(X)
        X = X.cpu().detach().numpy()
        X = np.float64(X)   # As it was before
        
        return X
    
    def inverse_transform_std(self, mu, std, scaler):
        """Makes the inverse transform of the std by transforming upper and lower bound. Returns upper and lower bound after the inverse transform.
        """
        lower_conf = mu - std
        upper_conf = mu + std
        lower_conf_trans = scaler.inverse_transform(lower_conf)
        upper_conf_trans = scaler.inverse_transform(upper_conf)
        mu_trans = scaler.inverse_transform(mu)
       
        assert np.allclose(np.mean([lower_conf_trans, upper_conf_trans], axis=0), mu_trans), 'Std bounds not symmetrical.'
        std_trans = (upper_conf_trans - lower_conf_trans) / 2
        
        return std_trans
        
    def fit(self, X, y):
        
        self.homogeneous_noise_models = [gpflow.models.GPR, gpflow.models.SGPR]
        self.sparse_models = [gpflow.models.SGPR, gpflow.models.SVGP]
        self.variational_models = [gpflow.models.VGP, gpflow.models.SVGP]
        
        # GPflow needs 2D input for y, otherwise bugs happen.
        X = deepcopy(np.asarray(X))
        y = deepcopy(np.asarray(y))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]
        X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)
        
        if self.NN_path != None:
            X = self.transform_X_with_NN(X)
            self.fitted_on_NN = True
        else:
            self.fitted_on_NN = False
        
        # Standard scale X and y to have mean 0 and std 1.
        if self.standard_scale:
            self.X_scaler, self.y_scaler = StandardScaler(), StandardScaler()
            X = self.X_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y)
            
        noise_variance, y = self.get_noise_var(y)
        
        if self.model != gpflow.models.SVGP:
            data = {'data': (X, y)}
        else:
            data = {'num_data': len(X)}
        
        inducing_variable = self.get_inducing_variable(X)
        
        if self.train_noise_scale:
            num_latent_GPs = noise_variance['likelihood'].latent_dim
        else:
            num_latent_GPs = n_targets
        print(f'Number of latent GPs: {num_latent_GPs}')

        self.gp = self.model(**data,
                           kernel=self.kernel,
                           mean_function=self.mean_function,
                           num_latent_gps=num_latent_GPs,
                           **noise_variance,
                           **inducing_variable
                           )
        
        if not self.train_noise:
            # Keep specified noise fixed.
            try:
                set_trainable(self.gp.likelihood.variance, False)
                # set_trainable(self.gp.kernel.kernels[1].variance, False)
            except AttributeError:
                print('Noise can not be trained.')
        
        # print_summary(self.gp)
        
        # Make fit.
        if not self.model in self.variational_models:
            opt = gpflow.optimizers.Scipy()
            opt.minimize(self.gp.training_loss, self.gp.trainable_variables)            
        else:
            self.run_adam(X, y)
        
        # print_summary(self.gp)
            
        return(self)
    
    def predict(self, X, return_std=False):
        X = np.asarray(X)
        
        if self.fitted_on_NN:
            X = self.transform_X_with_NN(X)
        
        if self.standard_scale:
            X = self.X_scaler.transform(X)
        
        if self.predict_y:
            y, var = self.gp.predict_y(X)
        else:
            y, var = self.gp.predict_f(X)
        y, var = np.asarray(y), np.asarray(var)
        std = np.sqrt(var)
        
        # if self.diff_std_for_sc_and_non_sc:
        #     y = y[:,:-1]    # Remove group column
        #     max_uncertain_tc = std[0,1]
        #     std = np.where(y.squeeze() > max_uncertain_tc, std[:,0], std[:,1])
        #     std = std.reshape(y.shape)
        
        # Transform mean and std back.
        if self.standard_scale:
            std = self.inverse_transform_std(y, std, self.y_scaler)
            y = self.y_scaler.inverse_transform(y)
        
        if return_std:
            return(y, std)
        else:
            return(y)
                
    
class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError