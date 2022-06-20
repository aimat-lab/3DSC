import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import TracerWarning
import math
import numpy as np
import random
from copy import deepcopy
import warnings


class GradientReversal(torch.autograd.Function):
    beta = 1.

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return -GradientReversal.beta * grad_output


class RGM(nn.Module):

    def __init__(self, featurizer, classifier, num_domains, rgm_e, erm_e, holdout_e, detach_classifier, oracle, loss_forward, num_train_domains):
        super(RGM, self).__init__()
        # Deepcopy to not influence the original featurizer and classifier.
        self.featurizer = deepcopy(featurizer)
        self.classifier = deepcopy(classifier)
        
        self.copy_f_k = nn.ModuleList(deepcopy(classifier) for _ in range(num_domains))
        for copy_f_k in self.copy_f_k:
            copy_f_k.requires_grad_(False)
        # extrapolator f_{-e}
        self.f_k = nn.ModuleList(deepcopy(classifier) for _ in range(num_domains))
        # oracle f_{e}
        self.g_k = nn.ModuleList(deepcopy(classifier) for _ in range(num_domains))

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.init_network = deepcopy(self.network)
        self.ensemble = nn.ModuleList([nn.Sequential(self.featurizer, self.f_k[k]) for k in range(num_domains)])
        
        self.rgm_e = rgm_e
        self.erm_e = erm_e
        self.holdout_e = holdout_e
        self.detach_classifier = detach_classifier
        self.oracle = oracle
        self.num_domains = num_domains
        if num_train_domains == None:
            self.num_train_domains = num_domains - 1
        else:
            self.num_train_domains = num_train_domains
        if self.num_train_domains > num_domains:
            raise ValueError('num_train_domains must be <= num_domains.')
        self.train_representation = True
        self.train_classifier = True
        self.loss_forward = loss_forward
        self.register_buffer('update_count', torch.tensor([0]))
        
        self.log_losses = True
        self.loss_curve_iter = {}
        self.loss_curve_iter[f'extrapol'] = []
        self.loss_curve_iter[f'oracle'] = []
        self.loss_curve_iter['erm'] = []
        self.loss_curve_iter['holdout'] = []
        self.loss_curve_iter['regret'] = []
    
    
    def get_train_and_extrapolation_domains(self):
        """Returns all train and extrapolation domain indices (second dimension) for all auxiliary classifiers (first dimension).
        """
        if self.num_train_domains == self.num_domains - 1:
            all_train_domains = [[i for i in range(self.num_domains) if i != k] for k in range(self.num_domains)]
            all_extra_domains = [[k] for k in range(self.num_domains)]
        elif self.num_train_domains == 1:
            all_train_domains = [[k] for k in range(self.num_domains)]
            all_extra_domains = [[i for i in range(self.num_domains) if i != k] for k in range(self.num_domains)]
        else:
            raise Warning('num_train_domains other than 1 or num_domains not implemented yet.')
        return(all_train_domains, all_extra_domains)


    def forward(self, batches):      
        warnings.filterwarnings("ignore", category=TracerWarning)
        
        # Check that each batch in batches is from one domain.
        for _, _, batch_d, _ in batches:
            assert len(torch.unique(batch_d)) == 1
        assert len(np.unique([torch.unique(batch_d).item() for _, _, batch_d, _ in batches])) == self.num_domains
        
        for k in range(self.num_domains):
            self.copy_f_k[k].load_state_dict(self.f_k[k].state_dict())
        
        # Evaluate representation
        all_phis = []
        for batch_x, _, _, _ in batches:
            phi_x = self.featurizer(batch_x)
            all_phis.append(phi_x)
        
        # Compute L(f◦phi)
        erm_loss = torch.tensor(0)
        if self.train_classifier:
            for k, (_, batch_y, batch_d, mask) in enumerate(batches):
                phi_x = all_phis[k]
                if self.detach_classifier:
                    phi_x = phi_x.detach()
                preds = self.classifier(phi_x) 
                erm_loss = erm_loss + self.loss_forward(preds, batch_y, mask)
        
        # Set train and extrapolation domain indices for each of the auxiliary classifiers.
        all_train_domains, all_extra_domains = self.get_train_and_extrapolation_domains()
            
        # Compute regret R^e(phi). 
        extra_loss = torch.tensor(0)
        oracle_loss = torch.tensor(0)
        if self.num_domains > 1 and self.train_representation:
            for k, extra_domains in enumerate(all_extra_domains):   # Loop classifiers

                for j in extra_domains:                             # Loop domains
                    _, batch_y, batch_d, mask = batches[j]
                    phi_k = all_phis[j]
                    preds = self.copy_f_k[k](phi_k)     # f_{-e}◦phi (extrapolator)
                    extra_loss = extra_loss + self.loss_forward(preds, batch_y, mask)
                    # The minus in the loss function in the paper 2020 Jin here is introduced by the GradientReversal Layer.
                    if self.oracle:
                        oracle_preds = self.g_k[k](GradientReversal.apply(phi_k)) # f_{e}◦phi
                        oracle_loss = \
                            oracle_loss + self.loss_forward(oracle_preds, batch_y, mask)                    
        regret = extra_loss + oracle_loss

        
        
        # Compute L^{-e}(f_{-e}◦phi). Only for training f_{-e}, gradient is not backpropagated on phi (detached).
        holdout_loss = torch.tensor(0)
        if self.num_domains > 1 and self.train_representation:
            for k, train_domains in enumerate(all_train_domains):   # Loop classifiers
                # Train the kth classifier on all train_domains.
                for j in train_domains:                             # Loop domains
                    _, batch_y, batch_d, mask = batches[j]
                    phi_x = all_phis[j].detach()  # phi does not help f_{-e}
                    preds = self.f_k[k](phi_x)
                    norm = len(train_domains)
                    holdout_loss = holdout_loss + self.loss_forward(preds, batch_y, mask) / norm
        

        loss = (self.erm_e * erm_loss + self.rgm_e * (self.holdout_e * holdout_loss + regret)) / self.num_domains
        
        if self.train_representation and self.log_losses:
            self.loss_curve_iter[f'extrapol'].append(extra_loss.item())
            self.loss_curve_iter[f'oracle'].append(oracle_loss.item())
            self.loss_curve_iter['regret'].append(regret.item())
            self.loss_curve_iter['holdout'].append(holdout_loss.item())
        if self.train_classifier and self.log_losses:
            self.loss_curve_iter['erm'].append(erm_loss.item())

        return loss 
