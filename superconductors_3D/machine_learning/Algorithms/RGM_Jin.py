import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import TracerWarning
import math
import numpy as np
import random
from copy import deepcopy
import warnings
from itertools import combinations


class GradientReversal(torch.autograd.Function):
    beta = 1.

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return -GradientReversal.beta * grad_output


class RGM(nn.Module):

    def __init__(self, featurizer, classifier, num_domains, rgm_e, erm_e, holdout_e, detach_classifier, oracle, loss_forward, num_train_domains, max_n_classifiers):
        super(RGM, self).__init__()
        self.num_domains = num_domains
        if num_train_domains < 0:
            num_train_domains = self.num_domains + num_train_domains
        if self.num_domains != 1 and (num_train_domains > self.num_domains - 1 or num_train_domains < 0):
            raise ValueError(f'Invalid value for num_train_domains: {num_train_domains}')
        
        # Set train and extrapolation domain indices for each of the auxiliary classifiers.
        self.all_train_domains, self.all_extra_domains = self.get_train_and_extrapolation_domains(num_train_domains, max_n_classifiers)
        self.n_classifiers = len(self.all_train_domains)
        
        # Deepcopy to not influence the original featurizer and classifier.
        self.featurizer = deepcopy(featurizer)
        self.classifier = deepcopy(classifier)
                
        # extrapolator f_{-e}
        self.f_k = nn.ModuleList(deepcopy(classifier) for _ in range(self.n_classifiers))
        # oracle f_{e}
        self.g_k = nn.ModuleList(deepcopy(classifier) for _ in range(self.n_classifiers))
        
        self.copy_f_k = nn.ModuleList(deepcopy(classifier) for _ in range(self.n_classifiers))
        for copy_f_k in self.copy_f_k:
            copy_f_k.requires_grad_(False)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ensemble = nn.ModuleList([nn.Sequential(self.featurizer, self.f_k[k]) for k in range(self.n_classifiers)])
        
        self.rgm_e = rgm_e
        self.erm_e = erm_e
        self.holdout_e = holdout_e
        self.detach_classifier = detach_classifier
        self.oracle = oracle
        self.loss_forward = loss_forward
        self.register_buffer('update_count', torch.tensor([0]))        
        
        self.log_losses = True
        self.loss_curve_iter = {}
        self.loss_curve_iter['extrapol'] = []
        self.loss_curve_iter['oracle'] = []
        self.loss_curve_iter['erm'] = []
        self.loss_curve_iter['holdout'] = []
        self.loss_curve_iter['regret'] = []
        self.loss_curve_iter['eff_regret'] = []
        self.loss_curve_iter['total'] = []
        self.loss_curve_iter['eff_loss'] = []
        self.loss_curve_iter['rep_loss'] = []
    
    
    def get_train_and_extrapolation_domains(self, num_train_domains, max_n_classifiers):
        """Returns all train and extrapolation domain indices (second dimension) for all auxiliary classifiers (first dimension).
        """
        if self.num_domains > 1:
            domain_idc = range(self.num_domains)
            all_train_domains = tuple(combinations(domain_idc, num_train_domains))
            n_train_domains = len(all_train_domains)
            if n_train_domains > max_n_classifiers:
                all_train_d_idc = range(0, n_train_domains)
                choose_random_idc = np.random.choice(all_train_d_idc, size=max_n_classifiers, replace=False)
                all_train_domains = tuple([domains for i, domains in enumerate(all_train_domains) if i in choose_random_idc])
            
            # Extrapolation domains are all domains that are not train domains.
            all_extra_domains = []
            for train_domains in all_train_domains:
                extra_domains = []
                for i in range(self.num_domains):
                    if not i in train_domains:
                        extra_domains.append(i)
                all_extra_domains.append(tuple(extra_domains))
            all_extra_domains = tuple(all_extra_domains)
        else:
            all_train_domains = ()
            all_extra_domains = ()
        return(all_train_domains, all_extra_domains)


    def forward(self, batches):      
        warnings.filterwarnings("ignore", category=TracerWarning)
        
        # Check that each batch in batches is from one domain.
        for _, _, batch_d, _ in batches:
            assert len(torch.unique(batch_d)) == 1
        assert len(np.unique([torch.unique(batch_d).item() for _, _, batch_d, _ in batches])) == self.num_domains
        assert len(batches) == self.num_domains
        
        # Get copy of f_k without gradient backprop that emulates frozen weights of f_k.
        for k in range(self.n_classifiers):
            self.copy_f_k[k].load_state_dict(self.f_k[k].state_dict())
            
        # Evaluate representation
        all_phis = []
        for batch_x, _, _, _ in batches:
            phi_x = self.featurizer(batch_x)
            all_phis.append(phi_x)
        
        
        # Compute L(f◦phi)
        erm_loss = torch.tensor(0)
        for k, (_, batch_y, batch_d, mask) in enumerate(batches):
            phi_x = all_phis[k]
            if self.detach_classifier:
                phi_x = phi_x.detach()              
            preds = self.classifier(phi_x) 
            erm_loss = erm_loss + self.loss_forward(preds, batch_y, mask)
        norm = self.num_domains
        erm_loss = erm_loss / norm
            
        # Compute regret R^e(phi). 
        extra_loss = torch.tensor(0)
        oracle_loss = torch.tensor(0)
        for k, extra_domains in enumerate(self.all_extra_domains):   # Loop classifiers
            for j in extra_domains:                             # Loop domains
                _, batch_y, batch_d, mask = batches[j]
                phi_k = all_phis[j]
                preds = self.copy_f_k[k](phi_k)     # f_{-e}◦phi (extrapolator)
                norm = len(extra_domains) * len(self.all_extra_domains)
                extra_loss = extra_loss + self.loss_forward(preds, batch_y, mask) / norm
                # The minus in the loss function in the paper 2020 Jin is introduced by the GradientReversal Layer.
                if self.oracle:
                    oracle_preds = self.g_k[k](GradientReversal.apply(phi_k)) # f_{e}◦phi
                    oracle_loss = \
                        oracle_loss + self.loss_forward(oracle_preds, batch_y, mask) / norm                            
        regret = extra_loss + oracle_loss
        eff_regret = extra_loss.item() - oracle_loss.item()
        
        
        # Compute L^{-e}(f_{-e}◦phi). Only for training f_{-e}, gradient is not backpropagated on phi (detached).
        holdout_loss = torch.tensor(0)
        for k, train_domains in enumerate(self.all_train_domains):   # Loop classifiers
            # Train the kth classifier on all train_domains.
            for j in train_domains:                             # Loop domains
                _, batch_y, batch_d, mask = batches[j]
                phi_x = all_phis[j].detach()  # phi does not help f_{-e}
                preds = self.f_k[k](phi_x)
                norm = len(train_domains) * len(self.all_train_domains)
                holdout_loss = holdout_loss + self.loss_forward(preds, batch_y, mask) / norm
        
        
        erm_loss = self.erm_e * erm_loss
        holdout_loss = self.rgm_e * self.holdout_e * holdout_loss
        regret = self.rgm_e * regret
        
        eff_regret = self.rgm_e * eff_regret
        oracle_loss = self.rgm_e * oracle_loss
        extra_loss = self.rgm_e * extra_loss
        eff_loss = (erm_loss + holdout_loss + extra_loss - oracle_loss).item()
        
        loss = erm_loss + holdout_loss + regret
        
        if self.log_losses:
            if self.detach_classifier:
                rep_loss = eff_regret
            else:
                rep_loss = erm_loss.item() + eff_regret        
            self.loss_curve_iter['extrapol'].append(extra_loss.item())
            self.loss_curve_iter['oracle'].append(oracle_loss.item())
            self.loss_curve_iter['regret'].append(regret.item())
            self.loss_curve_iter['eff_regret'].append(eff_regret)
            self.loss_curve_iter['holdout'].append(holdout_loss.item())
            self.loss_curve_iter['erm'].append(erm_loss.item())
            self.loss_curve_iter['total'].append(loss.item())
            self.loss_curve_iter['rep_loss'].append(rep_loss)
            self.loss_curve_iter['eff_loss'].append(eff_loss)
        
        return loss
