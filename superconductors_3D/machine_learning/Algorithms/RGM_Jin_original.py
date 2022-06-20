import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

class GradientReversal(torch.autograd.Function):
    beta = 1.

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return -GradientReversal.beta * grad_output


class RGM(nn.Module):

    def __init__(self, featurizer, linear, hidden_size, output_size, num_domains, rgm_e, erm_e, holdout_e, detach_classifier, oracle, loss_forward, num_train_domains):
        # Custom: Replaced args.(...) by the regarding variable.
        super(RGM, self).__init__()
        self.featurizer = featurizer
        if linear:
            self.classifier = nn.Linear(hidden_size, output_size)
            self.copy_f_k = nn.ModuleList(
                    [nn.Linear(hidden_size, output_size).requires_grad_(False) for _ in range(num_domains)]
            )
            self.f_k = nn.ModuleList(
                    [nn.Linear(hidden_size, output_size) for _ in range(num_domains)]
            )
            self.g_k = nn.ModuleList(
                    [nn.Linear(hidden_size, output_size) for _ in range(num_domains)]
            )
        else:
            self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
            )
            self.copy_f_k = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size),
                    ).requires_grad_(False) for _ in range(num_domains)
            ])
            self.f_k = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size),
                    ) for _ in range(num_domains)
            ])
            self.g_k = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size),
                    ) for _ in range(num_domains)
            ])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.loss_forward = loss_forward
        self.rgm_e = rgm_e
        self.num_domains = num_domains
        self.register_buffer('update_count', torch.tensor([0]))
        self.loss_curve_iter = {}   # Custom
        self.train_representation = True    # Custom
        self.train_classifier = True        # Custom

    # def loss_forward(self, preds, batch_y, mask): # Custom
    #     pred_loss = self.loss_func(preds, batch_y) * mask
    #     return pred_loss.sum() / mask.sum()

    def forward(self, batches):
        assert len(batches) == 2
        for k in range(self.num_domains):
            self.copy_f_k[k].load_state_dict(self.f_k[k].state_dict())

        erm_loss = 0
        all_phis = []
        for batch_x, batch_y, batch_d, mask in batches:
            phi_x = self.featurizer(batch_x)
            all_phis.append(phi_x)
            preds = self.classifier(phi_x) 
            erm_loss = erm_loss + self.loss_forward(preds, batch_y, mask)

        regret = 0
        for k in range(self.num_domains):
            _, batch_y, _, mask = batches[k]
            phi_k = all_phis[k]
            preds = self.copy_f_k[k](phi_k) 
            oracle_preds = self.g_k[k](GradientReversal.apply(phi_k))
            regret = regret + self.loss_forward(preds, batch_y, mask) + self.loss_forward(oracle_preds, batch_y, mask)

        holdout_loss = 0
        for k in range(self.num_domains):
            _, batch_y, _, mask = batches[1 - k]  # hardcode: 2 domains
            phi_x = all_phis[1 - k].detach()  # phi does not help f_{-e}
            preds = self.f_k[k](phi_x)
            holdout_loss = holdout_loss + self.loss_forward(preds, batch_y, mask)

        loss = erm_loss + holdout_loss + self.rgm_e * regret
        return loss / self.num_domains
