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

    def __init__(self, featurizer, loss_func, args):
        super(RGM, self).__init__()
        self.featurizer = featurizer
        if args.linear:
            # f,trained on all environments
            self.classifier = nn.Linear(args.hidden_size, args.output_size)
            # copy of f_{-e}
            self.copy_f_k = nn.ModuleList(
                    [nn.Linear(args.hidden_size, args.output_size).requires_grad_(False) for _ in range(args.num_domains)]
            )
            # extrapolator f_{-e}
            self.f_k = nn.ModuleList(
                    [nn.Linear(args.hidden_size, args.output_size) for _ in range(args.num_domains)]
            )
            # oracle predictor f_{e}
            self.g_k = nn.ModuleList(
                    [nn.Linear(args.hidden_size, args.output_size) for _ in range(args.num_domains)]
            )
        else:
            # f,trained on all environments
            self.classifier = nn.Sequential(
                    nn.Linear(args.hidden_size, args.hidden_size),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.output_size),
            )
            # copy of f_{-e}
            self.copy_f_k = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(args.hidden_size, args.hidden_size),
                        nn.ReLU(),
                        nn.Linear(args.hidden_size, args.output_size),
                    ).requires_grad_(False) for _ in range(args.num_domains)
            ])
            # extrapolator f_{-e}
            self.f_k = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(args.hidden_size, args.hidden_size),
                        nn.ReLU(),
                        nn.Linear(args.hidden_size, args.output_size),
                    ) for _ in range(args.num_domains)
            ])
            # oracle predictor f_{e}
            self.g_k = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(args.hidden_size, args.hidden_size),
                        nn.ReLU(),
                        nn.Linear(args.hidden_size, args.output_size),
                    ) for _ in range(args.num_domains)
            ])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.loss_func = loss_func
        self.rgm_e = args.rgm_e
        self.num_domains = args.num_domains
        self.register_buffer('update_count', torch.tensor([0]))

    def loss_forward(self, preds, batch_y, mask):
        pred_loss = self.loss_func(preds, batch_y) * mask
        return pred_loss.sum() / mask.sum()

    def forward(self, batches):
        # assert len(batches) == 2  # TODO
        for k in range(self.num_domains):
            self.copy_f_k[k].load_state_dict(self.f_k[k].state_dict())
        
        # Compute L(f◦phi)
        erm_loss = 0
        all_phis = []
        for batch_x, batch_y, mask in batches:
            phi_x = self.featurizer(batch_x)
            all_phis.append(phi_x)
            preds = self.classifier(phi_x) 
            erm_loss = erm_loss + self.loss_forward(preds, batch_y, mask)
        
        # Compute regret R^e(phi). 
        regret = 0
# =============================================================================
#         for k in range(self.num_domains):
#             _, batch_y, mask = batches[k]
#             phi_k = all_phis[k]
#             preds = self.copy_f_k[k](phi_k)     # f_{-e}◦phi
#             oracle_preds = self.g_k[k](GradientReversal.apply(phi_k))   # f_{e}◦phi
#             # The minus in the loss function in the paper 2020 Jin here is introduced by the GradientReversal Layer.
#             regret = regret + self.loss_forward(preds, batch_y, mask) + self.loss_forward(oracle_preds, batch_y, mask)
# =============================================================================
        
        
        # TODO
        # Compute L^{-e}(f_{-e}◦phi). Only for training f_{-e}, gradient is not backpropagated on phi (detached).
        holdout_loss = 0
# =============================================================================
#         for k in range(self.num_domains):
#             _, batch_y, mask = batches[1 - k]  # hardcode: 2 domains
#             phi_x = all_phis[1 - k].detach()  # phi does not help f_{-e}
#             preds = self.f_k[k](phi_x)
#             holdout_loss = holdout_loss + self.loss_forward(preds, batch_y, mask)
# =============================================================================

        loss = erm_loss + holdout_loss + self.rgm_e * regret
        return loss / self.num_domains
