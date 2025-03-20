import copy
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn

from fling.utils.registry_utils import CLIENT_REGISTRY
from .base_client import BaseClient


@CLIENT_REGISTRY.register('fedcac_client')
class FedCACClient(BaseClient):
    """
    Overview:
        This class is the base implementation of client in 'Bold but Cautious: Unlocking the Potential of Personalized
        Federated Learning through Cautiously Aggressive Collaboration' (FedCAC).
    """

    def __init__(self, args, client_id, train_dataset, test_dataset=None):
        """
        Initializing train dataset, test dataset(for personalized settings).
        """
        super(FedCACClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.critical_parameter = None  # record the critical parameter positions in FedCAC
        self.customized_model = copy.deepcopy(self.model)  # customized global model

    def train(self, lr, local_critic=0.5, device=None, train_args=None):
        """
        Local training.
        """
        # record the model before local updating, used for critical parameter selection
        initial_model = copy.deepcopy(self.model)

        # local update for several local epochs
        mean_monitor_variables = super().train(lr, device, train_args)

        if self.args.dst.name != 'local':
            self.evaluate_critical_parameter(prevModel=initial_model, model=self.model, tau=local_critic)

        return mean_monitor_variables

    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module,
                                    tau: float) -> Tuple[torch.Tensor, list, list]:
        r"""
        Overview:
            Implement critical parameter selection.
        """
        critical_parameter = []

        self.model.to(self.device)
        prevModel.to(self.device)

        # select critical parameters in each layer
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):

            if name1 not in self.local_masks:
                continue

            g = (param.data - prevparam.data)
            v = param.data

            mask = self.get_critical_mask(g, v, tau, density_init=self.args.dst.density_init, prune_method=self.args.dst.prune_method, prune_ll=self.args.dst.prune_ll)

            critical_parameter.append(mask.view(-1))
            self.local_masks[name1] = mask

            ## keep sparsity in every iteration
            if self.args.dst.update_mode == 'global' or self.args.dst.update_mode == 'fdst' or self.args.dst.update_mode == 'local':
                param.data = param.data * mask.to(self.args.learn.device)

        model.zero_grad()
        self.critical_parameter = torch.cat(critical_parameter)

        self.model.to('cpu')
        prevModel.to('cpu')


    def get_critical_mask(self, g, v, tau, density_init, prune_method='sensity', prune_ll='large'):
        """
        Args:
        - g: (param.data - prevparam.data)
        - v: (param.data)
        - tau: control the numbers of critical
        - density_init:
        - prune_method: sensity, random, magnitude
        """

        if prune_method == 'sensity':
            metric = torch.abs(g * v).view(-1)
        elif prune_method == 'magnitude':
            metric = torch.abs(v).view(-1)
        elif prune_method == 'gradient':
            metric = torch.abs(g).view(-1)
        elif prune_method == 'random':
            non_zero_mask = (v != 0).view(-1)
            metric = torch.zeros_like(v.view(-1))  # Initialize metric with zeros
            random_values = torch.rand(non_zero_mask.sum(), device=v.device)  # Generate random values for non-zero elements
            metric[non_zero_mask] = random_values

        num_nonzero_params = int(metric.size(0) * density_init)

        if self.args.dst.update_mode == 'global' or self.args.dst.update_mode == 'fdst' or self.args.dst.update_mode == 'local':
            tau = 1.0

        nz = int(tau * num_nonzero_params)

        mask = torch.zeros_like(metric)

        if prune_ll == 'large':
            _, indices = torch.topk(metric, nz, largest=True)
        else:
            # Select the smallest-k values
            _, indices = torch.topk(metric, nz, largest=False)


        mask[indices] = 1.0

        return mask.view_as(v).int().to('cpu')

