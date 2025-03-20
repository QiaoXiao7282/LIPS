import time
import copy
import torch
import torch.nn as nn
from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup
import matplotlib.pyplot as plt
import random
import numpy as np

@GROUP_REGISTRY.register('fedcac_group')
class FedCACServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedCAC.
    """

    def __init__(self, args: dict, logger: Logger):
        super(FedCACServerGroup, self).__init__(args, logger)
        # To be consistent with the existing pipeline interface. group maintains an epoch counter itself.
        self.epoch = -1
        self.train_round=0
        self.base_glob_dict=None


    def sync(self) -> None:
        r"""
        Overview:
            Perform the critical and non-critical parameter initialization steps in FedCAC.
        """
        if self.epoch == -1:
            super().sync()  # Called during system initialization
        else:
            tempGlobalModel = copy.deepcopy(self.clients[0].model)
            tempGlobalModel.load_state_dict(self.server.glob_dict)
            tempGlobalModel.to(self.args.learn.device)

            for client in range(self.args.client.client_num):
                index = 0
                self.clients[client].model.to(self.args.learn.device)
                # self.clients[client].customized_model.to(self.args.learn.device)

                ## local, global
                for (name1, param1), (name2, param2) in zip(self.clients[client].model.named_parameters(), tempGlobalModel.named_parameters()):

                    local_mask = self.clients[client].local_masks[name1].to(self.args.learn.device).float()

                    if name1 not in self.clients[client].local_masks:
                        continue

                    grad_list = [grad[name1] for grad in self.clients[client].local_grad if name1 in grad]

                    global_mask, global_weights = self.regrow_mask_func(local_mask, param2, self.args.learn.local_eps_first_d, regrow_method=self.args.dst.regrow_method, name=name1, grad_list=grad_list, no_margin=self.args.dst.no_margin)


                    ## only local
                    if self.args.dst.update_mode == 'local':
                        param1.data = local_mask * param2.data


                    elif self.args.dst.update_mode == 'global_sel':
                        param1.data = global_mask * global_weights

                        new_local_mask = torch.ones_like(global_mask).to('cpu')
                        self.clients[client].local_masks[name1] = new_local_mask

                    ## for zero position, init using ori weights/random noise
                    elif self.args.dst.update_mode == 'global_sel_init':

                        if self.args.learn.local_eps_first_d < 1.0:
                            local_init_on_device = self.clients[client].local_init[name1].to(global_weights.device)
                            # noise = torch.randn_like(local_init_on_device)  # Gaussian noise

                            # Apply the global mask and preserve initialization for unmasked regions
                            aa = global_mask * global_weights
                            bb = (1 - global_mask) * local_init_on_device #noise #local_init_on_device
                            param1.data = aa + bb
                        else:
                            param1.data = global_mask * global_weights

                        new_local_mask = torch.ones_like(global_mask).to('cpu')
                        self.clients[client].local_masks[name1] = new_local_mask


                    elif self.args.dst.update_mode == 'fdst': ## all from avg model
                        update_mask = torch.zeros_like(param2.data)
                        num_to_update = int(param2.data.numel() * self.clients[client].density_init)

                        abs_sorted_weights, idx = torch.sort(torch.abs(param2.view(-1)), descending=True)
                        top_indices = idx[:num_to_update]

                        update_mask.view(-1)[top_indices] = 1.0
                        param1.data = update_mask * param2.data

                        self.clients[client].local_masks[name1] = update_mask.to('cpu')


                    index += 1
                self.clients[client].model.to('cpu')
                self.clients[client].customized_model.to('cpu')
            tempGlobalModel.to('cpu')

        self.epoch += 1


    def aggregate(self, train_round: int) -> int:
        r"""
        Overview:
            Aggregate all client models.
            Generate customized global model for each client.
        Arguments:
            - train_round: current global epochs.
        Returns:
            - trans_cost: uplink communication cost.
        """
        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(self.clients, self.server)
            # trans_cost += self.get_customized_global_models()
            self.sync()  ## update local model
        else:
            print('Unrecognized compression method: ' + self.args.group.aggregation_method)
            assert False

        self.train_round = train_round

        if self.train_round == 2:
            self.base_glob_dict = copy.deepcopy(self.server.glob_dict)

        if self.train_round == 50:
            self.glob_dict_fix = copy.deepcopy(self.server.glob_dict)

        if self.base_glob_dict:
            current_glob_dict = copy.deepcopy(self.server.glob_dict)
            metrics = {}
            for key in self.base_glob_dict.keys():
                if self.base_glob_dict[key].dim() == 4 and key in current_glob_dict or 'classifier' in key:
                    # Compute cosine similarity
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        self.base_glob_dict[key].view(-1),
                        current_glob_dict[key].view(-1),
                        dim=0
                    )
                    # Compute L1 norm
                    l1_norm = torch.norm(
                        self.base_glob_dict[key].view(-1) - current_glob_dict[key].view(-1), p=1
                    )
                    # Compute L2 norm
                    l2_norm = torch.norm(
                        self.base_glob_dict[key].view(-1) - current_glob_dict[key].view(-1), p=2
                    )

                    metrics[key] = {
                        "cosine_similarity": cosine_similarity.item(),
                        "l1_norm": l1_norm.item(),
                        "l2_norm": l2_norm.item(),
                    }

            # Print the metrics
            for key, metric in metrics.items():
                print(
                    f"Metrics for {key}: Cosine Similarity = {metric['cosine_similarity']:.4f}, L1 Norm = {metric['l1_norm']:.4f}, L2 Norm = {metric['l2_norm']:.4f}"
                )

        # Add logger for time per round.
        # This time is the interval between two times of executing this ``aggregate()`` function.
        time_per_round = time.time() - self._time
        self._time = time.time()
        self.logger.add_scalar('time/time_per_round', time_per_round, train_round)

        return trans_cost

    def regrow_mask_func(self, local_mask, param2, density_init, regrow_method='topk', name='', grad_list=None, no_margin=True):

        candidate_mask = ~local_mask.bool()
        global_weights = candidate_mask.float() * param2.data  #
        num_to_regrow = int(local_mask.numel() * density_init - local_mask.sum().item())
        # num_to_regrow = int(local_mask.numel() * 1.0 - local_mask.sum().item())

        global_mask = torch.zeros_like(global_weights)

        if num_to_regrow > 0:
            flat_global_weights = global_weights.view(-1)
            flat_candidate_mask = candidate_mask.view(-1)

            valid_indices = torch.nonzero(flat_candidate_mask).squeeze()

            if density_init==1.0:
                abs_sorted_weights, idx = torch.sort(torch.abs(flat_global_weights[valid_indices]), descending=True)
                regrow_indices = valid_indices[idx[:num_to_regrow]]

            if regrow_method == 'topk' and density_init<1.0:

                ## sel layers
                module_name = '.'.join(name.split('.')[:-1])
                module = dict(self.clients[0].model.named_modules())[module_name]

                if no_margin:
                    if not isinstance(module, nn.Conv2d) or module.kernel_size == (7, 7) or 'features.0' in name or 'classifier' in name:  # nn.Conv2d, nn.Linear
                        return torch.ones_like(global_weights), global_weights.clone()

                abs_sorted_weights, idx = torch.sort(torch.abs(flat_global_weights[valid_indices]), descending=True)
                regrow_indices = valid_indices[idx[:num_to_regrow]]

            if regrow_method == 'sensityk' and density_init<1.0:

                total_grad = torch.sum(torch.stack(grad_list), dim=0)  # Sum across all gradients in the list
                sensity = total_grad * global_weights
                flat_global_weights = sensity.view(-1)

                ## sel layers
                module_name = '.'.join(name.split('.')[:-1])
                module = dict(self.clients[0].model.named_modules())[module_name]

                ## exclude first and last layer: return mask with all one
                if no_margin:
                    if not isinstance(module, nn.Conv2d) or module.kernel_size == (7, 7) or 'features.0' in name or 'classifier' in name:  # nn.Conv2d, nn.Linear
                        return torch.ones_like(global_weights), global_weights.clone()

                abs_sorted_weights, idx = torch.sort(torch.abs(flat_global_weights[valid_indices]), descending=True)
                regrow_indices = valid_indices[idx[:num_to_regrow]]

            if regrow_method == 'lowk' and density_init<1.0:

                ## sel layers
                module_name = '.'.join(name.split('.')[:-1])
                module = dict(self.clients[0].model.named_modules())[module_name]

                if no_margin:
                    if not isinstance(module, nn.Conv2d) or module.kernel_size == (7, 7) or 'features.0' in name or 'classifier' in name:  # nn.Conv2d, nn.Linear
                        return torch.ones_like(global_weights), global_weights.clone()

                abs_sorted_weights, idx = torch.sort(torch.abs(flat_global_weights[valid_indices]), descending=False)
                regrow_indices = valid_indices[idx[:num_to_regrow]]

            if regrow_method == 'randomk' and density_init<1.0:

                ## sel layers
                module_name = '.'.join(name.split('.')[:-1])
                module = dict(self.clients[0].model.named_modules())[module_name]

                if no_margin:
                    if not isinstance(module, nn.Conv2d) or module.kernel_size == (7, 7) or 'features.0' in name or 'classifier' in name:  # nn.Conv2d, nn.Linear
                        return torch.ones_like(global_weights), global_weights.clone()

                random_indices = torch.randperm(len(valid_indices))[:num_to_regrow]
                regrow_indices = valid_indices[random_indices]


            global_mask.view(-1)[regrow_indices] = 1.0

        regrow_weights = global_weights.clone()

        return global_mask, regrow_weights
