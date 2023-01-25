import math
from collections import OrderedDict
from copy import deepcopy
from typing import List

import torch
import numpy as np
from rich.progress import track
from tqdm import tqdm

from fedavg import FedAvgServer
from config.args import get_fedap_argparser
from config.utils import trainable_params
from client.fedap import FedAPClient

# Codes below are modified from FedAP's official repo: https://github.com/microsoft/PersonalizedFL
class FedAPServer(FedAvgServer):
    def __init__(self):
        args = get_fedap_argparser().parse_args()
        super().__init__("FedAP", args, unique_model=True, default_trainer=False)
        self.trainer = FedAPClient(deepcopy(self.model), self.args, self.logger)
        self.weight_matrix = torch.zeros(
            (self.client_num_in_total, self.client_num_in_total), device=self.device
        )

    def train(self):
        if self.args.pretrain_epoch > 0 and self.args.pretrain_ratio > 0:
            # Pre-training phase
            self.trainer.pretrain = True
            pretrain_params = OrderedDict(
                zip(self.trainable_params_name, trainable_params(self.model))
            )

            pretrain_progress_bar = (
                track(
                    range(self.args.pretrain_epoch),
                    "[bold green]Pre-Training...",
                    console=self.logger,
                )
                if not self.args.log
                else tqdm(range(self.args.pretrain_epoch), "Pre-Training...")
            )
            for E in pretrain_progress_bar:
                for client_id in self.train_clients:
                    new_params, _, _ = self.trainer.train(
                        client_id,
                        pretrain_params,
                        return_diff=False,
                        verbose=((E + 1) % self.args.verbose_gap) == 0,
                    )
                    for old_param, new_param in zip(
                        pretrain_params.values(), new_params
                    ):
                        old_param.data = new_param.data

            # update clients params to pretrain params
            self.model.load_state_dict(pretrain_params, strict=False)
            self.client_trainable_params = [
                deepcopy(trainable_params(self.model)) for _ in self.train_clients
            ]

        # generate weight matrix
        bn_mean_list, bn_var_list = [], []
        for client_id in track(
            self.train_clients,
            "[bold cyan]Generating weight matrix...",
            console=self.logger,
            disable=self.args.log,
        ):
            avgmeta = metacount(self.get_form()[0])
            client_local_params = self.generate_client_params(client_id)
            features_list, batch_size_list = self.trainer.get_all_features(
                client_id, client_local_params
            )
            with torch.no_grad():
                for features, batchsize in zip(features_list, batch_size_list):
                    tm, tv = [], []
                    for item in features:
                        if len(item.shape) == 4:
                            tm.append(
                                torch.mean(item, dim=[0, 2, 3]).detach().cpu().numpy()
                            )
                            tv.append(
                                torch.var(item, dim=[0, 2, 3]).detach().cpu().numpy()
                            )
                        else:
                            tm.append(torch.mean(item, dim=0).detach().cpu().numpy())
                            tv.append(torch.var(item, dim=0).detach().cpu().numpy())
                    avgmeta.update(batchsize, tm, tv)
            bn_mean_list.append(avgmeta.getmean())
            bn_var_list.append(avgmeta.getvar())
        self.generate_weight_matrix(bn_mean_list, bn_var_list)

        # regular training
        self.trainer.pretrain = False
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            client_params_cache = []
            for client_id in self.selected_clients:
                client_local_params = self.generate_client_params(client_id)
                new_params, self.clients_metrics[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                client_params_cache.append(new_params)

            self.update_client_params(client_params_cache)
            self.log_info()

    def get_form(self):
        tmpm = []
        tmpv = []
        for name in self.model.state_dict().keys():
            if "running_mean" in name:
                tmpm.append(self.model.state_dict()[name].detach().to("cpu").numpy())
            if "running_var" in name:
                tmpv.append(self.model.state_dict()[name].detach().to("cpu").numpy())
        return tmpm, tmpv

    def generate_weight_matrix(
        self, bnmlist: List[torch.Tensor], bnvlist: List[torch.Tensor]
    ):
        client_num = len(bnmlist)
        weight_m = np.zeros((client_num, client_num))
        for i in range(client_num):
            for j in range(client_num):
                if i == j:
                    weight_m[i, j] = 0
                else:
                    tmp = wasserstein(bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                    if tmp == 0:
                        weight_m[i, j] = 100000000000000
                    else:
                        weight_m[i, j] = 1 / tmp
        weight_s = np.sum(weight_m, axis=1)
        weight_s = np.repeat(weight_s, client_num).reshape((client_num, client_num))
        weight_m = (weight_m / weight_s) * (1 - self.args.model_momentum)
        for i in range(client_num):
            weight_m[i, i] = self.args.model_momentum
        self.weight_matrix = torch.from_numpy(weight_m).to(self.device)

    def generate_client_params(self, client_id) -> OrderedDict[str, torch.Tensor]:
        new_parameters = OrderedDict()
        for name, layer_params in zip(
            self.trainable_params_name, zip(*self.client_trainable_params)
        ):
            new_parameters[name] = torch.sum(
                torch.stack(layer_params, dim=-1) * self.weight_matrix[client_id],
                dim=-1,
            )
        return new_parameters


def wasserstein(m1, v1, m2, v2, mode="nosquare"):
    W = 0
    bn_layer_num = len(m1)
    for i in range(bn_layer_num):
        tw = 0
        tw += np.sum(np.square(m1[i] - m2[i]))
        tw += np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i])))
        if mode == "square":
            W += tw
        else:
            W += math.sqrt(tw)
    return W


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count + m
        for i in range(self.bl):
            tmpm = (self.mean[i] * self.count + tm[i] * m) / tmpcount
            self.var[i] = (
                self.count * (self.var[i] + np.square(tmpm - self.mean[i]))
                + m * (tv[i] + np.square(tmpm - tm[i]))
            ) / tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var


if __name__ == "__main__":
    server = FedAPServer()
    server.run()