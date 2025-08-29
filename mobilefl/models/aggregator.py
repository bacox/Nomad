import collections
import copy
import time
import warnings
from math import exp
from typing import List, OrderedDict, Tuple
import torch
from mobilefl.log_tools.logging_style import content, line
from mobilefl.models.cnncifar import CNNCIFAR
from mobilefl.models.cnnmnist import CNNMNIST
from mobilefl.models.lenet import LeNet
from mobilefl.models.lstm import NextCharacterLSTM
class Aggregator:
    def __init__(self, config, vocab_size=0, params=None):
        self.dataset = config.get("dataset")
        self.config = config
        self.client_async = config.get("client_async")
        self.lr = self.config.get("global_lr")
        self.lr_peer = self.config.get("global_lr_peer")
        self.alpha = self.config.get("staleness_alpha")
        self.a = self.config.get("staleness_a")
        self.momentum_constant = self.config.get("momentum_server")
        self.global_momentum_constant = self.config.get("momentum_global")
        self.momentum = None
        self.compute_time = collections.defaultdict(float)
        self.compute_cnt = collections.defaultdict(int)
        self.weight_tracking = collections.defaultdict(list)
        self.model = Aggregator.get_model(self.dataset, vocab_size=vocab_size, params=params)
        if self.config.get("cuda"):
            torch.cuda.manual_seed(0)
            self.device = torch.device(f"cuda:{self.config.get('cuda_to_use')}")
        else:
            self.device = "cpu"
        self.model.to(device=self.device)
    @staticmethod
    def get_model(dataset, vocab_size=None, params=None):
        """
        Gets the model according to model name and given parameters
        :param dataset: a string indicating the dataset (e.g. "mnist")
        :param params: parameters of the model
        """
        if dataset == "mnist":
            if not params:
                return CNNMNIST()
            else:
                if len(params) < 3:
                    raise ValueError(f"There should be 3 parameters for CNN model, got {len(params)}")
                return CNNMNIST(params[0], params[1], params[2])
        if dataset == "fmnist":
            if not params:
                return CNNMNIST()
            else:
                if len(params) < 3:
                    raise ValueError(f"There should be 3 parameters for CNN model, got {len(params)}")
                return CNNMNIST(params[0], params[1], params[2])
        elif dataset == "cifar":
            return CNNCIFAR()
        elif dataset == "cifar100":
            return LeNet()
        elif dataset == "wikitext2":
            return NextCharacterLSTM(vocab_size=vocab_size, embed_size=100, hidden_size=256, n_layers=2)
        else:
            raise ValueError(f"Dataset not supported: {dataset}")
    def aggregate(self, updates, cosine=False, aggregation_method="fedavg"):
        """
        Should be called by the server's global model
        gm.aggregate(updates, aggregation_method)
        :param updates: a list containing each update
        :param aggregation_method: fedavg or fedsgd
        :return: None
        """
        if not updates:
            warnings.warn("Updates is empty. Skipping aggregation")
            return
        if not self.client_async:
            if len(updates[0]) != 2:
                raise ValueError(f"dimension of every update for synchronous model should be 1, got {len(updates[0])}")
        else:
            if len(updates[0]) != 3:
                raise ValueError(f"dimension of every update for asynchronous model should be 3, got {len(updates[0])}")
        if aggregation_method == "fedasync":
            self.fedasync(updates)
        elif aggregation_method == "fedavg":
            raise NotImplementedError("FedAvg is not implemented yet")
            self.average(updates)
        elif aggregation_method == "fedsgd":
            return self.sgd(updates, cosine=cosine)
        else:
            raise Exception("Aggregation method not supported")
    def average(self, updates):
        """
        updates: a list of updates
        edit the global model
        Not suitable for async local training with buffer size 1
        """
        raise NotImplementedError("FedAvg is not implemented yet")
        start_time = time.perf_counter()
        n = len(updates)
        if not self.client_async:
            weights = [1 / n for _ in range(n)]
        else:  
            staleness = [updates[i][2] for i in range(n)]
            weights = self.get_weights_according_to_staleness_avg(staleness)
        weights = torch.tensor(weights)
        weights.to(device=self.device)
        for part in self.model.state_dict().keys():
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            self.model.state_dict()[part].data.fill_(0)
            for i in range(n):
                update = updates[i][0][part].data.clone() * weights[i]  
                update.type(self.model.state_dict()[part].data.dtype)
                self.model.state_dict()[part].data += update
        end_time = time.perf_counter()
        self.compute_time["fedavg"] += end_time - start_time
        self.compute_cnt["fedavg"] += 1
    def sgd(self, updates, cosine=False):
        """
        updates: a list of updates
        edit the global model
        return the offset of the global model
        """
        start_time = time.perf_counter()
        n = len(updates)
        original_state = copy.deepcopy(self)
        if not self.client_async:
            weights = [1 / n for _ in range(n)]
        else:  
            staleness = [updates[i][2] for i in range(n)]
            weights = self.get_weights_according_to_staleness_sgd(staleness)
        weights = torch.tensor(weights).to(device=self.device)
        print(f"Staleness factor: {weights}")
        if self.momentum is None:
            self.momentum = {}
            for part in self.model.state_dict().keys():
                self.momentum[part] = torch.zeros_like(self.model.state_dict()[part].data).to(device=self.device)
                self.momentum[part].type(self.model.state_dict()[part].data.dtype)
        gradients_before = []
        gradients_after = []
        for part in self.model.state_dict().keys():
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            gradient_before = self.model.state_dict()[part].data.clone().detach().fill_(0)
            for i in range(n):
                gradient_before += (
                    updates[i][0][part].data.clone() - self.model.state_dict()[part].data.clone()
                ) * weights[i]
            gradients_before.append(gradient_before.view(-1))
        for part in self.model.state_dict().keys():
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            gradient = self.model.state_dict()[part].data.clone().detach().fill_(0)
            for i in range(n):
                gradient += (updates[i][0][part].data.clone() - self.model.state_dict()[part].data.clone()) * weights[i]
            if self.global_momentum_constant > 0:
                self.momentum[part].data = (
                    self.global_momentum_constant * self.momentum[part].data.clone() + gradient.clone()
                )
                self.model.state_dict()[part].data += self.momentum[part].data.clone() * self.lr
            else:
                self.model.state_dict()[part].data += gradient.clone() * self.lr
        for part in self.model.state_dict().keys():
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            gradient_after = self.model.state_dict()[part].data.clone().detach().fill_(0)
            for i in range(n):
                gradient_after += (
                    updates[i][0][part].data.clone() - self.model.state_dict()[part].data.clone()
                ) * weights[i]
            gradients_after.append(gradient_after.view(-1))
        end_time = time.perf_counter()
        self.compute_time["sgd"] += end_time - start_time
        self.compute_cnt["sgd"] += 1
        if cosine:
            gradients_before = torch.cat(gradients_before)
            gradients_after = torch.cat(gradients_after)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            gradients_before = gradients_before.cuda() if torch.cuda.is_available() else gradients_before
            gradients_after = gradients_after.cuda() if torch.cuda.is_available() else gradients_after
            cos_value = cos(gradients_before, gradients_after).item()
            print("Cosine similarity = ", cos_value)
            self.__dict__.update(original_state.__dict__)
    def fedasync(self, updates):
        """
        New global model = (1 - lr) * global model + lr * update;
        lr = weight * lr
        """
        n = len(updates)
        try:
            staleness = [updates[i][2] for i in range(len(updates))]
            weights = self.get_weights_according_to_staleness_sgd(staleness)
        except IndexError:
            raise ValueError("Staleness not found in updates")
        weights = torch.tensor(weights)
        weights.to(device=self.device)
        if self.momentum is None:
            self.momentum = {}
            for part in self.model.state_dict().keys():
                self.momentum[part] = torch.zeros(self.model.state_dict()[part].data.shape).to(device=self.device)
                self.momentum[part].type(self.model.state_dict()[part].data.dtype)
        for part in self.model.state_dict().keys():
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            gradient = self.model.state_dict()[part].data.clone().detach().fill_(0)
            for i in range(n):
                gradient += (updates[i][0][part].data.clone() - self.model.state_dict()[part].data.clone()) * weights[i]
            self.momentum[part].data = self.momentum_constant * self.momentum[part].data.clone() + gradient.clone()
            self.model.state_dict()[part].data += self.momentum[part].data.clone() * self.alpha
    def get_weights_according_to_staleness_avg(self, staleness):
        """
        In the increasing theme, weight[i] = staleness[i] / sum(staleness).
        In the decreasing theme, weight[i] = staleness[i]^-a / sum of all inverse of staleness.
        In the none theme, weight[i] = 1 / len(staleness)
        """
        weights = []
        staleness_scheme = self.config.get("staleness_scheme")
        modified_staleness = [max(0, s) + 1 for s in staleness]  
        n = len(modified_staleness)
        if staleness_scheme == "increasing":
            denom = sum(modified_staleness)
        elif staleness_scheme == "decreasing":
            denom = sum([1 / s for s in modified_staleness])
        elif staleness_scheme == "none":
            return [1 / n for _ in range(n)]
        else:
            raise ValueError("Only increasing/decreasing/none is valid for staleness scheme")
        for s in modified_staleness:
            nom = s if staleness_scheme == "increasing" else 1 / s
            weights.append(nom / denom)
        return weights
    def get_weights_according_to_staleness_sgd(self, staleness):
        """
        In the increasing theme, weight[i] = staleness[i] / sum(staleness).
        In the decreasing theme, weight[i] = staleness[i]^-a.
        """
        _weights = []
        staleness_scheme = self.config.get("staleness_scheme")
        if staleness_scheme == "increasing":
            return staleness
        elif staleness_scheme == "decreasing":
            for i, s in enumerate(staleness):
                if s < 0:
                    staleness[i] = 0
            return [
                1 / (min(self.config.get("staleness_max"), s) + 1) ** self.config.get("staleness_a") for s in staleness
            ]
        elif staleness_scheme == "none":
            return [1 for _ in range(len(staleness))]
        else:
            raise ValueError("Only increasing/decreasing/none is valid for staleness scheme")
    def synchronize(self, peer_buffer: List[Tuple[OrderedDict, int]]):
        """
        Synchronize the global model with other global models
        Return the average age
        """
        start_time = time.perf_counter()
        n = len(peer_buffer)
        ages = [peer_buffer[i][1] for i in range(n)]
        weights = self.get_weights_according_to_age(ages)
        weights = torch.tensor(weights)
        print(f"weights: {weights}")
        print(f"ages: {ages}")
        weights.to(device=self.device)
        new_params = OrderedDict()
        for (sd, _a), w in zip(peer_buffer, weights):
            for k, v in sd.items():
                new_params[k] = new_params.get(k, 0) + v * w
        if self.momentum_constant > 0:
            gradient = {k: v.clone().detach().to(device=self.device) for k, v in self.model.state_dict().items()}
            for part in self.model.state_dict().keys():
                gradient[part].data -= new_params[part].data.clone()
                self.momentum[part] = (
                    self.momentum_constant * self.momentum[part].data.clone() + gradient[part].data.clone()
                )
        new_age = 0
        for i in range(n):
            new_age += ages[i] * weights[i].item()
        end_time = time.perf_counter()
        self.compute_time["sync"] += end_time - start_time
        self.compute_cnt["sync"] += 1
        return round(new_age, 1)
    def cloud_agg(self, server_buffer):
        for j, part in enumerate(self.model.state_dict().keys()):
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            self.model.state_dict()[part].data.fill_(0)
            for i in range(len(server_buffer)):
                update = server_buffer[i][part].data.clone() / len(server_buffer)
                self.model.state_dict()[part].data += update
    def get_weights_according_to_age(self, ages):
        """
        In the increasing theme, weight[i] = age[i] / sum(age).
        In the decreasing theme, weight[i] = age[i]^-1 / sum of all inverse of age.
        In the none theme, weight[i] = 1 / len(age)
        """
        if not any(ages):  
            return [1 / len(ages) for _ in range(len(ages))]
        weights = []
        _num_ages = len(ages)
        denom = sum(ages)
        for a in ages:
            nom = a
            weights.append(nom / denom)
        return weights
    def aggregate_peer_sgd(self, peer_state_dict, server_age, peer_age):
        """
        If age_diff < 0, then the peer is younger than the server
        Adjust the weight according to the age difference
        With the age diff increases, the peer model is more important, so the weight increases
        After aggregation, the age takes a step forward
        Eliminate all the offset history
        """
        start_time = time.perf_counter()
        model_offsets = {}
        for part in self.model.state_dict().keys():
            model_offsets[part] = torch.zeros(self.model.state_dict()[part].data.shape)
            model_offsets[part].type(self.model.state_dict()[part].data.dtype)
            model_offsets[part].to(device=self.device)
        age_diff = peer_age - server_age
        activation_rate = self.config.get("activation_rate")
        weight = 1 / (exp(-activation_rate * age_diff / max(1, server_age)) + 1)
        _decay_center = 100
        weight_decay = 1
        copied_peer_state_dict = {k: v.clone().detach().to(device=self.device) for k, v in peer_state_dict.items()}
        for part in self.model.state_dict().keys():
            self.momentum[part].data = self.momentum_constant * self.momentum[part].data.clone() + (
                copied_peer_state_dict[part].data - self.model.state_dict()[part].data
            )
            if not torch.is_floating_point(self.model.state_dict()[part].data):
                continue
            learning_coefficient = self.lr_peer * weight * weight_decay
            offset_part = (
                copied_peer_state_dict[part].data - self.model.state_dict()[part].data
            ) * learning_coefficient
            self.model.state_dict()[part].data.add_(offset_part)
            model_offsets[part] = offset_part
        new_age = round(
            (1 - self.lr_peer * weight * weight_decay) * server_age + self.lr_peer * weight * weight_decay * peer_age
        )  
        end_time = time.perf_counter()
        self.compute_time["aggr_peer"] += end_time - start_time
        self.compute_cnt["aggr_peer"] += 1
        return new_age, model_offsets, learning_coefficient
    def report(self):
        with open(
            f"./results/{self.config.get('result_file')}/{self.config.get('name')}.txt",
            "a",
        ) as f:
            f.write(line("Delays") + "\n")
            if self.compute_cnt["sgd"] > 0:
                f.write(content(f"total sgd time: {self.compute_time['sgd']} for one server") + "\n")
                f.write(content(f"average sgd time: {self.compute_time['sgd'] / self.compute_cnt['sgd']}") + "\n")
            if self.compute_cnt["aggr_peer"] > 0:
                f.write(content(f"total aggr_peer time: {self.compute_time['aggr_peer']} for one server") + "\n")
                f.write(
                    content(f"average aggr_peer time: {self.compute_time['aggr_peer'] / self.compute_cnt['aggr_peer']}")
                    + "\n"
                )
            if self.compute_cnt["sync"] > 0:
                f.write(content(f"totoal sync time: {self.compute_time['sync']} for one server") + "\n")
                f.write(content(f"average sync time: {self.compute_time['sync'] / self.compute_cnt['sync']}") + "\n")
            if self.compute_cnt["fedavg"] > 0:
                f.write(content(f"total fedavg time: {self.compute_time['fedavg']}") + "\n")
                f.write(
                    content(f"average fedavg time: {self.compute_time['fedavg'] / self.compute_cnt['fedavg']}") + "\n"
                )
