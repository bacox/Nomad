import collections
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from mobilefl.config import Config
from mobilefl.log_tools.logging_style import content, line
from mobilefl.models.aggregator import Aggregator
from mobilefl.utils import total_variation_distance


class Client:
    def __init__(
        self,
        config: Config,
        client_id: int,
        unique_id: int,
        server_id: int,
        location: Tuple[int, int],
        client_type: str,
        gaussian_mu: float = 0,
        vocab_size: int = 0,
        training_time: Optional[int] = None,
        server_latencies: Optional[List[float]] = None,
    ) -> None:
        self.config = config
        # self.client_id = client_id
        self.client_id = unique_id
        self.unique_id = unique_id
        self.location = location  # the location of clients on the map
        self.area: float = 0
        self.server = []  # the server_ids that it communicates with
        self.age: int = 0  # the age of the current model or model 1
        self.age1: int = 0  # the age of model 2
        self.training_time = training_time  # the training delay
        self.training_time_new = None
        self.server_latencies = server_latencies  # the latencies to the servers
        if not self.config.get("client_async"):
            self.age = -1
        self.update: dict = None  # type: ignore
        self.client_type: str = client_type  # slow, medium, fast, gaussian
        self.gaussian_mu: float = gaussian_mu
        self.serverid: int = -1
        self.flag: bool = False  # If client is trained under previous server: true, otherwise: false
        self.new_id: int = -1
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.client_dist: List[int] = None  # type: ignore
        self.label_count: Dict[int, int] = {}
        self.vocab_size = vocab_size
        self.dataset_name = self.config.get("dataset")

        self.loss_val: float = 0.0
        self.training_history: List[Tuple[float, float, float]] = []  # List of time and loss values, and learning rate

        self.local_delay: float = 0.0  # the local delay of the client

        self.move_ban: int = 0

        # device
        if self.config.get("cuda"):
            torch.cuda.manual_seed(0)
            self.device = torch.device(f"cuda:{self.config.get('cuda_to_use')}")
        else:
            self.device = torch.device("cpu")

        # initiate model
        self.model = Aggregator.get_model(self.dataset_name, vocab_size=vocab_size)
        self.num_updates = 0

        # # initiate 2 models for the client that will communicate with 2 servers
        # self.model1 = Aggregator.get_model(self.dataset_name, vocab_size=vocab_size)
        # self.model2 = Aggregator.get_model(self.dataset_name, vocab_size=vocab_size)
        # self.model1_dict: dict = None  # type: ignore
        # self.model2_dict: dict = None  # type: ignore
        # moving model to GPU is possible
        self.model.to(device=self.device)
        # self.model1.to(device=self.device)
        # self.model2.to(device=self.device)
        self.weight_decay: float = self.config.get("weight_decay")
        self.l2_mu: float = self.config.get("l2_mu")
        self.lr_start_value: float = self.config.get("local_lr")
        self.lr: float = self.config.get("local_lr")
        self.lr_bound: float = self.config.get("lr_bound")
        self.lr_decay_value: float = self.config.get("lr_decay")
        self.batch_size: int = self.config.get("batch_size")

        # plot the lr decay function

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.config.get("momentum_client"),
            weight_decay=self.weight_decay,
        )
        # weight decay is a regularization term, it is used to prevent overfitting

        self.compute_time: Dict[str, float] = collections.defaultdict(float)
        self.compute_cnt: Dict[str, int] = collections.defaultdict(int)
        if self.client_id == 0:
            self.lr_decay()

        if self.dataset_name == "wikitext2":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()  # type: ignore
            # self.criterion = F.nll_loss  # type: ignore
        # self.criterion = F.nll_loss

        # add server id to self.server_id
        self.server.append(server_id)

    def __str__(self) -> str:
        return f"{self.client_type} {self.client_id}"

    # def set_time(self):
    #     if self.type == "gaussian":
    #         mu = self.gaussian_mu
    #     else:
    #         mu = self.config.get(self.type + "_mu") * self.config.get("training_delay")
    #     self.training_time = max(0.1, np.random.normal(mu, 0.05 * mu))

    def set_dataloader(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.calculate_non_iid()

    def size_dataloader(self, train: bool = True) -> int:
        """
        Get the size of the dataloader.
        :param train: If True, get the size of the training set, otherwise get the size of the test set.
        :return: The size of the dataloader.
        """
        if train:
            return len(self.train_loader.dataset)  # type: ignore
        else:
            return len(self.test_loader.dataset)  # type: ignore

    def get_label_count_as_array(
        self, label_keys: list, train: bool = True, force_recalculate: bool = False
    ) -> np.ndarray:
        """
        Get the label count as a numpy array.
        :param train: If True, get the label count from the training set, otherwise from the test set.
        :param force_recalculate: If True, recalculate the label count even if it is already cached.
        :return: A numpy array of label counts.
        """
        label_count = self.get_data_label_count(train=train, force_recalculate=force_recalculate)
        # Get all unique labels from dataset
        return np.array([label_count.get(label, 0) for label in label_keys], dtype=float)

    def get_data_label_count(self, train: bool = True, force_recalculate: bool = False) -> Dict[int, int]:
        if self.label_count and not force_recalculate:
            return self.label_count
        if train:
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        labels = []
        for _batch_idx, (data, target) in enumerate(data_loader):
            labels.append(target)
        labels = torch.cat(labels, dim=0)
        label_values, counts = torch.unique(labels, return_inverse=False, return_counts=True)

        self.label_count = {label_values[i].item(): counts[i].item() for i in range(len(label_values))}
        return self.label_count

    def calculate_non_iid(self, global_label_count: Dict, force_recalculate: bool = False) -> float:
        # calculate the non-iidness of the dataset
        # For now we assume that we have classes or labels
        # Give error when the dataset is wikitext2
        if self.dataset_name == "wikitext2":
            print("Client %s: wikitext2 dataset does not have non-iidness" % self.client_id)
            raise ValueError("wikitext2 dataset does not have non-iidness")

        # calculate the non-iidness of the dataset
        # print("Client %s: calculating non-iidness" % self.id)

        # Compute the Total Variation Distance (TVD) between the client's label distribution
        # and the global label distribution
        all_label_keys = list(global_label_count.keys())

        if self.client_dist is None or force_recalculate:

            clients_label_count = self.get_data_label_count()

            # Make sure all label keys are present in the client's label count
            for key in all_label_keys:
                if key not in clients_label_count:
                    clients_label_count[key] = 0

            client_dist = [clients_label_count[x] for x in all_label_keys]
            # client_dist = np_normalize(client_dist)

            self.client_dist = client_dist

        global_dist = [global_label_count[x] / 16.0 for x in all_label_keys]
        # global_dist = np_normalize(global_dist)
        # client_dist[self.id] = 0
        # print(f"Client {self.client_id} global label count: {global_dist}")
        # print(f"Client {self.client_id} client label count: {self.client_dist}")

        # d = tvd(self.client_dist, global_dist)
        return total_variation_distance(self.client_dist, global_dist, all_label_keys)

    def calculate_server_latency(self, server_id: int) -> float:
        """
        Calculate the latency to the server based on the client's location and the server's location.
        :param server_id: The ID of the server.
        :return: The latency to the server.
        """
        if self.server_latencies is not None:
            return self.server_latencies[server_id]
        else:
            return 0

    def average_models(self) -> dict:
        averaged_model = {}
        for key in self.model1_dict.keys():
            averaged_model[key] = (self.model1_dict[key] + self.model2_dict[key]) / 2.0
        return averaged_model

    def lr_decay(self) -> None:
        # sigmoid decay
        sigmoid_bound = self.lr_bound / self.lr_start_value
        sigmoid = (1 - sigmoid_bound) / (
            1
            + np.exp(
                self.lr_decay_value
                * (
                    self.num_updates
                    - 0.7
                    * self.config.get("num_rounds")
                    // self.config.get("num_clients")  # @TODO: Why this value of 0.7?
                    - 4 / self.lr_decay_value
                )
            )
        ) + sigmoid_bound
        self.lr = sigmoid * self.lr_start_value
        # close the plot
        # print("lr: %f" % self.lr)

    def train(self, server_time: float, model: int = 0) -> bool:
        if self.config.get("num_servers") > 1:
            self.lr_decay()
        start_time = time.perf_counter()
        if model == 1:
            assert False, "model 1 is not implemented yet"
            self.model = self.model1
            del self.model1
            self.model1 = Aggregator.get_model(self.dataset_name, vocab_size=self.vocab_size)
            self.model1.to(device=self.device)

        elif model == 2:
            assert False, "model 2 is not implemented yet"
            self.model = self.model2
            del self.model2
            self.model2 = Aggregator.get_model(self.dataset_name, vocab_size=self.vocab_size)
            self.model2.to(device=self.device)
        # train 5 epochs

        lstm_trainig = False
        if self.dataset_name == "wikitext2":
            lstm_trainig = True

        # print(
        #     f"Client {self.client_id} from Server {self.server} started training on device {self.device} with model {model}."
        # )
        load_t = time.time()
        loss_val = 0.0
        pbar = tqdm(
            range(1, self.config.get("num_local_epochs") + 1), desc=f"Client {self.client_id} training", disable=True
        )
        # for epoch in range(1, self.config.get("num_local_epochs") + 1):
        for epoch in pbar:
            # print(f"Client {self.client_id} from Server {self.server} training epoch {epoch} on device {self.device}.")

            self.model.train()

            if lstm_trainig:
                hidden = self.model.init_hidden(self.batch_size, self.device)
            for batch_idx, batch in enumerate(self.train_loader):
                # self.optimizer.zero_grad()  # set the optimizer's gradient towards weight to 0, otherwise it will accumulat
                # inputs, target = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
                inputs, target = batch[0].to(self.device, non_blocking=True), batch[1].to(
                    self.device, non_blocking=True
                )

                if lstm_trainig:
                    hidden = self.model.detach_hidden(hidden)  # type: ignore
                    output, hidden = self.model(inputs, hidden)
                    output = output.reshape(inputs.shape[0] * inputs.shape[1], -1)
                    target = target.reshape(-1)
                else:
                    self.model.zero_grad()
                    output = self.model(inputs)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                loss_val = loss.item()
                # print(f"Loss value {loss_val} at epoch {epoch} on device {self.device}.")
            # pbar.set_postfix({"loss": loss.item()})

            # print(f"Client {self.client_id} got loss {loss.item()} at epoch {epoch} on device {self.device}.")
        # print(f"Number datapoints in train_loader: {len(self.train_loader.dataset)}")
        # exit()
        self.loss_val = loss_val

        self.training_history.append((server_time, loss_val, self.lr))
        end_load_t = time.time()
        _data_loading_time = end_load_t - load_t
        # print(f"\t\t::::>>\t\tData loading time: {data_loading_time:.2f} seconds")
        self.update = {k: v.clone().detach().to(device=self.device) for k, v in self.model.state_dict().items()}

        end_time = time.perf_counter()
        # print(
        #     f"Client {self.client_id} from Server {self.server} finished training in {end_time - start_time:.2f} seconds using device {self.device} and batch_idx: {batch_idx}."
        # )
        self.compute_time["train"] += end_time - start_time
        self.compute_cnt["train"] += 1
        self.num_updates += 1
        # if np.isnan(self.model.state_dict()["fc1.weight"][0][0].item()):
        #     return False

        del self.model
        del self.optimizer

        _pre_agg_time = time.time()
        # torch.cuda.empty_cache()
        self.model = Aggregator.get_model(self.dataset_name, vocab_size=self.vocab_size)
        self.model.to(device=self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.config.get("momentum_client"),
            weight_decay=self.weight_decay,
        )
        _endend_time = time.time()
        # print(f"\t\t::::>>\t\tNumber of batches: {batch_idx}")
        # print(f"\t\t::::>>\t\tData loading time: {data_loading_time:.2f} seconds")
        # print(f"\t\t::::>>\t\tModel reloading time: {endend_time - pre_agg_time:.2f} seconds")
        # print(f"\t\t::::>>\t\tTotal training time: {end_time - start_time:.2f} seconds")
        # print(f"\t\t::::>>\t\tTotal post training time: {end_load_t - endend_time:.2f} seconds")
        # print(f"\t\t::::>>\t\tTotal
        return True

    def test(self) -> None:
        # print("Client %s: testing..." % self.id)
        self.model.eval()
        test_loss = 0.0
        correct = 0
        if self.dataset_name == "wikitext2":
            total_test_cases = 0
        else:
            total_test_cases = self.size_dataloader(train=False)
        if self.dataset_name == "wikitext2":
            hidden = self.model.init_hidden(self.batch_size, self.device)
        with torch.no_grad():
            for batch in self.test_loader:
                if self.dataset_name == "wikitext2":
                    hidden = self.model.detach_hidden(hidden)  # type: ignore
                inputs, target = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
                if self.dataset_name == "wikitext2":
                    output, hidden = self.model(inputs, hidden)  # type: ignore
                    output = output.reshape(inputs.shape[0] * inputs.shape[1], -1)
                    target = target.reshape(-1)
                    total_test_cases += target.size(0)
                    # test_loss += self.criterion(output, target).item() * inputs.shape[1]
                else:
                    output = self.model(inputs)
                test_loss += self.criterion(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= total_test_cases
        perplexity = math.exp(test_loss)
        if np.isnan(test_loss):
            print("NAN")
        if self.dataset_name == "wikitext2":
            print(
                "Client %s: Average loss: %.4f, Accuracy: %d/%d (%.2f%%), Perplexity: %.2f"
                % (
                    self.client_id,
                    test_loss,
                    correct,
                    self.size_dataloader(train=False),
                    100.0 * correct / self.size_dataloader(train=False),
                    perplexity,
                )
            )
        else:
            print(
                "Client %s from Server %s: Average loss: %.4f, Accuracy: %d/%d (%.2f%%)"
                % (
                    self.client_id,
                    self.server,
                    test_loss,
                    correct,
                    self.size_dataloader(train=False),
                    100.0 * correct / self.size_dataloader(train=False),
                )
            )

    def receive_global_model(self, global_model: dict, model_age: int = -1, server_id: Optional[int] = None) -> None:
        # print(
        #     "-----------server {}, client{} of server{}, recieve_model, model age{}----------------".format(
        #         self.server, self.client_id, self.server[0], model_age
        #     )
        # )
        if self.config.get("avg_model") and len(self.server) > 1:
            if self.model1_dict is None:
                self.model1_dict = global_model
                self.age = model_age
            else:
                self.model2_dict = global_model
                self.model.load_state_dict(self.average_models())
                print(
                    "-----------client{}, recieve All 2 models, model_age_before:{}, model_age_now:{}----------------".format(
                        self.client_id, self.age, model_age
                    )
                )
                if model_age > self.age:
                    print("-------------update new age----------------")
                    self.age = model_age
        elif self.config.get("alternate_new") and len(self.server) > 1:
            assert False, "alternate_new is not implemented yet"
            # if server_id == self.server[0]:
            #     print(f"client {self.client_id} receive a model from {server_id}")
            #     self.model1.load_state_dict(global_model)
            #     self.age = model_age
            # else:
            #     print(f"client {self.new_id} receive a model from {server_id}")
            #     self.model2.load_state_dict(global_model)
            #     self.age1 = model_age
        else:
            self.model.load_state_dict(global_model)
            self.age = model_age

    def report(self) -> None:
        # how many time it has participated in training
        # computing time of the local training
        if self.compute_cnt["train"] == 0:
            return

        with open(
            f"./results/{self.config.get('result_file')}/{self.config.get('name')}.txt",
            "a",
        ) as f:
            f.write(line("Client Information"))
            f.write(
                content(
                    f"Client {self.client_id} from Server {self.server} contributed {self.compute_cnt['train']} updates"
                )
                + "\n"
            )
            f.write(
                content(f"average training time is {self.compute_time['train'] / self.compute_cnt['train']}") + "\n"
            )
