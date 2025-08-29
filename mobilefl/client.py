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
        self.client_id = unique_id
        self.unique_id = unique_id
        self.location = location  
        self.area: float = 0
        self.server = []  
        self.age: int = 0  
        self.age1: int = 0  
        self.training_time = training_time  
        self.training_time_new = None
        self.server_latencies = server_latencies  
        if not self.config.get("client_async"):
            self.age = -1
        self.update: dict = None  
        self.client_type: str = client_type  
        self.gaussian_mu: float = gaussian_mu
        self.serverid: int = -1
        self.flag: bool = False  
        self.new_id: int = -1
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.client_dist: List[int] = None  
        self.label_count: Dict[int, int] = {}
        self.vocab_size = vocab_size
        self.dataset_name = self.config.get("dataset")
        self.loss_val: float = 0.0
        self.training_history: List[Tuple[float, float, float]] = []  
        self.local_delay: float = 0.0  
        self.move_ban: int = 0
        if self.config.get("cuda"):
            torch.cuda.manual_seed(0)
            self.device = torch.device(f"cuda:{self.config.get('cuda_to_use')}")
        else:
            self.device = torch.device("cpu")
        self.model = Aggregator.get_model(self.dataset_name, vocab_size=vocab_size)
        self.num_updates = 0
        self.model.to(device=self.device)
        self.weight_decay: float = self.config.get("weight_decay")
        self.l2_mu: float = self.config.get("l2_mu")
        self.lr_start_value: float = self.config.get("local_lr")
        self.lr: float = self.config.get("local_lr")
        self.lr_bound: float = self.config.get("lr_bound")
        self.lr_decay_value: float = self.config.get("lr_decay")
        self.batch_size: int = self.config.get("batch_size")
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.config.get("momentum_client"),
            weight_decay=self.weight_decay,
        )
        self.compute_time: Dict[str, float] = collections.defaultdict(float)
        self.compute_cnt: Dict[str, int] = collections.defaultdict(int)
        if self.client_id == 0:
            self.lr_decay()
        if self.dataset_name == "wikitext2":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()  
        self.server.append(server_id)
    def __str__(self) -> str:
        return f"{self.client_type} {self.client_id}"
    def set_dataloader(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
    def size_dataloader(self, train: bool = True) -> int:
        """
        Get the size of the dataloader.
        :param train: If True, get the size of the training set, otherwise get the size of the test set.
        :return: The size of the dataloader.
        """
        if train:
            return len(self.train_loader.dataset)  
        else:
            return len(self.test_loader.dataset)  
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
        if self.dataset_name == "wikitext2":
            print("Client %s: wikitext2 dataset does not have non-iidness" % self.client_id)
            raise ValueError("wikitext2 dataset does not have non-iidness")
        all_label_keys = list(global_label_count.keys())
        if self.client_dist is None or force_recalculate:
            clients_label_count = self.get_data_label_count()
            for key in all_label_keys:
                if key not in clients_label_count:
                    clients_label_count[key] = 0
            client_dist = [clients_label_count[x] for x in all_label_keys]
            self.client_dist = client_dist
        global_dist = [global_label_count[x] / 16.0 for x in all_label_keys]
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
        sigmoid_bound = self.lr_bound / self.lr_start_value
        sigmoid = (1 - sigmoid_bound) / (
            1
            + np.exp(
                self.lr_decay_value
                * (
                    self.num_updates
                    - 0.7
                    * self.config.get("num_rounds")
                    // self.config.get("num_clients")  
                    - 4 / self.lr_decay_value
                )
            )
        ) + sigmoid_bound
        self.lr = sigmoid * self.lr_start_value
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
        lstm_trainig = False
        if self.dataset_name == "wikitext2":
            lstm_trainig = True
        load_t = time.time()
        loss_val = 0.0
        pbar = tqdm(
            range(1, self.config.get("num_local_epochs") + 1), desc=f"Client {self.client_id} training", disable=True
        )
        for epoch in pbar:
            self.model.train()
            if lstm_trainig:
                hidden = self.model.init_hidden(self.batch_size, self.device)
            for batch_idx, batch in enumerate(self.train_loader):
                inputs, target = batch[0].to(self.device, non_blocking=True), batch[1].to(
                    self.device, non_blocking=True
                )
                if lstm_trainig:
                    hidden = self.model.detach_hidden(hidden)  
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
        self.loss_val = loss_val
        self.training_history.append((server_time, loss_val, self.lr))
        end_load_t = time.time()
        _data_loading_time = end_load_t - load_t
        self.update = {k: v.clone().detach().to(device=self.device) for k, v in self.model.state_dict().items()}
        end_time = time.perf_counter()
        self.compute_time["train"] += end_time - start_time
        self.compute_cnt["train"] += 1
        self.num_updates += 1
        del self.model
        del self.optimizer
        _pre_agg_time = time.time()
        self.model = Aggregator.get_model(self.dataset_name, vocab_size=self.vocab_size)
        self.model.to(device=self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.config.get("momentum_client"),
            weight_decay=self.weight_decay,
        )
        _endend_time = time.time()
        return True
    def test(self) -> None:
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
                    hidden = self.model.detach_hidden(hidden)  
                inputs, target = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
                if self.dataset_name == "wikitext2":
                    output, hidden = self.model(inputs, hidden)  
                    output = output.reshape(inputs.shape[0] * inputs.shape[1], -1)
                    target = target.reshape(-1)
                    total_test_cases += target.size(0)
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
        else:
            self.model.load_state_dict(global_model)
            self.age = model_age
    def report(self) -> None:
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
