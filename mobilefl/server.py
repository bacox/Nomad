from __future__ import annotations
import random
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)
from mobilefl.ageToken import AgeToken
from mobilefl.data.data import Data
from mobilefl.data.subset import CustomSubset
from mobilefl.event_system import EventSystem
from mobilefl.types import EventHistory, MEvent
from mobilefl.utils import compute_weighted_tvd, np_normalize, sum_dicts
if TYPE_CHECKING:
    from mobilefl.client import Client
import collections
import heapq
import math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mobilefl.color import colors
from mobilefl.config import Config
from mobilefl.log_tools.logger import Logger
from mobilefl.models.aggregator import Aggregator
np.random.seed(0)
torch.manual_seed(0)
class Server:
    def __init__(
        self,
        config: Config,
        server_id: int,
        test_loader: DataLoader,
        logger: Logger,
        vocab_size: int = 0,
        location: Tuple[int, int] = (0, 0),
    ) -> None:
        self.config: Config = config
        self.server_id = server_id
        self.aggregator = Aggregator(config, vocab_size=vocab_size)
        self.test_loader = test_loader
        self.logger: Logger = logger
        self.print_flag = False
        self.leader: Server = None  
        self.hier_period = 100000
        self.age = 0  
        self.last_age = 0
        self.clients: Dict[int, Client] = {}
        self.history = {}
        self.sum_staleness = 0
        self.tmp_sum_staleness = 0
        self.area = 0  
        self.location = location  
        self.clientsSet: List[Client] = []  
        self.clients_num = 0  
        self.client_train_datasets: List[CustomSubset] = None  
        self.client_test_datasets: List[CustomSubset] = None  
        self.server_level_speed: float = 0  
        self.train_dataset_server_level: CustomSubset = None  
        self.label_count: Dict[int, int] = {}
        self.popped_heap_item = {}
        self.share_heap_item = {}
        self.push_heap_item = {}
        self.pop_client_id = None
        self.share_client_id = None
        self.recieved_moved = False
        self.sync_round = 1
        self.sent_peer = False
        self.update_buffer = []
        self.peer_buffer = []
        self.server_buffer = []
        self.slowest_sync = 0
        self.servers: List[Server] = []
        self.heap = []
        self.model_id = 0
        self.cur_time = 0
        self.information = ""
        self.num_updates = 0
        self.period: int = 0
        self.token_sequence = 0
        self.token: AgeToken = None  
        self.broadcasting = False
        self.token_broadcast_cnt = collections.defaultdict(int)
        self.current_total_weighted_tvd: float = 0.0
        self.updates = {}  
        if self.config.get("cuda"):
            torch.cuda.manual_seed(0)
            self.device = torch.device(f"cuda:{self.config.get('cuda_to_use')}")
        else:
            self.device = "cpu"
        self.criterion = F.nll_loss
        self.frequency_matrix: np.ndarray = None  
        self.global_label_count = {}  
        self.event_history: EventHistory
    def latency_stats(self) -> Dict[str, Union[float, List[float]]]:
        assert self.frequency_matrix is not None, "Frequency matrix is not set."
        latencies = [self.frequency_matrix[client.client_id, self.server_id] for client in self.clients.values()]
        return {
            "avg_latency": np.mean(latencies) if latencies else 0,
            "max_latency": np.max(latencies) if latencies else 0,
            "min_latency": np.min(latencies) if latencies else 0,
            "std_latency": np.std(latencies) if latencies else 0,
            "abs_diff": np.abs(latencies - np.mean(latencies)) if latencies else [],
            "latencies": latencies,
        }
    def set_event_history(self, event_history: EventHistory) -> None:
        """
        Sets the event history for the server.
        :param event_history: An instance of EventHistory to store events.
        """
        self.event_history = event_history
    def set_servers(self, servers: List[Server]) -> None:
        self.servers = servers
        self.shuffled_servers = self.servers.copy()
    def add_server(self, server: Server) -> None:
        """
        Adds a server to the list of servers.
        :param server: An instance of Server to be added.
        """
        self.servers.append(server)
        self.shuffled_servers = self.servers.copy()
        print(f"Server {self.server_id} added Server {server.server_id} to its list of servers.")
    def set_clients(self, clients: list[Client]):
        print(f"Server {self.server_id} set clients with {len(clients)} clients")
        print(f"Server {self.server_id} clients: {[client.client_id for client in clients]}")
        for client in clients:
            self.clients[client.client_id] = client
            self.history[client.client_id] = 0
    def get_data_label_count(self, force_recalculate: bool = False) -> Dict[int, int]:
        if self.label_count and not force_recalculate:
            return self.label_count
        data_loader = self.test_loader
        labels = []
        for _batch_idx, (data, target) in enumerate(data_loader):
            labels.append(target)
        labels = torch.cat(labels, dim=0)
        label_values, counts = torch.unique(labels, return_inverse=False, return_counts=True)
        self.label_count = {label_values[i].item(): counts[i].item() for i in range(len(label_values))}
        return self.label_count
    def collect_client_label_count(self) -> Tuple[np.ndarray, List[int]]:
        """
        Collects the label count from all clients and returns a numpy array of label counts and a list of labels.
        """
        lk = list(self.get_data_label_count().keys())
        all_counts = []
        client_ids = list(self.clients.keys())
        for client in self.clients.values():
            client_label_count = client.get_label_count_as_array(lk)
            all_counts.append(client_label_count)
        server_lc = np.vstack(all_counts) if all_counts else np.zeros((1, len(lk)))
        return server_lc, client_ids
    def compute_server_tvd(self, global_label_count: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, List[int]]:
        latencies = self.get_clients_latencies()
        server_label_count, client_ids = self.collect_client_label_count()
        total_weighted_tvd, client_distributions, client_weights, client_ids = compute_weighted_tvd(
            latencies, client_ids, server_label_count, global_label_count
        )
        self.current_total_weighted_tvd = total_weighted_tvd
        return total_weighted_tvd, client_distributions, client_weights, client_ids
    def tvd_and_propose_client(self, global_label_count: np.ndarray):
        latencies = self.get_clients_latencies()
        server_label_count, client_ids = self.collect_client_label_count()
        total_weighted_tvd, client_distributions, client_weights, client_ids = self.compute_server_tvd(
            global_label_count
        )
        self.current_total_weighted_tvd = total_weighted_tvd
        min_weighted_tvd = float("inf")
        client_to_remove = {
            "client_id": -1,
            "client_distribution": None,
            "client_latency": None,
        }
        for client in self.clients.values():
            if client.move_ban > 0:
                print(f"Client {client.client_id} is banned from moving. Skipping.")
                continue
            client_index = client_ids.index(client.client_id)
            client_distribution = client_distributions[client_index]
            client_latency = latencies[client_index]
            new_total_weighted_tvd = self.estimate_augmented_tvd(
                global_label_count,
                client.client_id,
                client_distribution,
                client_latency,
                action="remove",
            )
            tvd_diff = total_weighted_tvd - new_total_weighted_tvd
            if tvd_diff > 0 and new_total_weighted_tvd < min_weighted_tvd:
                min_weighted_tvd = new_total_weighted_tvd
                client_to_remove["client_id"] = client.client_id
                client_to_remove["client_distribution"] = client_distribution
                client_to_remove["client_latency"] = client_latency
        return total_weighted_tvd, new_total_weighted_tvd, client_to_remove
    def estimate_augmented_tvd(
        self,
        global_label_count: np.ndarray,
        client_id: int,
        client_distribution: np.ndarray,
        latency: float,
        action="add",
    ):
        latencies = self.get_clients_latencies()
        server_label_count, client_ids = self.collect_client_label_count()
        if action == "add":
            new_server_label_count = np.vstack((server_label_count, client_distribution))
            latencies = np.append(latencies, latency)
        elif action == "remove":
            client_index = client_ids.index(client_id)
            new_server_label_count = np.delete(server_label_count, client_index, axis=0)
            latencies = np.delete(latencies, client_index)
        else:
            raise ValueError(f"Action must be either 'add' or 'remove' but got {action}")
        new_total_weighted_tvd, _, _, _ = compute_weighted_tvd(
            latencies, client_ids, new_server_label_count, global_label_count
        )
        return new_total_weighted_tvd
    def calculate_server_noniidness(self, global_label_count: dict) -> float:
        client_non_iidness = []
        for client in self.clients.values():
            tvd_value = client.calculate_non_iid(global_label_count)
            client_frequency = 1.0
            client_effective_non_iid = client_frequency * tvd_value
            client_non_iidness.append(client_effective_non_iid)
        return np.mean(client_non_iidness)
    def get_clients_latencies(self, mode: str = "list") -> Union[Dict[str, float], np.ndarray]:
        """
        Returns a dictionary of client IDs and their latencies to the server.
        """
        latencies = {}
        latency_list = []
        for client in self.clients.values():
            latencies[client.client_id] = self.frequency_matrix[client.client_id, self.server_id]
            latency_list.append(latencies[client.client_id])
        if mode == "list":
            return np.array(latency_list)
        elif mode == "dict":
            return latencies
        raise ValueError(f"Mode must be either 'list' or 'dict' but got {mode}")
    def calculate_effective_non_iidness(
        self,
    ) -> float:
        client_latencies = self.get_clients_latencies()
        print(f"Server {self.server_id} client latencies: {client_latencies}")
        client_latencies = np_normalize(client_latencies)
        print(f"Server {self.server_id} normalized client latencies: {client_latencies}")
        server_label_count = []
        for client, client_latency in zip(self.clients.values(), client_latencies):
            lc = np.zeros((1, len(client.train_loader.dataset.classes)))
            clc = client.get_data_label_count()
            for label, count in clc.items():
                lc[0][label] = count
            elc = lc * client_latency
            print(f"Client {client.client_id} effective label count: {elc} :: {clc}")
            server_label_count.append(elc)
        server_label_count = np.sum(server_label_count, axis=0)
        print(f"Server {self.server_id} effective label count: {server_label_count} :: {self.global_label_count}")
        print(f"Against old label count: {self.get_all_client_label_count()}")
    def calculate_candidate_client(self) -> int:
        _label_keys = self.train_dataset_server_level.classes
    def get_all_client_label_count(self, reference_labels: List[int] = None) -> Union[Dict[int, int], np.ndarray]:
        all_labels = {}
        _client_ids = list(self.clients.keys())
        for client in self.clients.values():
            lc = client.get_data_label_count()
            all_labels = sum_dicts(all_labels, lc)
        if reference_labels is not None:
            result = []
            for label in reference_labels:
                if label in list(all_labels.keys()):
                    result.append(all_labels[label])
                else:
                    result.append(0)
            return np.array(result)
        return all_labels
    def calculate_clients_speed(self, moving: bool) -> np.ndarray:
        previous_id = list(self.clients.values())[-1].client_id
        if moving:
            N = self.clients_num + 1
            self.clients[self.clients_num].client_id = self.clients_num
        else:
            N = self.clients_num
        M = len(list(self.clients.values())[0].train_loader.dataset.classes)
        A = np.zeros((M, N))
        B = np.zeros(M)
        for i in range(N):
            client = list(self.clients.values())[i]
            labels = client.train_loader.dataset.targets
            labels = np.round(labels).astype(int)
            label_counts = Counter(labels)
            for label, count in label_counts.items():
                A[label][i] = count * (1 + 2 * self.get_comm_delay("client", client.client_id) + 2)
        all_labels = self.train_dataset_server_level.targets
        all_labels_count = Counter(all_labels)
        for label, count in all_labels_count.items():
            B[label] = count * self.server_level_speed
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        if moving:
            self.clients[self.clients_num].client_id = previous_id
        for i in range(N):
            client = list(self.clients.values())[i]
            if i == N - 1:
                print(
                    f"Calculate that: Server {self.server_id}: client {client.client_id}'s training time (speed) is {X[i]} "
                )
        for i in range(N):
            if X[i] <= 0:
                return X
            else:
                client = list(self.clients.values())[i]
                X[i] = X[i] + 2 * self.get_comm_delay("client", client.client_id) + 2
                return X
    def set_clients_speed(self):
        N = self.clients_num
        client_ids = list(self.clients.keys())
        print(f"Server {self.server_id} clients ids: {client_ids}")
        baseline = False
        for idx, i in enumerate(client_ids):
            X = self.frequency_matrix[:, self.server_id]
            self.clients[i].training_time = X[idx]
            if i == N - 1:
                print(
                    f"Server {self.server_id}: client {self.clients[i].client_id}'s training time (speed) is {X[idx]} "
                )
                if i in self.push_heap_item:
                    heapq.heappush(self.heap, (self.push_heap_item[i][0], i))
                else:
                    if self.config.get("move_a_client"):
                        heapq.heappush(
                            self.heap,
                            (X[idx] + 2 * self.get_comm_delay("client", i) + 2, i),
                        )
                    else:
                        baseline = True
                        heapq.heappush(
                            self.heap,
                            (X[idx] + 2 * self.get_comm_delay("client", i) + 2, i),
                        )
            else:
                heapq.heappush(self.heap, (X[idx] + 2 * self.get_comm_delay("client", i) + 2, i))
                self.config.write_config(self.updates)
            key = f"speed_of_server_{self.server_id}_client_{i}"
            self.updates[key] = X[idx]
        if baseline:
            self.config.write_config(self.updates)
        return X
    def heappop_condition(self, condition: Any) -> Any:
        """Pop the item off the heap that satisfies the condition."""
        for index, item in enumerate(self.heap):
            if condition(item):
                self.heap[index] = self.heap[-1]
                self.heap.pop()
                if index < len(self.heap):
                    heapq._siftup(self.heap, index)
                    heapq._siftdown(self.heap, 0, index)
                return item
        raise KeyError("No element found that satisfies the condition")
    def set_clients_dataloader(
        self,
        data: Data,
        clients_num: int,
        train_dataset_server_level: CustomSubset,
        test_dataset_server_level: CustomSubset,
        out_path: Path,
    ) -> None:
        train_loaders, test_loaders = data.get_client_data_loaders(
            self, clients_num, train_dataset_server_level, test_dataset_server_level, out_path=out_path
        )
        i = 0
        for key in self.clients:
            self.clients[key].set_dataloader(train_loaders[i], test_loaders[i])
            i = i + 1
    def get_comm_delay(self, who, node_id):
        region_matrix = {
            0: {0: 0, 1: 51.37, 2: 130.57, 3: 201.35, 4: 201.41, 5: 152.90},
            1: {0: 51.37, 1: 0, 2: 109.36, 3: 157.12, 4: 225.84, 5: 110.73},
            2: {0: 130.57, 1: 109.36, 2: 0, 3: 199, 4: 267.46, 5: 139.4},
            3: {0: 201.35, 1: 157.12, 2: 199, 3: 0, 4: 78.61, 5: 79.4},
            4: {0: 201.41, 1: 225.84, 2: 267.46, 3: 78.61, 4: 0, 5: 148.13},
            5: {0: 152.90, 1: 110.73, 2: 139.4, 3: 79.4, 4: 148.13, 5: 0},
        }
        if node_id == "max":
            max_comm = -1
            for server in self.servers:
                max_comm = max(max_comm, region_matrix[server.area][self.area])
            return max_comm
        if who == "server":
            if self.config.get("result_file") == "mtest":
                return 150
            server = self.servers[node_id]
            return region_matrix[server.area][self.area]
        elif who == "client":
            client = self.clients[node_id]
            distance = self.calculateDistance(self.location, client)
            max_comm_delay = 267.46
            max_distance = 270.81  
            mapped_delay = (distance / max_distance) * max_comm_delay
            return mapped_delay
    def calculateDistance(self, location: Tuple[int, int], client: Client) -> float:
        distance = np.sqrt((client.location[0] - location[0]) ** 2 + (client.location[1] - location[1]) ** 2)
        return distance
    def set_server_location(self, coordinates: np.ndarray) -> None:
        self.location = coordinates[self.area]
    def print_assigned_clients(self):
        print(f"Server {self.server_id} assigned clients: {[client.client_id for client in self.clients.values()]}")
    def get_assigned_client_ids(self) -> List[int]:
        """
        Returns a list of client IDs assigned to this server.
        """
        return list(self.clients.keys())
    def update_local_async_global_sync(
        self, current_round: int = 0, event_system: Optional[EventSystem] = None
    ) -> bool:
        """
        1. use a minheap to decide the next client to update
        2. The heap item is a tuple (update_time, client_id)
        Process:
        Find the next client to update;
        Pop the client from to the heap with the new update time;
        If the server does not reach the synchronization period, update and return True;
        If the server reaches the synchronization period, it should wait and return False;
        """
        _s_time = time.time()
        log_prefix = f"[Server {self.server_id}] :: "
        while len(self.update_buffer) < self.config.get("aggregation_buffer_size"):
            next_update_time, client_id = self.heap[0]
            if (
                self.sync_round > 0 and max(self.cur_time, next_update_time) > self.sync_round * self.period
            ):  
                if not self.sent_peer:
                    self.sent_peer = True
                    self.send_peer()
                    print(f"{log_prefix}Server {self.server_id} waiting")
                return False  
            next_time, client_id = heapq.heappop(self.heap)
            self.cur_time = max(self.cur_time, next_time) + self.config.get("fedasync_delay")
            client = self.clients[client_id]
            assert event_system is not None, "Event system must be provided for async updates"
            start_time = time.time()
            if client.train(self.cur_time):
                _client_loss_val = client.loss_val
                self.update_buffer.append((client.update, client.client_id, self.age - client.age))
                _t_time = time.time() - start_time
            else:
                print(f"{log_prefix}Client {client_id} failed to train, skipping...")
            self.sum_staleness += self.age - client.age
            self.tmp_sum_staleness += self.age - client.age  
            assert client.training_time is not None, f"Client {client_id} training time is None"
            heapq.heappush(
                self.heap,
                (
                    self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client.client_id) + 2,
                    client_id,
                ),
            )
        self.aggregator.aggregate(self.update_buffer, cosine=False, aggregation_method=self.config.get("agr"))
        client_information = ""
        for _, clientid, _ in self.update_buffer:
            client_information += f"{self.clients[clientid]} - {self.clients[clientid].lr:.3f}"
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | {client_information} | Staleness {self.tmp_sum_staleness / len(self.update_buffer):.1f}"
        self.test_acc()  
        self.age += 1
        self.num_updates += 1
        self.deliver_model()
        for item in self.update_buffer:
            del item
        self.update_buffer = []
        torch.cuda.empty_cache()
        self.tmp_sum_staleness = 0
        return True
    def update_local_sync(self):
        """
        Poll each client to update
        """
        selected_client_ids = []
        if self.config.get("client_selection") == "random":
            rounded_val = round(len(self.clientsSet) * self.config.get("client_fraction_per_round"))
            client_keys = list(self.clients.keys())
            print(f"{client_keys=}")
            selected_client_ids = random.sample(client_keys, rounded_val)
            if self.num_updates == 0:
                print(f"{len(selected_client_ids)} clients are selected")
        slowest = -1
        for client_id in selected_client_ids:
            client = self.clients[client_id]
            duration = (
                2 * self.get_comm_delay("client", client_id) + 2 + client.training_time
            )  
            slowest = max(duration, slowest)
            if client.train(self.cur_time):
                self.event_history.add_event(MEvent(self.cur_time, f"client-{client_id}", "train"))
                self.update_buffer.append((client.update, client_id))
            else:
                print(f"Client {client_id} failed to train, skipping...")
        self.aggregator.aggregate(self.update_buffer, cosine=False, aggregation_method=self.config.get("agr"))
        self.cur_time += slowest + self.config.get("fedavg_delay")
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age}"
        self.test_acc()
        self.deliver_model()
        self.num_updates += len(self.update_buffer)
        self.update_buffer = []
        self.age += 1
    def update_local_async_global_async(self):
        while not isinstance(self.heap[0][1], int):  
            arr_time, peer_state_dict, peer_age, token_id, init_id, sender_id = heapq.heappop(self.heap)
            self.token_receive_broadcast(arr_time, peer_state_dict, peer_age, token_id, init_id, sender_id)
        while len(self.update_buffer) < int(self.config.get("aggregation_buffer_size")):
            next_time, client_id = heapq.heappop(self.heap)
            self.cur_time = max(self.cur_time, next_time) + float(self.config.get("fedasync_delay"))
            client = self.clients[client_id]
            if self.config.get("avg_model"):
                if len(client.server) == 2 and client.server[1] == self.server_id:
                    if not client.flag:
                        return
                    else:
                        if client.train(self.cur_time):
                            self.update_buffer.append((client.update, client_id, self.age - client.age))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        self.sum_staleness += self.age - client.age
                        self.tmp_sum_staleness += self.age - client.age  
                        heapq.heappush(
                            self.heap,
                            (
                                self.cur_time
                                + client.training_time_new
                                + 2 * self.get_comm_delay("client", client_id)
                                + 2,
                                client_id,
                            ),
                        )
                        client.flag = False
                elif (
                    len(client.server) == 2 and client.server[0] == self.server_id
                ):  
                    if client.flag:
                        return
                    else:
                        if client.train(self.cur_time):
                            self.update_buffer.append((client.update, client_id, self.age - client.age))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        self.sum_staleness += self.age - client.age
                        self.tmp_sum_staleness += self.age - client.age  
                        heapq.heappush(
                            self.heap,
                            (
                                self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                                client_id,
                            ),
                        )
                        client.flag = True
                else:
                    if client.train(self.cur_time):
                        self.update_buffer.append((client.update, client_id, self.age - client.age))
                    else:
                        print(f"Client {client_id} failed to train, skipping...")
                    self.sum_staleness += self.age - client.age
                    self.tmp_sum_staleness += self.age - client.age  
                    heapq.heappush(
                        self.heap,
                        (
                            self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                            client_id,
                        ),
                    )
            elif self.config.get("alternate_new"):
                if len(client.server) == 2 and client.server[1] == self.server_id:
                    if not client.flag:
                        return
                    else:
                        if client.train(self.cur_time, model=2):
                            self.update_buffer.append((client.update, client_id, self.age - client.age1))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        self.sum_staleness += self.age - client.age1
                        self.tmp_sum_staleness += self.age - client.age1  
                        heapq.heappush(
                            self.heap,
                            (
                                self.cur_time
                                + client.training_time_new
                                + 2 * self.get_comm_delay("client", client_id)
                                + 2,
                                client_id,
                            ),
                        )
                        client.flag = False
                elif (
                    len(client.server) == 2 and client.server[0] == self.server_id
                ):  
                    if client.flag:
                        return
                    else:
                        if client.train(self.cur_time, model=1):
                            self.update_buffer.append((client.update, client_id, self.age - client.age))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        self.sum_staleness += self.age - client.age
                        self.tmp_sum_staleness += self.age - client.age  
                        heapq.heappush(
                            self.heap,
                            (
                                self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                                client_id,
                            ),
                        )
                        client.flag = True
                else:
                    if client.train(
                        self.cur_time,
                    ):
                        self.update_buffer.append((client.update, client_id, self.age - client.age))
                    else:
                        print(f"Client {client_id} failed to train, skipping...")
                    self.sum_staleness += self.age - client.age
                    self.tmp_sum_staleness += self.age - client.age  
                    heapq.heappush(
                        self.heap,
                        (
                            self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                            client_id,
                        ),
                    )
            else:
                if client.train(self.cur_time):
                    self.update_buffer.append((client.update, client_id, self.age - client.age))
                else:
                    print(f"Client {client_id} failed to train, skipping...")
                self.sum_staleness += self.age - client.age
                self.tmp_sum_staleness += self.age - client.age  
                heapq.heappush(
                    self.heap,
                    (
                        self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                        client_id,
                    ),
                )
        self.aggregator.aggregate(self.update_buffer, cosine=False, aggregation_method=self.config.get("agr"))
        client_information = ""
        for _, clientid, _ in self.update_buffer:
            client_information += f"{self.clients[clientid]} - {self.clients[clientid].lr:.3f}"
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | {client_information} | Staleness {self.tmp_sum_staleness / len(self.update_buffer):.1f}"
        self.test_acc()  
        self.age += 1
        self.num_updates += 1
        self.deliver_model()
        self.update_buffer = []
        self.tmp_sum_staleness = 0
        if self.token is not None and not self.broadcasting:
            self.token.age_dict[self.server_id] = self.age
            if (
                max(self.token.age_dict.values()) - min(self.token.age_dict.values())
                >= round(len(self.clients) * float(self.config.get("token_threshold")))
                or self.too_long()
            ):
                print(
                    colors[self.server_id % len(colors)],
                    f"Server {self.server_id} | {self.cur_time:.2f}: token {self.token.sequence_id} broadcasted.",
                    colors[-1],
                )
                self.broadcasting = True  
                self.token_broadcast(token_id=self.token.sequence_id)
            else:
                self.token_send_next()
    def add_client(self, client: Client, next_time=0) -> None:
        self.clientsSet.append(client)
        if self.server_id not in client.server:
            client.server.append(self.server_id)
        self.set_clients([client])
        assert isinstance(client.client_id, int), "Client ID must be an integer."
        heapq.heappush(self.heap, (next_time, client.client_id))
    def remove_client(self, client_id: int) -> Tuple[Client, float]:
        if client_id not in self.clients:
            raise ValueError(f"Client ID {client_id} not found in server {self.server_id}.")
        next_time, client_id = self.pop_item_with_client_id(self.heap, client_id)
        time_remaining = next_time - self.cur_time
        if time_remaining < 0:
            print(f"Warning: Negative time remaining for client {client_id} in server {self.server_id}. Setting to 0.")
            time_remaining = 0  
        assert time_remaining >= 0, "Time remaining should be non-negative."
        client = self.clients.pop(client_id)
        self.clientsSet.remove(client)
        if self.server_id in client.server:
            client.server.remove(self.server_id)
        self.update_buffer = [item for item in self.update_buffer if item[1] != client_id]
        if client_id in self.history:
            del self.history[client_id]
        self.clients_num -= 1
        return client, next_time
    def too_long(self) -> bool:
        if self.age - self.last_age > int(self.config.get("update_per_sync")):
            return True
        else:
            return False
    def pop_item_with_client_id(self, heap: List[Any], client_id: int) -> Optional[Tuple[float, str]]:
        for index, heap_obj in enumerate(heap):
            if len(heap_obj) != 2:
                continue
            delays, c_id = heap_obj
            if c_id == client_id:
                removed_item = heap.pop(index)
                heapq.heapify(heap)
                return removed_item
        return None
    def test_acc(self, log: bool = True) -> None:
        self.aggregator.model.eval()
        test_loss = 0
        total_correct = 0
        if str(self.config.get("dataset")) == "wikitext2":
            total_test_cases = 0
        else:
            total_test_cases = len(self.test_loader.dataset)
        if str(self.config.get("dataset")) == "wikitext2":
            hidden = self.aggregator.model.init_hidden(self.config.get("batch_size"), self.device)
        with torch.no_grad():
            for batch in self.test_loader:
                if str(self.config.get("dataset")) == "wikitext2":
                    hidden = self.aggregator.model.detach_hidden(hidden)
                inputs, target = Variable(batch[0]).to(self.device), Variable(batch[1]).to(self.device)
                if self.config.get("dataset") == "wikitext2":
                    output, hidden = self.aggregator.model(inputs, hidden)
                    output = output.reshape(inputs.shape[0] * inputs.shape[1], -1)
                    target = target.reshape(-1)
                    output = F.log_softmax(output, dim=1)
                    total_test_cases += target.size(0)
                else:
                    output = self.aggregator.model(inputs)
                test_loss += self.criterion(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= total_test_cases
        if str(self.config.get("dataset")) == "wikitext2":
            perplexity = math.exp(test_loss)
        else:
            perplexity = 0.0
        accuracy = total_correct / total_test_cases
        queue_len = self.get_queue_len()
        if self.print_flag or self.num_updates % 100 == 0:
            if self.config.get("dataset") == "wikitext2":
                self.information += f"|Perplexity: {perplexity:.2f} | Accuracy: {total_correct}/{total_test_cases} ({100 * accuracy:.2f}%) | Queue Length: {queue_len}"
            else:
                self.information += f"| Accuracy: {total_correct}/{total_test_cases} ({100 * accuracy:.2f}%) | Queue Length: {queue_len}"
            if self.current_total_weighted_tvd > 0:
                self.information += f" | TVD: {self.current_total_weighted_tvd:.4f}"
            print(colors[self.server_id % len(colors)], self.information, colors[-1])
        self.information = ""
        try:
            server_tvd = self.calculate_server_noniidness(self.global_label_count)
        except Exception as e:
            print(f"Error calculating server non-iidness: {e}")
            server_tvd = 0.0
        latency_status = self.latency_stats()
        avg_latency = float(latency_status["avg_latency"])
        max_latency = float(latency_status["max_latency"])
        min_latency = float(latency_status["min_latency"])
        std_latency = float(latency_status["std_latency"])
        if log:
            if not self.config.get("dataset") == "wikitext2":
                data = {
                    "id": self.server_id,
                    "acc": accuracy,
                    "q_len": queue_len,
                    "time": self.cur_time,
                    "tvd": server_tvd,
                    "avg_latency": avg_latency,
                    "max_latency": max_latency,
                    "min_latency": min_latency,
                    "std_latency": std_latency,
                }
                if not self.config.get("client_async") and not self.config.get("server_async"):  
                    self.logger.log(
                        data,
                        num_fedavg=round(len(self.clientsSet) * self.config.get("client_fraction_per_round")),
                    )
                else:
                    self.logger.log(data)
            else:
                data = {
                    "id": self.server_id,
                    "acc": perplexity,
                    "q_len": queue_len,
                    "time": self.cur_time,
                    "tvd": server_tvd,
                    "avg_latency": avg_latency,
                    "max_latency": max_latency,
                    "min_latency": min_latency,
                    "std_latency": std_latency,
                }
                if not self.config.get("client_async") and not self.config.get("server_async"):  
                    self.logger.log(
                        data,
                        num_fedavg=round(len(self.clientsSet) * self.config.get("client_fraction_per_round")),
                    )
                else:
                    self.logger.log(data)
    def deliver_model(self, leader: bool = False) -> None:
        if leader:
            message = f"Server {self.server_id} is a leader, delivering model to all servers"
            print("=" * len(message))
            print(message)
            print("=" * len(message))
            for server in self.servers:
                copied_model_state_dict = {
                    k: v.clone().detach().to(device=self.device) for k, v in self.aggregator.model.state_dict().items()
                }
                server.set_global_model(copied_model_state_dict)
                print(f"Server {server.server_id} Set to new global model")
                server.deliver_model()
            return
        for item in self.update_buffer:
            client = self.clients[item[1]]
            copied_model_state_dict = {
                k: v.clone().detach().to(device=self.device) for k, v in self.aggregator.model.state_dict().items()
            }
            client.receive_global_model(copied_model_state_dict, self.age, self.server_id)
            if self.config.get("alternate_new"):
                if len(client.server) > 1 and client.model2 is not None:
                    print(
                        f"-----------client id{client.client_id} new_id{client.new_id} receive all model-----------------------"
                    )
                    self.history[item[1]] += 1
                elif len(client.server) > 1 and client.model2 is None:
                    print(
                        f"-----------client id{client.client_id} new_id{client.new_id} receive only one model-----------------------"
                    )
            else:
                self.history[item[1]] += 1
    def set_global_model(self, state_dict: Mapping) -> None:
        self.aggregator.model.load_state_dict(state_dict)
    def send_peer(self) -> None:
        """
        Used to synchronize among servers
        broadcast the model to other servers
        Push the event into other server' heap
        """
        message = f"Server {self.server_id} is sending model to all servers"
        print("%" * len(message))
        print(message)
        print("%" * len(message))
        self.event_history.add_event(
            MEvent(self.cur_time, f"server-{self.server_id}", "server_synchronization", self.age)
        )
        state_dict_copy = (
            self.aggregator.model.state_dict().copy()
        )  
        send_age = self.age  
        for server in self.servers:
            print(f"Server {self.server_id} sending model to Server {server.server_id}")
            arr_time = self.get_comm_delay("server", server.server_id)
            server.receive_peer(state_dict_copy, arr_time, send_age)  
    def receive_peer(self, model: OrderedDict, arr_time: float, age: int = 0) -> None:
        """
        Used to synchronize among servers
        receive the model from other servers
        Synchronize when all the models are received
        """
        self.slowest_sync = max(self.slowest_sync, arr_time)
        self.peer_buffer.append((model, age))
        if len(self.peer_buffer) == len(self.servers):
            if self.print_flag or self.sync_round % 20 == 0:
                print(f"(Server: {self.server_id}) started SYNCHRONIZATION {self.sync_round}")
            new_age = self.aggregator.synchronize(self.peer_buffer)  
            self.age = new_age
            self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | Synchronization"
            self.test_acc(log=False)
            self.sync_round += 1
            self.sent_peer = False
            self.peer_buffer = []
            self.cur_time += self.slowest_sync
            self.slowest_sync = 0
    def receive_server(self, model: torch.Tensor) -> None:
        """
        Used for hierarchical leader
        receive the model from other servers
        Synchronize when all the models are received
        """
        self.server_buffer.append(model)
        if len(self.server_buffer) == len(self.servers):
            self.aggregator.cloud_agg(self.server_buffer)
            self.sync_round += 1
            self.server_buffer = []
    def token_broadcast(self, token_id: int, init_id: int = None) -> None:
        """
        Called by the server who holds the token
        when the max age difference is over the threshold
        Broadcast the model to other servers -> Push the event into other servers' heaps
        """
        copied_model_state_dict = {k: v.to(device=self.device) for k, v in self.aggregator.model.state_dict().items()}
        send_age = self.age
        random.shuffle(self.shuffled_servers)
        if init_id is None:  
            self.token_sequence = token_id
            for server in self.servers:
                if server.server_id != self.server_id:
                    arr_time = self.cur_time + self.get_comm_delay("server", server.server_id)
                    if arr_time <= server.heap[0][0]:  
                        server.token_receive_broadcast(
                            arr_time,
                            copied_model_state_dict,
                            send_age,
                            token_id,
                            self.server_id,
                            self.server_id,
                        )
                    else:
                        heapq.heappush(
                            server.heap,
                            (
                                arr_time,
                                copied_model_state_dict,
                                send_age,
                                token_id,
                                self.server_id,
                                self.server_id,
                            ),
                        )
        else:
            for server in self.servers:
                if server.server_id != self.server_id:
                    arr_time = self.get_comm_delay("server", server.server_id)
                    if arr_time <= server.heap[0][0]:  
                        server.token_receive_broadcast(
                            arr_time,
                            copied_model_state_dict,
                            send_age,
                            token_id,
                            init_id,
                            self.server_id,
                        )
                    else:
                        heapq.heappush(
                            server.heap,
                            (
                                arr_time,
                                copied_model_state_dict,
                                send_age,
                                token_id,
                                init_id,
                                self.server_id,
                            ),
                        )
    def token_receive_broadcast(self, arr_time, peer_state_dict, peer_age, token_id, init_id, sender_id) -> None:
        self.cur_time = max(self.cur_time, arr_time)
        if self.token_sequence < token_id:
            print(f"{sender_id} -> {self.server_id} First Stage")
            self.token_sequence = token_id
            self.token_broadcast(token_id, init_id)
        else:
            print(f"{sender_id} -> {self.server_id} Second Stage")
        new_age, _, weight = self.aggregator.aggregate_peer_sgd(peer_state_dict, self.age, peer_age)
        self.cur_time += self.config.get("fedasync_delay")
        self.age = new_age
        self.last_age = new_age
        self.token_broadcast_cnt[token_id] += 1
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | TOKEN-BASED | Aggregation with server {sender_id}"
        self.test_acc(log=False)
        if self.token_broadcast_cnt[token_id] == len(self.servers) - 1:
            self.token_broadcast_cnt[token_id] = 0
            if self.server_id == init_id:  
                self.broadcasting = False
                print(f"BROADCAST {token_id} DONE")
                self.token.age_dict = {self.server_id: self.age}
                self.token.sequence_id += 1
                self.token_send_next()
    def token_send_next(self) -> None:
        self.servers[(self.server_id + 1) % len(self.servers)].token = self.token
        self.token = None
    def get_queue_len(self) -> int:
        queue_len: int = 0
        def count_in_q(idx) -> None:
            nonlocal queue_len
            if idx < len(self.heap) and self.heap[idx][0] < self.cur_time:
                queue_len += 1
                count_in_q(2 * idx + 1)
                count_in_q(2 * idx + 2)
        count_in_q(0)
        return queue_len
    def report(self) -> None:
        slow = 0
        fast = 0
        medium = 0
        for client in self.clients.values():
            if client.client_type == "slow":
                slow += 1
            elif client.client_type == "medium":
                medium += 1
            else:
                fast += 1
        print(
            colors[-1],
            f"{self.server_id}\t|{fast}\t|{medium}\t|{slow}\t|{self.age}\t\t|{self.sum_staleness/max(1, self.age)}",
            colors[-1],
        )
