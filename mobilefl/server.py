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
        self.leader: Server = None  # type: ignore
        self.hier_period = 100000
        self.age = 0  # how many times has the server updated
        self.last_age = 0
        self.clients: Dict[int, Client] = {}
        self.history = {}
        self.sum_staleness = 0
        self.tmp_sum_staleness = 0
        self.area = 0  # id of each server's location, e.g 0 means hongkong
        self.location = location  # reflect the 2D location of the server
        self.clientsSet: List[Client] = []  # to store the clients before set_clients()
        self.clients_num = 0  # the required number of clients this server should have
        self.client_train_datasets: List[CustomSubset] = None  # type: ignore
        self.client_test_datasets: List[CustomSubset] = None  # type: ignore
        self.server_level_speed: float = 0  # the average speed of server's clientset
        self.train_dataset_server_level: CustomSubset = None  # type: ignore
        self.label_count: Dict[int, int] = {}
        self.popped_heap_item = {}
        self.share_heap_item = {}
        self.push_heap_item = {}
        self.pop_client_id = None
        self.share_client_id = None
        # If recieved_moved = true, the server recieves client from other server
        self.recieved_moved = False
        # Synchronization Algorithm Parameters
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

        # Token-baed Broadcast Parameters
        self.token_sequence = 0
        self.token: AgeToken = None  # type: ignore
        self.broadcasting = False
        self.token_broadcast_cnt = collections.defaultdict(int)

        self.current_total_weighted_tvd: float = 0.0

        self.updates = {}  # to store the updates of its clients' training delay

        if self.config.get("cuda"):
            torch.cuda.manual_seed(0)
            self.device = torch.device(f"cuda:{self.config.get('cuda_to_use')}")
        else:
            self.device = "cpu"

        # if self.config.get("dataset") == "wikitext2":
        #     self.criterion = torch.nn.CrossEntropyLoss()
        # else:
        #     self.criterion = F.nll_loss
        self.criterion = F.nll_loss

        self.frequency_matrix: np.ndarray = None  # to store the frequency matrix of all clients

        # @TODO: This should be removed in the end
        self.global_label_count = {}  # to store the global label count of all clients

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
        # print(f"Server {self.server_id} label keys: {lk}")
        # lk = np.arange(len(self.train_dataset_server_level.classes))
        all_counts = []
        client_ids = list(self.clients.keys())
        for client in self.clients.values():
            client_label_count = client.get_label_count_as_array(lk)
            all_counts.append(client_label_count)

        # Combine all client label counts into a single matrix
        server_lc = np.vstack(all_counts) if all_counts else np.zeros((1, len(lk)))
        return server_lc, client_ids

    # def compute_weighted_tvd(self, global_label_count: np.ndarray):
    #     latencies = self.get_clients_latencies()
    #     server_label_count, client_ids = self.collect_client_label_count()

    #     return compute_weighted_tvd(latencies, client_ids, server_label_count, global_label_count)
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
            # print(f"Client {client.client_id} new total weighted TVD: {new_total_weighted_tvd}")
            tvd_diff = total_weighted_tvd - new_total_weighted_tvd
            # print(f"Client {client.client_id} TVD difference: {tvd_diff}")

            if tvd_diff > 0 and new_total_weighted_tvd < min_weighted_tvd:
                # print(f"Client {client.client_id} is a candidate for removal")
                min_weighted_tvd = new_total_weighted_tvd
                client_to_remove["client_id"] = client.client_id
                client_to_remove["client_distribution"] = client_distribution
                client_to_remove["client_latency"] = client_latency

        return total_weighted_tvd, new_total_weighted_tvd, client_to_remove
        # new_tvd = self.estimate_augmented_tvd(
        #     global_label_count, client.client_id, client.get_label_count_as_array(), client.latency, action="subtract")

        # new_total_weighted_tvd, client_to_remove = self.propose_client_tvd(client_distributions, client_weights)

        # return total_weighted_tvd, new_total_weighted_tvd, client_to_remove

    # def propose_client_tvd(
    #     self, client_distributions, client_weights
    # ) -> Tuple[float, Dict[str, Union[int, np.ndarray, float]]]:

    #     # Try removing each client and recompute weighted TVD
    #     min_weighted_tvd = float("inf")
    #     client_to_remove = {
    #         "client_id": -1,
    #         "client_distribution": None,
    #         "client_latency": None,
    #     }

    #     print(f"Dim client_distributions: {client_distributions.shape}")
    #     for i in range(len(self.clients)):

    #         # new_tvd = compute_weighted_tvd(client_latencies, client_ids, server_label_count, global_label_count)

    #         # Remove i-th client
    #         remaining_dists = np.delete(client_distributions, i, axis=0)
    #         remaining_weights = np.delete(client_weights, i)

    #         # Normalize remaining weights
    #         remaining_weights /= remaining_weights.sum()

    #         new_global = np.average(remaining_dists, axis=0, weights=remaining_weights)
    #         new_tvd_matrix = np.abs(remaining_dists - new_global)
    #         new_scalar_tvds = 0.5 * np.sum(new_tvd_matrix, axis=1)
    #         new_total_weighted_tvd = np.sum(remaining_weights * new_scalar_tvds)

    #         if new_total_weighted_tvd < min_weighted_tvd:
    #             min_weighted_tvd = new_total_weighted_tvd
    #             client_to_remove["client_id"] = i
    #             client_to_remove["client_distribution"] = remaining_dists[i]
    #             client_to_remove["client_latency"] = self.frequency_matrix[i, self.server_id]

    #     return new_total_weighted_tvd, client_to_remove

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
            # Add cliient's distribution to server_label_count using vstack
            new_server_label_count = np.vstack((server_label_count, client_distribution))
            latencies = np.append(latencies, latency)
        elif action == "remove":
            # Get the index of the client to remove
            client_index = client_ids.index(client_id)
            # Remove the client distribution and latency
            new_server_label_count = np.delete(server_label_count, client_index, axis=0)
            latencies = np.delete(latencies, client_index)
        else:
            raise ValueError(f"Action must be either 'add' or 'remove' but got {action}")

        # Compute the new total weighted TVD
        new_total_weighted_tvd, _, _, _ = compute_weighted_tvd(
            latencies, client_ids, new_server_label_count, global_label_count
        )
        return new_total_weighted_tvd

    def calculate_server_noniidness(self, global_label_count: dict) -> float:
        # print(f"Non-iddness for server {self.server_id=}")
        # print(f"Server label count: {dict(sorted(self.get_all_client_label_count().items()))}")
        client_non_iidness = []
        for client in self.clients.values():
            tvd_value = client.calculate_non_iid(global_label_count)
            # print(f"Client {client.client_id} tvd value: {tvd_value}")
            client_frequency = 1.0
            client_effective_non_iid = client_frequency * tvd_value
            client_non_iidness.append(client_effective_non_iid)
        # print(f"Client non-iidness: {client_non_iidness}")
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
            # elc = lc * self.frequency_matrix[client.client_id, self.server_id]
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
            # print(f"Client {client.id} label count: {lc}")
            # print(f"All clients label count: {all_labels}")
        if reference_labels is not None:
            # print(f"Reference labels: {reference_labels}")
            result = []
            for label in reference_labels:
                if label in list(all_labels.keys()):
                    result.append(all_labels[label])
                else:
                    result.append(0)
            return np.array(result)
        return all_labels

    # def get_client_label_count(self, client_id: int) -> Dict[int, int]:
    #     return self.clients[client_id].get_data_label_count()

    def calculate_clients_speed(self, moving: bool) -> np.ndarray:
        previous_id = list(self.clients.values())[-1].client_id
        if moving:
            N = self.clients_num + 1

            self.clients[self.clients_num].client_id = self.clients_num
        else:
            N = self.clients_num
        M = len(list(self.clients.values())[0].train_loader.dataset.classes)
        A = np.zeros((M, N))
        # B:average speed of all clients associated to the same server * total labels of the client set
        B = np.zeros(M)
        for i in range(N):
            client = list(self.clients.values())[i]
            labels = client.train_loader.dataset.targets
            labels = np.round(labels).astype(int)
            label_counts = Counter(labels)
            # If labels is a Tensor, it needs to be converted to a numpy array
            for label, count in label_counts.items():
                A[label][i] = count * (1 + 2 * self.get_comm_delay("client", client.client_id) + 2)
            # if i == N-1:
            #     print('get_comm_delay',self.get_comm_delay('client', client.id))

        all_labels = self.train_dataset_server_level.targets
        all_labels_count = Counter(all_labels)

        # print("all labels",sum(Counter(all_labels).values()))
        for label, count in all_labels_count.items():
            B[label] = count * self.server_level_speed
        # Use numpy.linalg.lstsq to solve AX = B
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        if moving:
            self.clients[self.clients_num].client_id = previous_id

        for i in range(N):
            client = list(self.clients.values())[i]
            # self.clients[i].training_time = X[i]
            if i == N - 1:
                print(
                    f"Calculate that: Server {self.server_id}: client {client.client_id}'s training time (speed) is {X[i]} "
                )

        # print("x",X)
        for i in range(N):
            if X[i] <= 0:
                return X
            else:
                client = list(self.clients.values())[i]
                X[i] = X[i] + 2 * self.get_comm_delay("client", client.client_id) + 2
                # print("speed of all is", X)
                # for i in range(N):
                #         # self.clients[i].training_time = X[i]
                #         print(f"Calculate that: Server {self.id}: client {self.clients[i].id}'s training time (speed) is {X[i]} ")
                return X

    def set_clients_speed(self):
        N = self.clients_num
        # M = len(list(self.clients.values())[0].train_loader.dataset.classes)
        # # Filling coefficient matrix A and resultant vector B
        # # AX = B
        # # To store the labels number of each class for each client.
        # # row: class id; coloum: client id
        # # C[i,j] = Client-j has C[i,j] class-i samples
        # A = np.zeros((M, N))
        # # B:average speed of all clients associated to the same server * total labels of the client set
        # B = np.zeros(M)
        client_ids = list(self.clients.keys())
        print(f"Server {self.server_id} clients ids: {client_ids}")
        # for idx, i in enumerate(client_ids):
        #     client = self.clients[i]
        #     labels = client.train_loader.dataset.targets
        #     labels = np.round(labels).astype(int)
        #     label_counts = Counter(labels)  # If labels is a Tensor, it needs to be converted to a numpy array
        #     for label, count in label_counts.items():
        #         if client.new_id != None:
        #             A[label][idx] = count * (1 + 2 * self.get_comm_delay("client", client.new_id) + 2)
        #         else:
        #             A[label][idx] = count * (1 + 2 * self.get_comm_delay("client", client.client_id) + 2)

        # all_labels = self.train_dataset_server_level.targets
        # all_labels_count = Counter(all_labels)
        # for label, count in all_labels_count.items():
        #     B[label] = count * self.server_level_speed
        # # Use numpy.linalg.lstsq to solve AX = B
        # X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        baseline = False
        for idx, i in enumerate(client_ids):
            # if self.clients[i].training_time != None and self.config.get("move_late"):

            #     # if moving clients during running the program or moving client before sharing
            #     # the tuple in the heap is ((curr_time + communcation delay+training_delay), clientid)
            #     # for server removing clients, heap will pop
            #     # for server recieving clients, heap will add the recieved clients' previous total time
            #     if idx == N - 1:
            #         key = f"speed_of_server_{self.server_id}_client_{i+1}"
            #         if key in self.updates:
            #             del self.updates[key]
            #         condition = lambda x: x[1] == self.pop_client_id
            #         self.popped_heap_item[self.pop_client_id] = self.heappop_condition(condition)
            #     updated_heap = []
            #     while self.heap:
            #         item = heapq.heappop(self.heap)
            #         if isinstance(item[1], int):
            #             total_time = item[0]
            #             item_id = item[1]  # What the hell is item?
            #             if item_id == i:
            #                 updated_time = total_time - self.clients[i].training_time + X[i]
            #                 heapq.heappush(updated_heap, (updated_time, item_id))
            #             else:
            #                 heapq.heappush(updated_heap, item)
            #         else:
            #             heapq.heappush(updated_heap, item)
            #     self.clients[i].training_time = X[i]
            #     self.heap = updated_heap

            # elif self.clients[i].training_time == None:
            #     if (self.config.get("alternate_new") and not self.config.get("move_a_client")) or self.config.get(
            #         "avg_model"
            #     ):

            #         server_id = "server_" + str(self.server_id) + "_clients"
            #         if self.clients_num == self.config.get("num_clients") * self.config.get(server_id):
            #             # For the server who will share client
            #             heapq.heappush(self.heap, (X[i] + 2 * self.get_comm_delay("client", i) + 2, i))
            #             for item in self.heap:
            #                 if item[1] == self.share_client_id:
            #                     self.share_heap_item[self.share_client_id] = item
            #             self.clients[i].training_time = X[i]
            #         elif self.clients_num > self.config.get("num_clients") * self.config.get(server_id):
            #             # For the server who will receive the shared client
            #             self.clients[i].training_time = X[i]
            #             heapq.heappush(self.heap, (X[i] + 2 * self.get_comm_delay("client", i) + 2, i))

            #     else:
            # @Bart: We use this branch!
            # for 1. baseline; 2.3.move at begin or after running
            X = self.frequency_matrix[:, self.server_id]
            self.clients[i].training_time = X[idx]
            if i == N - 1:
                print(
                    f"Server {self.server_id}: client {self.clients[i].client_id}'s training time (speed) is {X[idx]} "
                )
                if i in self.push_heap_item:
                    # for scenario: moving after running the program
                    # means the client i is recieved from other server
                    # we need to push the recieved clients' previous total time
                    heapq.heappush(self.heap, (self.push_heap_item[i][0], i))
                else:
                    # for scenario: move then share
                    # the moving first condition is calculate as baseline
                    if self.config.get("move_a_client"):
                        heapq.heappush(
                            self.heap,
                            (X[idx] + 2 * self.get_comm_delay("client", i) + 2, i),
                        )

                    else:
                        # for baseline condition
                        baseline = True
                        heapq.heappush(
                            self.heap,
                            (X[idx] + 2 * self.get_comm_delay("client", i) + 2, i),
                        )
            else:
                heapq.heappush(self.heap, (X[idx] + 2 * self.get_comm_delay("client", i) + 2, i))
                self.config.write_config(self.updates)

            # elif (
            #     (self.clients[i].training_time != None and self.config.get("alternate_late"))
            #     or (self.clients[i].training_time != None and self.config.get("avg_model"))
            #     or (self.clients[i].training_time != None and self.config.get("alternate_new"))
            # ):
            #     # if sharing a client after running the program or after moving a client
            #     # The server owned chosen client should share its client's (curr_time + communcation delay+training_delay)
            #     # The target server who will receive new client's updates will compute new training delay for this client
            #     # The target server's all clients' training_time are not None

            #     server_id = "server_" + str(self.server_id) + "_clients"
            #     # if self.clients_num == self.config.get("num_clients")* self.config.get(server_id):
            #     if self.recieved_moved or (
            #         self.clients_num == self.config.get("num_clients") * self.config.get(server_id)
            #         and self.config.get("alternate_late")
            #     ):
            #         # The server owned chosen client which is recieved from moving process
            #         for item in self.heap:
            #             if item[1] == self.share_client_id:
            #                 print("22222222222222222", self.share_client_id)
            #                 self.share_heap_item[self.share_client_id] = item
            #                 print(f"Server {self.server_id} share client {self.clients[i].client_id}'s heap is {item} ")
            #         print(
            #             f"Server {self.server_id}: client {self.clients[i].client_id}'s training time (speed) is {X[i]} "
            #         )

            #     elif (not self.recieved_moved) or (
            #         self.clients_num > self.config.get("num_clients") * self.config.get(server_id)
            #         and self.config.get("alternate_late")
            #     ):
            #         print(
            #             f"Server {self.server_id}: client {self.clients[i].client_id}'s training time (speed) is {X[i]} "
            #         )
            #         # The server recieve the shared client
            #         if i in self.push_heap_item:
            #             self.clients[i].training_time_new = X[i]
            #             print(
            #                 f"Server {self.server_id} received the shared client {self.clients[i].new_id}'s heap is {self.push_heap_item[i][0]} "
            #             )
            #             heapq.heappush(self.heap, (self.push_heap_item[i][0], i))
            #         else:
            #             self.clients[i].training_time = X[i]

            key = f"speed_of_server_{self.server_id}_client_{i}"
            self.updates[key] = X[idx]

        if baseline:
            self.config.write_config(self.updates)
        return X

    def heappop_condition(self, condition: Any) -> Any:
        """Pop the item off the heap that satisfies the condition."""
        for index, item in enumerate(self.heap):
            if condition(item):
                # Remove the item that satisfies the condition
                self.heap[index] = self.heap[-1]
                self.heap.pop()
                # Restore the heap property
                if index < len(self.heap):
                    heapq._siftup(self.heap, index)
                    heapq._siftdown(self.heap, 0, index)
                return item
        raise KeyError("No element found that satisfies the condition")

    # def set_clients(self, clients):
    #     self.slowest = clients[-1].training_time
    #     cnt = [0] * 3
    #     mean = self.config.get('training_delay')
    #     for client in clients:
    #         if self.config.get("num_servers") > 1:
    #             client.area = self.area
    #         if client.type == 'slow':
    #             cnt[0] += 1
    #         elif client.type == 'medium':
    #             cnt[1] += 1
    #         elif client.type == 'fast':
    #             cnt[2] += 1
    #         else:
    #             if client.gaussian_mu > 0 and client.gaussian_mu <=  0.8 * mean:
    #                 cnt[2] += 1
    #             elif client.gaussian_mu > 0.8 * mean and client.gaussian_mu <=  1.2 * mean:
    #                 cnt[1] += 1
    #             elif client.gaussian_mu > 1.2 * mean:
    #                 cnt[0] += 1
    #         self.clients[client.id] = client
    #         self.history[client.id] = 0
    #         # self.slowest = max(self.slowest, client.training_time)
    #         # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
    #         heapq.heappush(self.heap, (client.training_time + 2 * self.get_comm_delay('client', client.id), client.id))

    #     print(colors[self.id % len(colors)], f"Server {self.id} Slow : Medium : Fast = {cnt[0]} : {cnt[1]} : {cnt[2]}", colors[-1])

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
            # print(self.clients)

    #################### communication delay between servers is according to the cloudping####################
    # 1 to 6 are: hongkong, tokyo, Sydney, Canada, London, California
    # the communication delay is provided by cloudping

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
        # calculate the communication delays between clients and server
        elif who == "client":
            client = self.clients[node_id]
            distance = self.calculateDistance(self.location, client)
            # max communication delays and distance are from area 2 to 4
            max_comm_delay = 267.46
            max_distance = 270.81  # calculate the euclidean distance
            # the delay from client to server is mapped_delay + 2
            mapped_delay = (distance / max_distance) * max_comm_delay
            return mapped_delay

    def calculateDistance(self, location: Tuple[int, int], client: Client) -> float:
        # euclidean distance: the distance in the map is the communication delay
        # server.location (x,y)
        # client.location (x,y)

        distance = np.sqrt((client.location[0] - location[0]) ** 2 + (client.location[1] - location[1]) ** 2)
        # print(f"the distance between self{location} and client{client} is {distance}")
        return distance

    def set_server_location(self, coordinates: np.ndarray) -> None:
        self.location = coordinates[self.area]

    # def get_comm_delay(self, who, id):
    #     if self.config.get('comm_01') == 0:
    #         return 0
    #     if id == 'max':
    #         max_comm = -1
    #         for server in self.servers:
    #             if self.area == server.area:
    #                 continue
    #             if self.area < server.area:
    #                 left = self.area
    #                 right = server.area
    #             elif self.area > server.area:
    #                 left = server.area
    #                 right = self.area
    #             max_comm = max(max_comm, self.config.get(f"comm_{left}{right}"))
    #         return max_comm

    #     if who == 'client':
    #         client = self.clients[id]
    #         if client.area == self.area:
    #             return self.config.get('comm_delay')
    #         if self.area < client.area:
    #             left = self.area
    #             right = client.area
    #         else:
    #             left = client.area
    #             right = self.area
    #         return self.config.get('comm_delay') + self.config.get(f"comm_{left}{right}")

    #     elif who == 'server':
    #         if self.config.get("result_file") == 'mtest':
    #             return 150
    #         server = self.servers[id]
    #         if server.id == self.id:
    #             return 0
    #         if self.area < server.area:
    #             left = self.area
    #             right = server.area
    #         else:
    #             left = server.area
    #             right = self.area
    #         try:
    #             return self.config.get(f"comm_{left}{right}")
    #         except:
    #             return 150

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
        # fedasync
        # @TODO:This is the main server loop. Bart look here!
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
        # print(f"{log_prefix}Starting local async global sync update at time {self.cur_time:.2f}")
        # print(
        #     f'{log_prefix} len(self.update_buffer) < self.config.get("aggregation_buffer_size"): {len(self.update_buffer)} < {self.config.get("aggregation_buffer_size")}'
        # )

        # print(f"Number of clients in heap: {len(self.heap)}")
        # print(f"Number of clients in update buffer: {len(self.update_buffer)}")
        # print(f"NUmber of clients in server: {len(self.clients)}")
        # exit()
        while len(self.update_buffer) < self.config.get("aggregation_buffer_size"):
            next_update_time, client_id = self.heap[0]

            # print(f"{log_prefix} checking next client: {client_id} at time {next_update_time:.2f}")
            # Seperate terms for better readability
            # print(
            #     f"{log_prefix}: values {next_update_time=}, {self.cur_time=}, {self.sync_round=}, {self.period=}, {self.sent_peer=}"
            # )

            if (
                self.sync_round > 0 and max(self.cur_time, next_update_time) > self.sync_round * self.period
            ):  # check if the server should wait
                if not self.sent_peer:
                    self.sent_peer = True
                    self.send_peer()
                    print(f"{log_prefix}Server {self.server_id} waiting")
                # print(
                #     f"{log_prefix} returning False because {self.sync_round=} and {self.cur_time=}, {next_update_time=} = {max(self.cur_time, next_update_time)} and {self.sync_round=} * {self.period=}"
                # )
                return False  # wait

            next_time, client_id = heapq.heappop(self.heap)
            self.cur_time = max(self.cur_time, next_time) + self.config.get("fedasync_delay")
            client = self.clients[client_id]
            assert event_system is not None, "Event system must be provided for async updates"
            # @TODO: Implement events in the event system

            ###################### receive update ####################
            # _mem = torch.cuda.memory_allocated(0)
            start_time = time.time()
            if client.train(self.cur_time):
                #     if self.history[client.id]:
                #         print(line(f"000: {client.id} -> {torch.cuda.memory_allocated(0) - mem}"))
                #     else:
                #         print(line(f"111: {client.id} -> {torch.cuda.memory_allocated(0) - mem}"))
                # print(f"{log_prefix}client {client_id} updates number {client.num_updates} ")
                _client_loss_val = client.loss_val
                self.update_buffer.append((client.update, client.client_id, self.age - client.age))
                _t_time = time.time() - start_time
                # print(f"{log_prefix}Client {client_id} trained in {t_time:.2f} seconds")
                # client.test() # original client test
            else:
                print(f"{log_prefix}Client {client_id} failed to train, skipping...")
            ##########################################################
            self.sum_staleness += self.age - client.age
            self.tmp_sum_staleness += self.age - client.age  # for report
            # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
            # print(f"{log_prefix}{self.cur_time=}, {next_time=}, {client_id=}, {client.training_time=}")
            assert client.training_time is not None, f"Client {client_id} training time is None"
            heapq.heappush(
                self.heap,
                (
                    self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client.client_id) + 2,
                    client_id,
                ),
            )

        ##################### aggregation ########################
        # print(f"{log_prefix}Aggregating updates from {len(self.update_buffer)} clients")
        self.aggregator.aggregate(self.update_buffer, cosine=False, aggregation_method=self.config.get("agr"))

        # print(f"(Server: {self.id}) has model after training: {self.aggregator.model.state_dict()[list(self.aggregator.model.state_dict().keys())[0]][0][0][0][0].item():.4f}")
        client_information = ""
        for _, clientid, _ in self.update_buffer:
            client_information += f"{self.clients[clientid]} - {self.clients[clientid].lr:.3f}"
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | {client_information} | Staleness {self.tmp_sum_staleness / len(self.update_buffer):.1f}"

        # if self.age % 10 == 0:
        self.test_acc()  # server test

        # send back the updated model and the age of the model
        # if self.tmp_sum_staleness / len(self.update_buffer) <= self.config.get("staleness_threshold") * len(self.clients):
        #     # too large staleness reduce this update's effect. Should not make the server grow. Or it will keep bring down the afterward update's influence
        self.age += 1
        self.num_updates += 1

        self.deliver_model()
        # print(f"(Server: {self.id}) has model after deliver: {self.aggregator.model.state_dict()[list(self.aggregator.model.state_dict().keys())[0]][0][0][0][0].item():.4f}")
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
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            # print(len(self.clientsSet))
            rounded_val = round(len(self.clientsSet) * self.config.get("client_fraction_per_round"))
            # print(f'{rounded_val=}')
            # print(f'{len(self.clients)=}')
            # print(f'{self.clients=}')
            # Make sure client_keys is a list and not dict_keys
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
            )  # duration: receiving model + training + sending model
            slowest = max(duration, slowest)
            if client.train(self.cur_time):
                self.event_history.add_event(MEvent(self.cur_time, f"client-{client_id}", "train"))
                self.update_buffer.append((client.update, client_id))
                # client.test()
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
        # multiasync
        while not isinstance(self.heap[0][1], int):  # it is a token broadcast
            arr_time, peer_state_dict, peer_age, token_id, init_id, sender_id = heapq.heappop(self.heap)
            self.token_receive_broadcast(arr_time, peer_state_dict, peer_age, token_id, init_id, sender_id)

        while len(self.update_buffer) < int(self.config.get("aggregation_buffer_size")):
            next_time, client_id = heapq.heappop(self.heap)
            self.cur_time = max(self.cur_time, next_time) + float(self.config.get("fedasync_delay"))
            client = self.clients[client_id]
            if self.config.get("avg_model"):
                if len(client.server) == 2 and client.server[1] == self.server_id:
                    if not client.flag:
                        # heapq.heappush(self.heap, (self.cur_time + next_time, client_id))
                        return
                    else:
                        ###################### receive update ####################
                        if client.train(self.cur_time):
                            self.update_buffer.append((client.update, client_id, self.age - client.age))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        ##########################################################
                        self.sum_staleness += self.age - client.age
                        self.tmp_sum_staleness += self.age - client.age  # for report
                        # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                        # if client.server[0]==self.id:
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
                        # else:
                        #     heapq.heappush(self.heap, (self.cur_time + client.training_time_of_new_server[self.id] + 2 * self.get_comm_delay('client', client_id)+2, client_id))
                elif (
                    len(client.server) == 2 and client.server[0] == self.server_id
                ):  # ; means client belongs to this server
                    if client.flag:
                        # heapq.heappush(self.heap, (self.cur_time + next_time, client_id))
                        return
                    else:
                        ###################### receive update ####################
                        if client.train(self.cur_time):
                            self.update_buffer.append((client.update, client_id, self.age - client.age))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        ##########################################################
                        self.sum_staleness += self.age - client.age
                        self.tmp_sum_staleness += self.age - client.age  # for report
                        # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                        # if client.server[0]==self.id:
                        heapq.heappush(
                            self.heap,
                            (
                                self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                                client_id,
                            ),
                        )
                        # server_other = self.servers[client.server[1]]
                        # heapq.heappush(server_other.heap,(self.cur_time + client.training_time + 2 * self.get_comm_delay('client', client_id)+2, client.new_id))
                        client.flag = True
                        # else:
                        #     heapq.heappush(self.heap, (self.cur_time + client.training_time_of_new_server[self.id] + 2 * self.get_comm_delay('client', client_id)+2, client_id))
                else:
                    ###################### receive update ####################
                    if client.train(self.cur_time):
                        self.update_buffer.append((client.update, client_id, self.age - client.age))
                    else:
                        print(f"Client {client_id} failed to train, skipping...")
                    ##########################################################
                    self.sum_staleness += self.age - client.age
                    self.tmp_sum_staleness += self.age - client.age  # for report
                    # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                    # if client.server[0]==self.id:
                    heapq.heappush(
                        self.heap,
                        (
                            self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                            client_id,
                        ),
                    )
                    # else:
                    #     heapq.heappush(self.heap, (self.cur_time + client.training_time_of_new_server[self.id] + 2 * self.get_comm_delay('client', client_id)+2, client_id))

            elif self.config.get("alternate_new"):
                if len(client.server) == 2 and client.server[1] == self.server_id:
                    if not client.flag:
                        # heapq.heappush(self.heap, (self.cur_time + next_time, client_id))
                        return
                    else:
                        ###################### receive update ####################
                        if client.train(self.cur_time, model=2):
                            self.update_buffer.append((client.update, client_id, self.age - client.age1))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        ##########################################################
                        self.sum_staleness += self.age - client.age1
                        self.tmp_sum_staleness += self.age - client.age1  # for report
                        # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                        # if client.server[0]==self.id:
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
                        # else:
                        #     heapq.heappush(self.heap, (self.cur_time + client.training_time_of_new_server[self.id] + 2 * self.get_comm_delay('client', client_id)+2, client_id))
                elif (
                    len(client.server) == 2 and client.server[0] == self.server_id
                ):  # ; means client belongs to this server
                    if client.flag:
                        # heapq.heappush(self.heap, (self.cur_time + next_time, client_id))
                        return
                    else:
                        ###################### receive update ####################
                        if client.train(self.cur_time, model=1):
                            self.update_buffer.append((client.update, client_id, self.age - client.age))
                        else:
                            print(f"Client {client_id} failed to train, skipping...")
                        ##########################################################
                        self.sum_staleness += self.age - client.age
                        self.tmp_sum_staleness += self.age - client.age  # for report
                        # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                        # if client.server[0]==self.id:
                        heapq.heappush(
                            self.heap,
                            (
                                self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                                client_id,
                            ),
                        )
                        # server_other = self.servers[client.server[1]]
                        # heapq.heappush(server_other.heap,(self.cur_time + client.training_time + 2 * self.get_comm_delay('client', client_id)+2, client.new_id))
                        client.flag = True
                else:
                    ###################### receive update ####################
                    if client.train(
                        self.cur_time,
                    ):
                        self.update_buffer.append((client.update, client_id, self.age - client.age))
                    else:
                        print(f"Client {client_id} failed to train, skipping...")
                    ##########################################################
                    self.sum_staleness += self.age - client.age
                    self.tmp_sum_staleness += self.age - client.age  # for report
                    # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                    # if client.server[0]==self.id:
                    heapq.heappush(
                        self.heap,
                        (
                            self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                            client_id,
                        ),
                    )
                # else:
                #     heapq.heappush(self.heap, (self.cur_time + client.training_time_of_new_server[self.id] + 2 * self.get_comm_delay('client', client_id)+2, client_id))

            else:
                ###################### receive update ####################
                if client.train(self.cur_time):
                    self.update_buffer.append((client.update, client_id, self.age - client.age))
                else:
                    print(f"Client {client_id} failed to train, skipping...")
                ##########################################################
                self.sum_staleness += self.age - client.age
                self.tmp_sum_staleness += self.age - client.age  # for report
                # print(f"(Server {self.id}) - (Client {client.id}) || training: {client.training_time} comm: {2 * self.get_comm_delay('client', client.id)}")
                # if client.server[0]==self.id:
                heapq.heappush(
                    self.heap,
                    (
                        self.cur_time + client.training_time + 2 * self.get_comm_delay("client", client_id) + 2,
                        client_id,
                    ),
                )
                # else:
                #     heapq.heappush(self.heap, (self.cur_time + client.training_time_of_new_server[self.id] + 2 * self.get_comm_delay('client', client_id)+2, client_id))

        ##################### aggregation ########################
        self.aggregator.aggregate(self.update_buffer, cosine=False, aggregation_method=self.config.get("agr"))
        client_information = ""
        for _, clientid, _ in self.update_buffer:
            client_information += f"{self.clients[clientid]} - {self.clients[clientid].lr:.3f}"
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | {client_information} | Staleness {self.tmp_sum_staleness / len(self.update_buffer):.1f}"
        # print(colors[self.id % len(colors)], f"Server {self.id}: updated with average staleness {round(self.tmp_sum_staleness / len(self.update_buffer), 3)} at time {self.cur_time} | server age: {self.age} | {client_information}" , colors[-1])

        self.test_acc()  # server test

        self.age += 1
        self.num_updates += 1
        self.deliver_model()
        self.update_buffer = []
        self.tmp_sum_staleness = 0

        ############################ token-based communication ############################
        # check if the token is in hand, if yes:
        # 1. Update its age in the token
        # 2. Calculate the largest difference of age, if larger than threshold, initialize the broadcast
        # 3. If not larger than threshold, send the token to the next server
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
                self.broadcasting = True  # stop sending out the token
                self.token_broadcast(token_id=self.token.sequence_id)
            else:
                self.token_send_next()

    def add_client(self, client: Client, next_time=0) -> None:

        # Update the clients_set
        # Update the self.num_clients

        self.clientsSet.append(client)

        # Add server to client's server list
        if self.server_id not in client.server:
            client.server.append(self.server_id)

        self.set_clients([client])
        # update the number of clients

        assert isinstance(client.client_id, int), "Client ID must be an integer."
        heapq.heappush(self.heap, (next_time, client.client_id))

    def remove_client(self, client_id: int) -> Tuple[Client, float]:

        if client_id not in self.clients:
            raise ValueError(f"Client ID {client_id} not found in server {self.server_id}.")

        next_time, client_id = self.pop_item_with_client_id(self.heap, client_id)
        # Remaining time for client to finish training
        time_remaining = next_time - self.cur_time
        if time_remaining < 0:
            print(f"Warning: Negative time remaining for client {client_id} in server {self.server_id}. Setting to 0.")
            time_remaining = 0  # Ensure non-negative time remaining
        assert time_remaining >= 0, "Time remaining should be non-negative."

        # Remove client from self.clients
        client = self.clients.pop(client_id)
        # Remove client from self.clientsSet
        self.clientsSet.remove(client)

        # Remove server from client's server list
        if self.server_id in client.server:
            client.server.remove(self.server_id)

        # Remove client from self.update_buffer if it exists
        self.update_buffer = [item for item in self.update_buffer if item[1] != client_id]

        # Remove client from self.history if it exists
        if client_id in self.history:
            del self.history[client_id]
        # Remove client from self.push_heap_item if it exists
        # if client_id in self.push_heap_item:
        #     del self.push_heap_item[client_id]

        # Update the number of clients
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
                    # test_loss += self.criterion(output, target).item() * inputs.shape[1]
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
        # @TODO: Bart here is the main printing
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

        # log
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
                if not self.config.get("client_async") and not self.config.get("server_async"):  # fedavg
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
                if not self.config.get("client_async") and not self.config.get("server_async"):  # fedavg
                    self.logger.log(
                        data,
                        num_fedavg=round(len(self.clientsSet) * self.config.get("client_fraction_per_round")),
                    )
                else:
                    self.logger.log(data)

    def deliver_model(self, leader: bool = False) -> None:
        if leader:
            # deliver models to the servers not clients
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

    ########################### Server Synchronization #################################
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
        )  # copy the state dict to avoid changing it during synchronization

        # copied_model_state_dict = {
        #     k: v.clone().detach().to(device=self.device) for k, v in self.aggregator.model.state_dict().items()
        # }
        send_age = self.age  # avoid change of age after synchronization
        for server in self.servers:
            print(f"Server {self.server_id} sending model to Server {server.server_id}")
            arr_time = self.get_comm_delay("server", server.server_id)
            # server.receive_peer(
            #     copied_model_state_dict, arr_time, send_age
            # )  # self.age is the number of updates of this server
            server.receive_peer(state_dict_copy, arr_time, send_age)  # self.age is the number of updates of this server
        # self.cur_time += self.get_comm_delay('server', 'max')

    # def receive_peer(self, model: torch.Tensor, arr_time: float, age: int = 0) -> None:
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
            new_age = self.aggregator.synchronize(self.peer_buffer)  # once all the models are received, synchronize
            self.age = new_age
            self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | Synchronization"
            self.test_acc(log=False)
            self.sync_round += 1
            self.sent_peer = False
            self.peer_buffer = []
            self.cur_time += self.slowest_sync
            self.slowest_sync = 0

    def receive_server(self, model: torch.Tensor) -> None:
        # def receive_server(self, model: torch.Tensor) -> None:
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

    ########################### Token-based server communication #################################
    def token_broadcast(self, token_id: int, init_id: int = None) -> None:
        """
        Called by the server who holds the token
        when the max age difference is over the threshold
        Broadcast the model to other servers -> Push the event into other servers' heaps
        """
        copied_model_state_dict = {k: v.to(device=self.device) for k, v in self.aggregator.model.state_dict().items()}
        send_age = self.age

        random.shuffle(self.shuffled_servers)
        if init_id is None:  # I am the initiator
            self.token_sequence = token_id
            for server in self.servers:
                if server.server_id != self.server_id:
                    arr_time = self.cur_time + self.get_comm_delay("server", server.server_id)
                    if arr_time <= server.heap[0][0]:  # the next event to deal with for this server
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

            # for server in self.shuffled_servers:
            #     if server.id != self.id:
            #         server.token_receive_broadcast(copied_model_state_dict, send_age, token_id, self.id, self.id)
        else:
            for server in self.servers:
                if server.server_id != self.server_id:
                    arr_time = self.get_comm_delay("server", server.server_id)
                    if arr_time <= server.heap[0][0]:  # the next event to deal with for this server
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

        # if it is the first time to receive the broadcast of this id, broadcast its own model
        self.cur_time = max(self.cur_time, arr_time)
        if self.token_sequence < token_id:
            print(f"{sender_id} -> {self.server_id} First Stage")
            self.token_sequence = token_id
            # print(f"Server {self.id} second-stage broadcast")
            self.token_broadcast(token_id, init_id)
        else:
            print(f"{sender_id} -> {self.server_id} Second Stage")
        # aggregate
        new_age, _, weight = self.aggregator.aggregate_peer_sgd(peer_state_dict, self.age, peer_age)
        self.cur_time += self.config.get("fedasync_delay")
        # if self.token_sequence % 20 == 0:
        # print(f"Server {self.id} learning {weight} from Server {sender_id}")
        self.age = new_age
        self.last_age = new_age
        self.token_broadcast_cnt[token_id] += 1
        self.information += f"Server {self.server_id} | {self.cur_time:.2f} | {self.age} | TOKEN-BASED | Aggregation with server {sender_id}"
        self.test_acc(log=False)
        if self.token_broadcast_cnt[token_id] == len(self.servers) - 1:
            self.token_broadcast_cnt[token_id] = 0
            if self.server_id == init_id:  # the server who initiates the token-based broadcast
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

    ############################ Report ########################################

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
