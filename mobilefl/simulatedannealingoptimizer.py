import time
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder  # type: ignore

from mobilefl.config import Config
from mobilefl.data.data import Data
from mobilefl.data.subset import CustomSubset
from mobilefl.server import Server
from mobilefl.simulateconfig import SimulateConfig

np.random.seed(0)
torch.manual_seed(0)


class SimulatedAnnealingOptimizer:
    # The reason to use SimulatedAnnealingOptimizer:
    # a. To use 1: moving client; 2: choose a client to commucation alternatively with servers
    # b. To choose: How many clients and which clients
    # a and can be solved by evaluating the iidness when doing so.
    def __init__(
        self,
        data: Data,
        servers: List[Server],
        target_server_id: int,
        target_servers: List[Server],
        config: Config,
        move_from_servers: bool,
    ) -> None:
        self.servers = servers
        self.data = data
        self.target_server_id = target_server_id
        self.target_servers = target_servers
        self.config = config
        self.max_iterations = int(config.get("max_iterations"))
        self.move_from_servers = move_from_servers

    def evaluate_iidness(self, config: SimulateConfig, moving: bool) -> float:
        # The calculation of iidness is:
        # Both the target server and previous server requires to be calculated
        # The iidness = (iidness of s1 + iidness of s2 )/ 2 * baseline
        # The bigger results, the higher non-iidness

        # TODO: to figure out if we want to move more than 1 clients
        full_subset = CustomSubset(
            self.data.train,
            indices=range(len(self.data.train)),
            targets=self.data.train.targets,
            classes=self.data.train.classes,
        )
        flattened_data = full_subset.targets
        label_encoder = LabelEncoder()
        flattened_data_int = label_encoder.fit_transform(flattened_data)
        global_distribution = np.bincount(flattened_data_int)

        def calculate_kl_divergence(subsets: List[CustomSubset]) -> float:
            kl_divergences = []
            for subset in subsets:
                flattened_data = subset.targets
                # flattened_data = torch.cat(flattened_data, dim=0)
                # flattened_data = flattened_data.to(torch.int64)
                # flattened_data = flattened_data.numpy().reshape(-1)
                flattened_data_int = label_encoder.fit_transform(flattened_data)
                subset_distribution = np.bincount(flattened_data_int, minlength=len(global_distribution))
                kl_divergence = entropy(subset_distribution, qk=global_distribution)
                kl_divergences.append(kl_divergence)
            return np.mean(kl_divergences)

        target_server_subsets = config.server_dataset
        target_client_dataset = config.target_client_dataset
        move_from_server_dataset = config.move_from_server_dataset
        move_client_id = config.move_client_id
        if moving:
            target_server_subsets.extend([target_client_dataset])
            move_from_server_dataset_1 = move_from_server_dataset[:]
            move_from_server_dataset_1.pop(move_client_id)
            iidness = (
                calculate_kl_divergence(target_server_subsets) + calculate_kl_divergence(move_from_server_dataset_1)
            ) / 2
            print("Moving: iidness", 1 - iidness)
        else:
            target_server_subsets.extend([target_client_dataset])
            iidness = (
                calculate_kl_divergence(target_server_subsets) + calculate_kl_divergence(move_from_server_dataset)
            ) / 2
            print("Sharing:iidness", 1 - iidness)
        target_server_subsets = target_server_subsets[:-1]
        return 1 - iidness

    def calculate_global_speed(self, server: Server, client_id: int, moving: bool, move_from_server: Server) -> float:
        # The global speed of the system = total updates / all delays (including all communication delays..)

        _global_speed_before = (
            sum(Counter(server.train_dataset_server_level.targets).values())
            + sum(Counter(move_from_server.train_dataset_server_level.targets).values())
        ) / (sum(server.calculate_clients_speed(False)) + sum(move_from_server.calculate_clients_speed(False)))
        # print("before moving :global_speed,",global_speed_before)

        server.clients[server.clients_num] = move_from_server.clients[client_id]
        _client = server.clients[server.clients_num]
        # client.new_id = server.clients_num
        server.client_train_datasets.extend([move_from_server.client_train_datasets[client_id]])
        new_client_training_delay = server.calculate_clients_speed(True)[server.clients_num]
        # print("new_client_training_delay",new_client_training_delay)
        if new_client_training_delay <= 0:
            global_speed = new_client_training_delay
        else:
            # server.client_train_datasets.extend([move_from_server.client_train_datasets[client_id]])
            delay_list_server = server.calculate_clients_speed(True)
            total_delay_server = sum(delay_list_server)
            # for moving clients condition: speed = 1 / training delay
            # for alternative condition: speed = 1/(training delay with previous server + training delay with new server)
            if moving:
                client_train_datasets = move_from_server.client_train_datasets.pop(client_id)
                # print("client_train_datasets",client_train_datasets)
                move_from_server.clients.pop(client_id)
                move_from_server.clients_num = move_from_server.clients_num - 1
                training_delay_list = move_from_server.calculate_clients_speed(False)
                delay_list_server_list_moving_server = training_delay_list
                total_training_delay_previous_server = sum(delay_list_server_list_moving_server)
                global_delay = total_delay_server + total_training_delay_previous_server
                total_updates = sum(
                    Counter(target for subset in server.client_train_datasets for target in subset.targets).values()
                ) + sum(
                    Counter(
                        target for subset in move_from_server.client_train_datasets for target in subset.targets
                    ).values()
                )
                global_speed = total_updates / global_delay
                # restore move_from_server
                # print(move_from_server.client_train_datasets)
                # print("client_id",client_id)
                move_from_server.client_train_datasets.insert(client_id, client_train_datasets)
                move_from_server.clients[client_id] = server.clients[server.clients_num]
                move_from_server.clients_num = move_from_server.clients_num + 1
                # print(move_from_server.client_train_datasets)

            else:
                total_delay_previous_server = sum(move_from_server.calculate_clients_speed(moving))
                global_delay = total_delay_server + total_delay_previous_server
                total_updates = sum(
                    Counter(target for subset in server.client_train_datasets for target in subset.targets).values()
                ) + sum(
                    Counter(
                        target for subset in move_from_server.client_train_datasets for target in subset.targets
                    ).values()
                )
                global_speed = total_updates / global_delay
                # print("sharing :global_speed,",global_speed)

        server.client_train_datasets.pop()
        # client.new_id = None

        del server.clients[server.clients_num]
        # print("move_from_server.client_train_datasets",len(move_from_server.client_train_datasets))
        return global_speed

    # def calculate_global_speed(self, server, client_id, moving, move_from_server):
    #     # The global speed of one server is
    #     # 1/ maximum (2 * communication delay + training delay + 2)
    #     # owned by one of the clients in the server
    #     global_speed_before = min([1/x for x in server.calculate_clients_speed(False)]) + min([1/x for x in move_from_server.calculate_clients_speed(False)])
    #     print("before moving :global_speed,",global_speed_before)
    #     print("global speed of receiving server",min([1/x for x in server.calculate_clients_speed(False)]))
    #     print("global speed of giving server", min([1/x for x in move_from_server.calculate_clients_speed(False)]))

    #     server.clients[server.clients_num] = move_from_server.clients[client_id]
    #     new_client_training_delay = server.calculate_clients_speed(True)[server.clients_num]

    #     if new_client_training_delay <= 0:
    #         global_speed = 0
    #     else:
    #         speed_list_server = [1/x for x in server.calculate_clients_speed(True)]
    #         total_training_delay_server = min(speed_list_server)
    #         # for moving clients condition: speed = 1 / training delay
    #         # for alternative condition: speed = 1/(training delay with previous server + training delay with new server)
    #         if moving:
    #             move_from_server.clients.pop(client_id)
    #             move_from_server.clients_num = move_from_server.clients_num - 1
    #             training_delay_list = move_from_server.calculate_clients_speed(False)
    #             speed_list_moving_server = [1/x for x in training_delay_list]
    #             total_training_delay_previous_server = min(speed_list_moving_server)
    #             global_speed = total_training_delay_server + total_training_delay_previous_server
    #             move_from_server.clients[client_id] = server.clients[server.clients_num]
    #             move_from_server.clients_num = move_from_server.clients_num +1
    #             print("moving :global_speed,",global_speed)
    #             print("global speed of receiving server",total_training_delay_server)
    #             print("global speed of giving server", total_training_delay_previous_server)

    #         else:
    #             total_training_delay_previous_server = min([1/x for x in move_from_server.calculate_clients_speed(moving)])
    #             global_speed = total_training_delay_server + total_training_delay_previous_server
    #             print("sharing :global_speed,",global_speed)
    #             print("global speed of receiving server",total_training_delay_server)
    #             print("global speed of giving server", total_training_delay_previous_server)

    #     del server.clients[server.clients_num]
    #     return global_speed

    # def calculate_speed(self, server, client_id, moving, move_from_server):
    #     server.clients[server.clients_num] = move_from_server.clients[client_id]
    #     new_client_training_delay = server.calculate_clients_speed(True)[server.clients_num]

    #     if new_client_training_delay <= 0:
    #         speed = 0
    #     else:
    #         # for moving clients condition: speed = 1 / training delay
    #         # for alternative condition: speed = 1/(training delay with previous server + training delay with new server)
    #         if moving:
    #             speed = 1 / new_client_training_delay
    #         else:
    #             previous_delay = move_from_server.calculate_clients_speed(moving)[client_id]
    #             speed = 1/(new_client_training_delay + previous_delay)

    #     del server.clients[server.clients_num]
    #     return speed

    def performance_function(self, iidness: float, speed: float, coefficient: float) -> float:
        print("performance_function", iidness + coefficient * speed)
        return iidness + coefficient * speed

    def evaluate_performance(self, config: SimulateConfig, moving: bool) -> float:
        iidness = self.evaluate_iidness(config, moving)
        speed = self.calculate_global_speed(
            config.target_server, config.move_client_id, moving, config.move_from_server
        )
        if speed <= 0:
            return float("-inf")  # Ignore configurations with zero speed
        else:
            coefficient = 0.01

            return self.performance_function(iidness, speed, coefficient)

    def iterative_optimizer(self) -> Tuple[SimulateConfig, float, int]:
        step = 0
        target_server = self.servers[self.target_server_id]
        server_id_list = [i for i in range(len(self.servers))]

        # Initialize configuration to ensure the initial configuration has a positive speed
        current_config = None
        current_performance = float("-inf")
        moving_final = None

        print("---------Initiialize config and performance---------")
        for move_from_server_id in [server_id for server_id in server_id_list if server_id not in self.target_servers]:
            for move_client_id in range(len(self.servers[move_from_server_id].clients)):
                temp_config = SimulateConfig(self.servers, target_server, move_from_server_id, move_client_id)
                temp_performance = self.evaluate_performance(temp_config, True)
                if temp_performance != float("-inf"):
                    current_config = temp_config
                    current_performance = temp_performance
                    moving_final = True
                    break
            if current_performance != float("-inf"):
                break

        if current_config is None:
            raise ValueError("No valid initial configuration found")

        print("Current initial config:", current_config)
        print("Current performance:", current_performance)

        print("-----------Start iterations----------")

        for _ in range(self.max_iterations + 1):
            print("step", step)
            new_config = self.perturb_configuration(current_config)
            print("move_client_id", new_config.move_client_id)
            for moving in [True, False]:
                moving = True  # only consider moving methods
                new_performance = self.evaluate_performance(new_config, moving)
                if new_performance == float("-inf"):
                    step += 1
                    continue  # ingore negative value and zero
                if new_performance > current_performance:
                    current_config = new_config
                    current_performance = new_performance
                    moving_final = moving
            step += 1

        print("choose", moving_final)

        return current_config, current_performance, step

    def perturb_configuration(self, config: SimulateConfig) -> SimulateConfig:
        # Again, for all error with config, I know this is wrong but cannot be bothered to fix it now
        _target_servers = self.config.get("target_servers")
        # move_from_server_id = np.random.choice([id for id in [i for i in range(len(self.servers))] if (id != self.target_server_id and id not in target_servers and id not in self.move_from_servers)])
        move_from_server_id = np.random.choice(
            [server_id for server_id in [i for i in range(len(self.servers))] if server_id != self.target_server_id]
        )

        if move_from_server_id == config.move_from_server_id:
            move_client_id = np.random.choice(
                [
                    client_id
                    for client_id in [i for i in range(config.servers[move_from_server_id].clients_num)]
                    if (client_id != config.move_client_id)
                ]
            )
        else:
            move_client_id = np.random.choice(
                [client_id for client_id in [i for i in range(config.servers[move_from_server_id].clients_num)]]
            )
        return SimulateConfig(config.servers, config.target_server, move_from_server_id, move_client_id)

    def cosine_distance_method(self) -> None:
        server_id_list = [i for i in range(len(self.servers))]
        target_server = self.servers[self.target_server_id]
        for move_from_server_id in [server_id for server_id in server_id_list if server_id not in self.target_servers]:
            for move_client_id in self.servers[move_from_server_id].clients:
                print("move_client_id", move_client_id)
                move_client = self.servers[move_from_server_id].clients[move_client_id]
                # target_server.clients[target_server.clients_num] = move_client
                move_client.model.load_state_dict(target_server.aggregator.model.state_dict())
                # target_server.clients_num += 1
                if move_client.train():
                    target_server.update_buffer.append(
                        (
                            move_client.update,
                            target_server.clients_num,
                            target_server.age - move_client.age,
                        )
                    )
                else:
                    print(
                        f"Client {move_client.client_id} from Server {move_from_server_id} failed to train with Server {self.target_server_id}"
                    )
                    continue
                _cosvalue = target_server.aggregator.aggregate(
                    target_server.update_buffer,
                    cosine=True,
                    aggregation_method="fedsgd",
                )

    def find_best_clients(self) -> Tuple[int, int]:
        start = time.time()
        if self.config.get("cosine"):
            self.cosine_distance_method()
        best_configuration, best_value, step = self.iterative_optimizer()
        end = time.time()

        print("--------------------------------")
        print("--------iterative_optimizer -------")
        print("--------------------------------")
        print("Configurations examined: {}    time needed:{}".format(step, end - start))
        print(
            "Best Configuration: For server {}, the best client is from server {}, the client_id is {}, the performance: {}".format(
                self.target_server_id,
                best_configuration.move_from_server_id,
                best_configuration.move_client_id,
                best_value,
            )
        )

        return best_configuration.move_from_server_id, best_configuration.move_client_id
