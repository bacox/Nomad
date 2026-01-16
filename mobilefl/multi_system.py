import os
import pickle
import random
import shutil
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns
import torch
import torchtext  # type: ignore
from scipy.spatial import cKDTree
from scipy.stats import entropy
from shapely.geometry import Point  # type: ignore
from sklearn.manifold import MDS  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from torch.utils.data import DataLoader

from mobilefl.ageToken import AgeToken
from mobilefl.client import Client
from mobilefl.color import colors
from mobilefl.config import Config
from mobilefl.data.data import Data
from mobilefl.data.subset import CustomSubset
from mobilefl.event_system import Event, EventSystem, EventType
from mobilefl.log_tools.logger import Logger
from mobilefl.log_tools.logging_style import content, line
from mobilefl.server import Server
from mobilefl.types import ClientAllocationTracker, EventHistory, MEvent
from mobilefl.utils import calculate_delay_matrix, sum_dicts
from mobilefl.voronoi import VoronoiMap

# Disable the deprecation warning message
# torchtext.disable_torchtext_deprecation_warning()  # noqa

os.environ["WANDB_SILENT"] = "true"
torch.manual_seed(0)


def evaluate_data_distribution(
    servers: List[Server], unallocated_clients: List[Client] = [], out_path: Path = Path(".")
) -> None:

    print(f'{"=" * 50}\nEvaluating data distribution across servers:\n{"=" * 50}')
    print(f"{servers}")

    label_data = []
    client_data = {}
    label_keys = set()
    for server in servers:
        for client in server.clientsSet:
            print(f"Client {client.unique_id} is at server {server.server_id}")
            values = client.get_data_label_count()
            client_data[client.unique_id] = (server.server_id, values)
            label_keys.update(values.keys())

            # print(f"Client {client.unique_id} label count: {values}")
    for uclients in unallocated_clients:
        print(f"Unallocated client {uclients.unique_id} has label count: {uclients.get_data_label_count()}")
        values = uclients.get_data_label_count()
        client_data[uclients.unique_id] = (-1, values)
        label_keys.update(values.keys())
    print(f"Label keys: {label_keys}")
    label_keys = sorted(label_keys)
    print(f"Sorted label keys: {label_keys}")
    for client_id, (server_id, values) in client_data.items():
        row = [client_id, server_id] + [values.get(label, 0) for label in label_keys]
        label_data.append(row)
    label_df = pd.DataFrame(label_data, columns=["client_id", "server_id"] + label_keys)
    print(f"Label DataFrame:\n{label_df}")

    # Save the label data to a CSV file
    label_df.to_csv(out_path / "label_distribution.csv", index=False)
    # Plot the label distribution
    plt.figure(figsize=(12, 8))

    sns.barplot(label_df, x="client_id", y=label_keys[0], hue="server_id", palette="viridis")
    plt.savefig(out_path / "label_distribution.png")
    plt.close()

    grouped = label_df.groupby("server_id").sum().drop(columns=["client_id"])
    print(f"Grouped label counts by server:\n{grouped}")

    # Plot a barplot with the labels on the x-axis and the counts on the y-axis and hue by server_id
    plt.figure(figsize=(16, 8))
    grouped.plot(kind="bar", stacked=True)
    plt.title("Label Distribution Across Servers")
    plt.xlabel("Server ID")
    plt.ylabel("Label Count")
    plt.legend(title="Labels")
    # Legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path / "label_distribution_by_server.png")
    plt.close()

    # Loop over the data and print for every client the amount of labels it has
    # Collect the min, max, and average label counts for each client and standard deviation
    print("*" * 50 + "\nClient Label Counts:\n" + "*" * 50)
    label_counts = []
    for client_id, (server_id, values) in client_data.items():
        total_labels = sum(values.values())
        label_counts.append(total_labels)
        print(f"Client {client_id} (Server {server_id}) has {total_labels}\t\t labels: {values}")
    if label_counts:
        print("*" * 50 + "\nSummary of Client Label Counts:\n" + "*" * 50)
        print(f"Total clients: {len(label_counts)}")
        print(f"Min labels: {min(label_counts)}")
        print(f"Max labels: {max(label_counts)}")
        print(f"Average labels: {sum(label_counts) / len(label_counts)}")
        print(f"Standard deviation of labels: {np.std(label_counts)}")

        # Plot the distribution of label counts per label with standard deviation and hue by server_id
        # The labels are on  the x-

    # exit()


class MultiSystem:
    def __init__(self, config_file: Union[str, Path], verbose: bool = False) -> None:
        assert isinstance(config_file, Path), "config_file should be a Path object"
        self.config_file: Path = config_file
        self.clients: List[Client] = []
        self.servers: List[Server] = []
        self.data: Data
        self.plotter = None
        self.num_clients: int = 0
        self.num_servers: int = 0
        self.leader: Server = None  # type:ignore
        self.cur_round: int = 0
        self.heap = []  # main loop time heap for asynchronous server communication
        self.period: int = 0
        self.print_flag: bool = False
        self.hier_period: float = None  # type:ignore
        self.logger: Logger
        self.coordinates = None  # build the coordinates based on the matrix
        self.num_clients_per_server: List[int] = []  # index means the id of each server
        self.servers_location = []  # to store the location of each server
        # to store the associate area of each server which ensures the shortest distance from the client to the server it belongs to
        # the index means the area id (server.area) of each server
        self.server_vor_areas = []
        self.clients_speed_per_server = (
            {}
        )  # index means the server id; to store the speed of each clients for every server
        self.var_control: bool = False  # to mark if the config.json file has been modified before

        self.updates = {}  # to store the updates which is required to be written in the config.json
        self.selected_clients = {}  # to store the chosen clients with their target servers
        self.server_iidness = {}  # to store the iidness of each server level
        self.verbose: bool = verbose
        self.frequency_matrix = None

        self.client_assignment_history = []
        self.client_allocation_tracker: ClientAllocationTracker  # to track the client allocation history
        self.event_system: EventSystem = EventSystem()  # to manage the events in the system
        self.event_history: EventHistory = EventHistory()  # to store the event history
        self.alternative_server_locations = []

        self.available_clients: List[Client] = []  # Store clients that are not assigned to any server
        self.rebalance_data: List[Tuple[float, int, int, float]] = (
            []
        )  # Store rebalance data as (time, round_id, server_id, total_weighted_tvd)

        # @TODO: This should be removed in the end
        self.global_label_count = {}  # to store the global label count of all the clients
        self.base_result_dir: Path

        # cevents = []
        # cevents = generate_events("linear", EventType.CHURN, EventUnit.ROUND, 100, 10)
        # jevents = []
        # # jevents = generate_events("uniform", EventType.JOIN, EventUnit.ROUND, 100, 30)

        # # For debugging purposes only
        # for ev in cevents + jevents:
        #     self.event_system.add_event(ev.when, ev.unit, ev.event_type)

    def loadConfig(self) -> Config:
        print("LOADING CONFIGURATION...")
        self.config = Config(self.config_file)

        if self.verbose:
            print("==========================================================================")
            print("Configuration:")
            for key, value in self.config.dct.items():
                print(f"{key}: {value}")

        if self.config.get("cuda"):
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")

        self.var_control = bool(self.config.get("var_control"))
        # var_control = true: load the config file
        if self.var_control:
            self.num_clients_per_server = int(self.config.get("num_clients_per_server"))

        else:
            num_clients = int(self.config.get("num_clients"))
            self.num_clients_per_server = [0] * int(self.config.get("num_servers"))
            for server_index in range(int(self.config.get("num_servers"))):
                # config_key = "server_" + str(server_index) + "_clients"
                # print(f"Config key for server {server_index}: {config_key}")
                # print(f"Number of clients for server {server_index}: {self.config.get(config_key)}")
                # self.num_clients_per_server[server_index] = num_clients * self.config.get(config_key)
                starting_clients = self.config.get("num_start_clients", num_clients)
                self.num_clients_per_server[server_index] = starting_clients // int(self.config.get("num_servers"))
            updates = {}
            updates["num_clients_per_server"] = self.num_clients_per_server
            self.config.write_config(updates)

            self.updates["num_clients_per_server"] = self.num_clients_per_server

        if self.config.get("client_async") and not self.config.get("server_async"):
            # self.calculate_period()
            print(f"MultiSync period: {self.calculate_period()}")

        self.var_control = bool(self.config.get("var_control"))
        self.base_result_dir = Path(f"./results/{self.config.get('result_file')}/{self.config.get('name')}")
        self.alternative_server_locations = self.config._world_config_dct.get("alternative_servers")
        self.result_file_name = (
            "results/"
            + self.config.get("name")
            + "_"
            + str(self.config.get("num_servers"))
            + "x"
            + str(self.num_clients_per_server)
            + "_"
            + strftime("%m%d_%H%M%S", gmtime())
            + ".txt"
        )
        self.logger = Logger(
            f"results/{self.config.get('result_file')}/{self.config.get('name')}",
            self.config,
        )
        print("==========================================================================")
        return self.config

    def load_events(self) -> None:
        if self.config._system_events:
            for ev in self.config._system_events:
                self.event_system.add_event(ev.when, ev.unit, ev.event_type)
            print(
                f"Loaded {len(self.event_system.events)} events from the configuration file {self.event_system.events}."
            )
        else:
            print("No events found in the configuration file.")

    def getData(self) -> None:
        print("PREPARING DATASETS...")
        self.data = Data(self.config)

    def handleEvent(self, ev: Event) -> None:
        """
        Handle the event based on its type.
        :param ev: The event to handle.
        """
        if ev.event_type == EventType.CHURN:
            self.handleChurnEvent(ev)
        elif ev.event_type == EventType.JOIN:
            self.handleJoinEvent(ev)
        else:
            raise ValueError(f"Unknown event type: {ev.event_type}")

    def handleChurnEvent(self, ev: Event):
        print(f"Handling churn event at {ev.when} with unit {ev.unit} and type {ev.event_type}")

        # Randomly pick a server to remove a client from that has at least 2 clients
        if len(self.servers) == 0:
            print("No servers available to handle churn event.")
            return
        eligible_servers = [s for s in self.servers if len(s.clientsSet) > 1]
        if not eligible_servers:
            print("No eligible servers with more than one client to handle churn event.")
            return
        server = random.choice(eligible_servers)
        curr_time = server.cur_time
        # Randomly pick a client from the server's client set
        client: Client = random.choice(server.clientsSet)

        print(f"Removing client {client.unique_id} from server {server.server_id} due to churn event.")

        # Remove the client from the server's client set
        [removed_client, time_left] = server.remove_client(client.unique_id)

        self.available_clients.append(removed_client)

        # Add the event to the event history
        self.event_history.add_event(
            MEvent(
                curr_time,
                "system",
                "churn",
                round=self.cur_round,
                description=f"remove client {removed_client.unique_id} from server {server.server_id}",
            )
        )

    def handleJoinEvent(self, ev: Event):
        print(f"Handling join event at {ev.when} with unit {ev.unit} and type {ev.event_type}")

        # If there are available clients, randomly assign one to a server
        if not self.available_clients:
            print("No available clients to handle join event.")
            return
        # Randomly pick a server to add a client to
        random_server = random.choice(self.servers)

        # Randomly pick a client from the available clients
        client: Client = random.choice(self.available_clients)

        print(f"Adding client {client.unique_id} to server {random_server.server_id} due to join event.")
        # Remove the client from the available clients list
        self.available_clients.remove(client)
        # Add the client to the server's client set
        random_server.add_client(client)
        # Add the event to the event history
        self.event_history.add_event(
            MEvent(
                random_server.cur_time,
                "system",
                "join",
                round=self.cur_round,
                description=f"add client {client.unique_id} to server {random_server.server_id}",
            )
        )

    def calculate_frequency_matrix(self) -> np.ndarray:  # type: ignore
        client_delays = np.array([client.local_delay for client in self.clients])
        client_locations = np.array([client.location for client in self.clients])
        print(f"Server locations: {[server.location for server in self.servers]}")
        server_locations = np.array([server.location for server in self.servers])
        client_server_delays = calculate_delay_matrix(client_delays, client_locations, server_locations)

        return client_server_delays

    def createClients(self, server: Server, unique_client_idx: int) -> Tuple[int, List[Client]]:
        """
        Create clients
        The number of slow clients is num_clients * slow_ratio ...
        """

        print(f"Creating {server.clients_num} clients for server {server.server_id}...")
        updates = {}
        # get the range of the x and y axies
        # polygon_vertices = self.server_vor_areas[server.area]
        # polygon = Polygon(polygon_vertices)
        num_clients = server.clients_num
        # min_x, min_y, max_x, max_y = polygon.bounds
        # _x_range = (min_x, max_x)
        # _y_range = (min_y, max_y)
        locations = []
        clients = []
        polygon = []
        client_local_delays = []
        if self.var_control:
            locations = self.config.get(f"server_{server.server_id}_clientset_locations")
        else:

            world_cfg_clients = [
                x for x in self.config._world_config_dct["clients"] if x["assigned_server"] == server.server_id
            ]
            locations = [(x["x"], x["y"]) for x in world_cfg_clients]
            client_local_delays = [x["delay"] for x in world_cfg_clients]
            # print(f"Locations from world config: {locations}")
            # # Read from world config
            # i = 0
            # while i < num_clients:
            #     location = (random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]))
            #     if polygon.contains(Point(location)):
            #         locations.append(location)
            #         i = i + 1
            updates[f"server_{server.server_id}_clientset_locations"] = locations
            self.config.write_config(updates)

        # train_loaders, test_loaders = self.data.get_client_data_loaders()
        client_speeds = []
        if self.config.get("samespeed"):
            client_speeds = ["fast"] * num_clients
        elif not self.config.get("gaussianclient"):
            client_speeds = []
            for i in range(num_clients):
                if i < int(num_clients * float(self.config.get("slow_ratio"))):
                    client_speeds.append("slow")
                elif i < int(
                    num_clients * (float(self.config.get("slow_ratio")) + float(self.config.get("medium_ratio")))
                ):
                    client_speeds.append("medium")
                else:
                    client_speeds.append("fast")

        if len(client_speeds):
            # i = 0
            for idx, client_type in enumerate(client_speeds):
                # while i < num_clients:
                print(f"Location {idx}: {locations}")
                print(f"Creating client {idx} for server {server.server_id}...")
                location = locations[idx]
                client_local_delay = client_local_delays[idx]
                if True:
                    # if polygon.contains(Point(location)) or True:
                    client = Client(
                        self.config,
                        idx,
                        unique_client_idx,
                        server.server_id,
                        location,
                        client_type=client_type,
                        vocab_size=len(self.data.vocab),
                    )
                    client.local_delay = client_local_delay
                    # i = i + 1
                    self.clients.append(client)
                    clients.append(client)
                    self.num_clients += 1
                    unique_client_idx += 1

            if not self.config.get("server_heter"):
                # shuffle the clients
                random.shuffle(self.clients)

        # ndom.shuffle(self.clients)
        else:
            try:
                with open(f"clients/client{self.config.get('num_clients')}.pkl", "rb") as f:
                    weights = pickle.load(f)
                    print(weights)
                # for i in range(self.config.get("num_clients")):
                i = 0
                while i < num_clients:
                    location = locations[i]
                    if polygon.contains(Point(location)):
                        delay = weights[i] * self.config.get("training_delay")
                        client = Client(
                            self.config,
                            i,
                            unique_client_idx,
                            server.server_id,
                            location,
                            client_type="gaussian",
                            gaussian_mu=delay,
                            vocab_size=len(self.data.vocab),
                        )
                        # client = Client(self.config, i, server.server_id,location, type="gaussian", train_loader=train_loaders[i], test_loader=test_loaders[i],  gaussian_mu=delay, vocab_size=len(self.data.vocab))
                        self.clients.append(client)
                        clients.append(client)
                        i = i + 1
                        self.num_clients += 1
                        unique_client_idx += 1
                        if i % 5 == 0:
                            print(f"Client {i} training time: {delay}")
            except FileNotFoundError:
                print(f"no existing client{self.config.get('num_clients')}.pkl file, creating new one...")
                # gaussian clients with mean = 0, std = 1
                mu = 1
                # TODO: gaussian communication delay
                delays = []
                # for i in range(self.config.get("num_clients")):
                i = 0
                while i < num_clients:
                    delay = max(0.1 * mu, np.random.normal(mu, 0.4 * mu))
                    delays.append(delay)
                    gaussian_mu = delay * float(self.config.get("training_delay"))
                    location = locations[i]
                    if polygon.contains(Point(location)):
                        client = Client(
                            self.config,
                            i,
                            unique_client_idx,
                            server.server_id,
                            location,
                            client_type="gaussian",
                            gaussian_mu=gaussian_mu,
                            vocab_size=len(self.data.vocab),
                        )
                        self.clients.append(client)
                        clients.append(client)
                        i = i + 1
                        self.num_clients += 1
                        unique_client_idx += 1
                        if i % 5 == 0:
                            print(f"Client {i} training time: {delay}")
                with open(f"clients/client{self.config.get('num_clients')}.pkl", "wb") as f:
                    pickle.dump(delays, f)
                print(f"Generated {self.config.get('num_clients')} clients")

        # x, y = polygon.exterior.xy
        # plt.plot(x, y, color="blue", label="Polygon")
        # for point in self.clients:
        #     plt.plot(point.location[0], point.location[1], "ro")
        # plt.title(f"Random Points Inside Polygon server {server.server_id} in area {server.area}")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.legend()
        # plt.show()

        return unique_client_idx, clients

    def createServer(
        self,
        server_id: int,
        test_loader: DataLoader,
        token_start: int,
        random_areas: List[int],
        train_dataset_server_level: CustomSubset,
    ) -> Server:
        server_world_cfg = [x for x in self.config._world_config_dct["servers"] if x["id"] == server_id][0]
        server_location = (server_world_cfg["x"], server_world_cfg["y"]) if server_world_cfg else (0, 0)
        print(f"{server_location=}")

        server = Server(
            self.config,
            server_id,
            test_loader,
            logger=self.logger,
            vocab_size=len(self.data.vocab),
            location=server_location,
        )
        server.frequency_matrix = self.frequency_matrix
        # load the training delay (average speed of each server's clients)
        average_speed = self.config.get("server_" + str(server_id) + "_training_delay")
        if self.config.get("server_iid"):
            server.train_dataset_server_level = train_dataset_server_level
        else:
            server.train_dataset_server_level = train_dataset_server_level[server_id]  # type: ignore
        server.server_level_speed = average_speed
        server.area = random_areas[server_id]  # the area of the map
        # set the required number of clients for this server
        server.clients_num = int(self.num_clients_per_server[server_id])

        print(f"Number of clients for server {server_id}: {server.clients_num}")
        assert (
            server.clients_num > 0
        ), f"Server {server_id} has no clients assigned. Value of self.num_clients_per_server[i]: {self.num_clients_per_server[server_id]}"
        server.print_flag = self.print_flag
        if server.location is None or server.location == (0, 0):
            raise NotImplementedError(
                'Currently "createServer" requires a valid server location. Please set "x" and "y" in the world config for each server.'
            )
            server.set_server_location(self.coordinates)  # get the location of the server in the map
        self.servers_location.append(server.location)
        if self.period > 0:
            server.period = self.period
        self.servers.append(server)
        self.num_servers += 1
        if server_id == token_start:
            server.token = AgeToken()
        if self.hier_period is not None:
            server.hier_period = int(self.hier_period)
            server.leader = self.leader
        return server

    def introduce_new_server(self, curr_time: float = 0) -> None:
        """
        Introduce a new server to the system.
        Steps:
        1. Find next available server ID.
        2. Get new server location.
        3. Create a new server with the new ID and location.
        4. Add the new server to the list of servers.
        5. Update the servers' locations and frequency matrix.
        6. Assign existing clients to the new server.

        @TODO:
        -
        """
        print("Introducing a new server...")
        # Find the next available server ID
        new_server_id = self.num_servers
        print(f"New server ID: {new_server_id}")

        # Get new server location. Assume the new server is not in self.coordinates yet.
        assert self.alternative_server_locations is not None, "Alternative server locations are not set in the config."
        assert len(self.alternative_server_locations) > 0, "No alternative server locations available."

        # Get a random location from the alternative server locations
        new_server_location = random.choice(self.alternative_server_locations)

        # Remove the chosen location from the list to avoid duplicates
        self.alternative_server_locations.remove(new_server_location)

        new_server_location = (new_server_location["x"], new_server_location["y"])
        print(f"New server location: {new_server_location}")

        test_loader = self.data.get_server_data_loaders()

        new_server = Server(
            self.config,
            new_server_id,
            test_loader,
            logger=self.logger,
            vocab_size=len(self.data.vocab),
            location=new_server_location,
        )

        new_server.logger.init_server(new_server_id)
        new_server.set_event_history(self.event_history)

        self.num_servers += 1
        # @TODO: This will be problematic when we can remove servers
        if self.period > 0:
            new_server.period = self.period
        self.num_servers += 1
        if self.hier_period is not None:
            new_server.hier_period = int(self.hier_period)
            new_server.leader = self.leader
        self.servers_location.append(new_server_location)
        new_server.print_flag = self.print_flag

        self.servers.append(new_server)

        self.frequency_matrix = self.calculate_frequency_matrix()

        new_server.frequency_matrix = self.frequency_matrix
        new_server.frequency_matrix = self.frequency_matrix

        # # Update the servers' locations and frequency matrix
        self.servers_location.append(new_server_location)

        print(f"Frequency matrix shape: {self.frequency_matrix.shape}")

        print(f"New server {new_server_id} introduced at location {new_server_location}.")

        print(f"Updated frequency matrix {self.frequency_matrix}.")

        # @TODO: Assign existing clients to the new server
        servers_with_clients = [s for s in self.servers if len(s.clientsSet) > 2]
        # Get 2 clients from some servers with clients
        found_clients = []

        for s in self.servers:
            print(f"Server {s.server_id} has {len(s.clientsSet)} clients.")

        for server in servers_with_clients:
            print(f"Searching for clients in server {server.server_id} with {len(server.clientsSet)} clients.")
            if len(found_clients) >= 2:
                break
            if len(server.clientsSet) == 3:
                to_select = 1
            else:
                to_select = 2
            # Randomly select 2 clients from the server's client set
            selected_clients = random.sample(server.clientsSet, 2)
            found_clients.extend(selected_clients)
            # Remove the selected clients from the server's client set
            server.remove_client(selected_clients[0].unique_id)
            if to_select == 2:
                server.remove_client(selected_clients[1].unique_id)

        print(f"Found clients to assign to the new server: {found_clients}")

        # Assign the found clients to the new server
        assert len(found_clients) > 0, "No clients found to assign to the new server."
        for new_client in found_clients:
            new_server.add_client(new_client)  # Add the client to the new server's client set
        print(f"New server {new_server_id} has {len(new_server.clientsSet)} clients assigned.")

        # Update the servers in all the servers
        for server in self.servers:
            server.set_servers(self.servers)  # type:ignore

        self.event_history.add_event(
            MEvent(
                curr_time,
                "system",
                "introduce_server",
                round=self.cur_round,
                description=f"new server {new_server_id} introduced",
            )
        )

    def createServers(self, train_dataset_server_level: CustomSubset, test_dataset_server_level: CustomSubset) -> None:

        # alternative_server_locations = self.config._world_config_dct.get("alternative_servers")

        # print(f"Alternative server locations: {alternative_server_locations}")
        # exit()

        print("CREATING SERVERS...")
        # Create the frequency_matrix
        world_cfg_clients = self.config._world_config_dct["clients"]

        frequency_matrix = []
        for client in world_cfg_clients:
            frequency_matrix.append(client["latencies"])
        # Change into a numpy array and save it to a csv file
        print(f"Frequency matrix shape: {len(frequency_matrix)} x {len(frequency_matrix[0])}")
        self.frequency_matrix: np.ndarray = np.array(frequency_matrix)

        updates = {}
        test_loader = self.data.get_server_data_loaders()
        token_start: int = np.random.randint(0, self.config.get("num_servers"))
        print(f"Token starts in Server {token_start}")

        try:
            if self.config.get("leader"):
                # leader's id = the number of servers
                self.leader = Server(
                    self.config,
                    self.config.get("num_servers"),
                    test_loader,
                    logger=self.logger,
                    vocab_size=len(self.data.vocab),
                )
                self.hier_period = self.config.get("hier_period")
        except Exception:
            print("NO hier_period or leader server")
            pass

        # Create a list of unique random values for server areas
        if self.var_control:
            # If var_control is true, load the server_areas from the config file
            random_areas = self.config.get("server_areas")
        else:
            # If var_control is false, generate random areas for the servers
            random_areas = random.sample(range(self.config.get("num_servers")), self.config.get("num_servers"))
            # self.updates["server_areas"] = random_areas
            updates["server_areas"] = random_areas
            self.config.write_config(updates)

        unique_client_idx = 0

        for i in range(self.config.get("num_servers")):
            server = self.createServer(
                i,
                test_loader,
                token_start,
                random_areas,
                train_dataset_server_level,
            )

            # @TODO: Extract this part out of this functions
            unique_client_idx, created_clients = self.createClients(server, unique_client_idx)

            # Take the first n starting clients from the created_clients list

            # remaining_clients = created_clients[num_start_clients:]
            # created_clients = created_clients[:num_start_clients]

            server.clientsSet = created_clients  # set the clients for this server

            # Add the remaining clients to the available clients list
            # self.available_clients.extend(remaining_clients)

        # create a kdtree for finding the shortest distance between each clients and server
        print("build kd tree")
        # print(servers_location)
        _kdtree = cKDTree(self.servers_location)

        # set the clients to the nearest server
        clientsLocation = []
        # print(f"self.clients{self.clients}")
        for client in self.clients:
            clientsLocation.append(client.location)

        # print(f"clientlocationlist{clientsLocation}")
        # nearest_server_indices = kdtree.query(clientsLocation)[1]
        # for i, server_id in enumerate(nearest_server_indices):
        #     # print(f"client {i+1} assigned to server in area {server_area}")
        #     for server in self.servers:
        #         if server.server_id == server_id:
        #             server.clientsSet.append(self.clients[i])

        # world_cfg_servers = self.config._world_config_dct["servers"]
        # for client in self.clients:
        #     unique_client_id = client.unique_id

        #     for s in world_cfg_servers:
        #         server_id = s["id"]
        #         if unique_client_id in s["assigned_clients"]:
        #             [s for s in self.servers if s.server_id == server_id][0].clientsSet.append(client)
        #             print(f"Client {unique_client_id} assigned to server {server_id}")
        #             break
        #     else:
        #         print(f"Client {unique_client_id} not assigned to any server in world config.")
        #         # if s["id"] == client.server_id:
        #         #     if unique_client_id in s["assigned_clients"]:
        #         #         # Add to server
        #         #         for server in self.servers:
        #         #             if server.server_id == client.server_id:
        #         #                 server.clients.append(client)
        #         #                 # print(f"Client {unique_client_id} assigned to server {server.server_id}")
        #         #                 break
        # print(f"World config servers: {world_cfg_servers}")
        # for server in self.servers:
        #     print(f"Server {server.server_id} with clients {server.clients}")
        #     for c in server.clients.values():
        #         server.clientsSet.append(c)
        # server.clientsSet = list(set(server.clientsSet))  # remove duplicates
        for server in self.servers:
            print(f"Server {server.server_id} has {len(server.clientsSet)} clients")

            server.set_clients(server.clientsSet)
            # set the dataloader for this servers' clients
            server.set_clients_dataloader(
                self.data,
                len(server.clientsSet),
                train_dataset_server_level,
                test_dataset_server_level,
                self.base_result_dir,
            )

            # Set the "shared" event history for each server
            server.set_event_history(self.event_history)

            # Update the speed of clients for the server based on the configuration settings.
            # This block checks various conditions to determine whether to update the client speeds:
            # - If none of the configurations for moving a client, averaging models, or alternate sharing are enabled.
            # - Or if the configuration for late movement or late alternate sharing is enabled.
            if (
                (
                    not self.config.get("move_a_client")
                    and not self.config.get("avg_model")
                    and not self.config.get("alternate_new")
                )
                or self.config.get("move_late")
                or self.config.get("alternate_late")
            ):
                # Set the speed of clients for the current server.
                self.clients_speed_per_server[server.server_id] = server.set_clients_speed()
            server.set_servers(self.servers)

            # Remove clients if num_start < num_clients_per_server
            if self.config.get("dynamic_clients", False):
                num_start_clients = self.config.get("dynamic_start_clients_per_server", server.clients_num)
                assert (
                    num_start_clients > 0
                ), f"Server {server.server_id} has no starting clients assigned. Value of num_start_clients: {num_start_clients}"
                if num_start_clients < server.clients_num:
                    client_ids_to_move = [x.client_id for x in server.clientsSet[num_start_clients:]]
                    for c_id in client_ids_to_move:
                        client, _rem_time = server.remove_client(c_id)
                        self.available_clients.append(client)
                print(f"Server {server.server_id} has {len(server.clientsSet)} clients after removing excess clients.")
            else:
                print(
                    f"Dynamic clients are not enabled. Server {server.server_id} has {len(server.clientsSet)} clients."
                )

        # start = 0
        # over = self.config.get('num_clients') % self.config.get('num_servers')
        # print('over is' , over)
        # # all the servers know each other
        # for i, server in enumerate(self.servers):
        #     if self.config.get("num_servers") == 1:
        #         j = 0
        #         for client in self.clients:
        #             client.area = j % 4
        #             j += 1
        #     add = 0
        #     if i < over:
        #         add = 1
        #     clients = self.clients[start: start + self.config.get("num_clients_per_server") + add]
        #     start = start + self.config.get("num_clients_per_server") + add
        #     server.set_clients(clients)
        #     server.set_servers(self.servers)

        if self.hier_period is not None:
            self.leader.set_servers(self.servers)

        for server in self.servers:
            st = ""
            for client in server.clients.values():
                st += str(client.area)
            _num = len(server.clients)
            # print(f"Server {server.server_id} in area {server.area} has {num} clients: {st}")

        # Record label count for each server and each client into a pandas DataFrame
        label_data = []
        client_id_count = 0
        for server in self.servers:
            for client in server.clients.values():
                lc = client.get_data_label_count()

                for label, count in lc.items():
                    label_data.append(
                        {
                            "server_id": server.server_id,
                            "client_id": client_id_count,
                            "label": label,
                            "count": count,
                        }
                    )
                client_id_count += 1

        if not os.path.exists(f"results/{self.config.get('result_file')}"):
            os.makedirs(f"results/{self.config.get('result_file')}")
        # Save the frequency matrix to a CSV file
        np.savetxt(
            f"results/{self.config.get('result_file')}/{self.config.get('name')}_frequency_matrix.csv",
            self.frequency_matrix,
            delimiter=",",
        )
        # np.ndarray(frequency_matrix).tofile(
        #     f"results/{self.config.get('result_file')}/{self.config.get('name')}_frequency_matrix.csv",
        #     sep=",",
        # )

        label_df = pd.DataFrame(label_data)
        # Save the DataFrame to a CSV file
        label_df.to_csv(
            f"results/{self.config.get('result_file')}/{self.config.get('name')}_label_count.csv",
            index=False,
        )

        print(f"Frequency matrix is: {self.frequency_matrix.shape[0]} x {self.frequency_matrix.shape[1]}")
        print(f"Frequency matrix = {self.frequency_matrix}")

    def calculate_period(self) -> int:
        # calculate the time of one period
        if self.config.get("sync_period") is not None:
            self.period = int(self.config.get("sync_period"))
            return self.config.get("sync_period")
        # n is the maxmum number of clients
        n: int = np.max(self.num_clients_per_server)
        m = self.config.get("update_per_sync")
        time_per_update = self.config.get("training_delay")
        if self.config.get("comm_delay") != 0:
            # round(m / (n/time_per_update)): the time of every server's each clients
            period = (
                round(m / (n / time_per_update))
                + self.config.get("globalsync_delay")
                + 400
                + 2 * self.config.get("comm_delay")
            )
        else:
            period = (
                round(m / (n / time_per_update))
                + self.config.get("globalsync_delay")
                + 2 * self.config.get("comm_delay")
            )
        self.period = period

        # n = self.config.get('num_clients_per_server')
        # time_per_fast = self.config.get('training_delay') * self.config.get('fast_mu')
        # time_per_medium = self.config.get('training_delay') * self.config.get('medium_mu')
        # time_per_slow = self.config.get('training_delay') * self.config.get('slow_mu')
        # time_per_update = self.config.get('training_delay')
        # num_fast = n * self.config.get('fast_ratio')
        # num_medium = n * self.config.get('medium_ratio')
        # num_slow = n * self.config.get('slow_ratio')

        return period

    # The Voronoi diagram splits the plane into polygonal regions centered on each server,
    # such that the points in each region have the closest distance to the corresponding server.
    # We create the required amount of clients randomly in this region
    def createClientsRegion(self, coordinates: np.ndarray) -> List[np.ndarray]:
        # x axies range: [-106.25361836466242, 69.35664279371939]
        # y axies range: [-152.51261639895282, 172.00273049054766]
        voronoi1 = VoronoiMap(coordinates)
        voronoi1.buildVoronoi()
        self.server_vor_areas: List[np.ndarray] = voronoi1.area_for_server

        return self.server_vor_areas

    # create the map according to the cloudping
    def createMap(self) -> np.ndarray:
        raise ValueError("createMap() method should not be used!! It is deprecated and should be removed.")
        # communication delay matrix from row to coloum: 50.92 (from 0 to 1)
        communication_delays = np.array(
            [
                [0, 51.37, 130.57, 201.35, 201.41, 152.90],
                [51.37, 0, 109.36, 157.12, 225.84, 110.73],
                [130.57, 109.36, 0, 199, 267.46, 139.4],
                [201.35, 157.12, 199, 0, 78.61, 79.4],
                [201.41, 225.84, 267.46, 78.61, 0, 148.13],
                [152.90, 110.73, 139.4, 79.4, 148.13, 0],
            ]
        )

        # Use MDS to reflect communication delay to 2-dimention map
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        self.coordinates: np.ndarray = mds.fit_transform(communication_delays)
        self.createClientsRegion(self.coordinates)
        # print("Voronoi areas of each server:")
        # for i, area in enumerate(self.server_vor_areas):
        #     print(f"Server {i} area: {area}")

        return self.coordinates
        # max_distance = euclidean(self.coordinates[4],self.coordinates[2])

    def get_global_label_count(self) -> np.ndarray:
        all_lcs = []
        for server in self.servers:
            lc, _client_ids = server.collect_client_label_count()
            # Take the sum of label counts across all clients in the server
            lc = np.sum(lc, axis=0)

            all_lcs.append(lc)
        server_lcs = np.sum(all_lcs, axis=0)
        return server_lcs

    def get_all_label_count(self) -> Dict[int, int]:
        label_count = {}
        for server in self.servers:
            label_count = sum_dicts(label_count, server.get_all_client_label_count())
            # label_count.update(server.get_all_client_label_count())
        return label_count

    def plot_location_map(self) -> None:
        plot_name = f'{self.config.get("name")}_location_map'
        assignments = {}
        for server in self.servers:
            for client in server.clients.values():
                assignments[server.server_id] = assignments.get(server.server_id, []) + [client.unique_id]
            # if server.clientsSet:
            #     assignments[server.server_id] = [client.client_id for client in server.clientsSet]

        client_locations = np.array([client.location for client in self.clients])
        server_locations = np.array([server.location for server in self.servers])
        max_width = np.max(client_locations[:, 0]) + 10
        min_width = np.min(client_locations[:, 0]) - 10
        max_height = np.max(client_locations[:, 1]) + 10
        min_height = np.min(client_locations[:, 1]) - 10
        # server =

        # print(f"Max width: {max_width}, Max height: {max_height}")
        # print(f"Server locations: {server_locations}")
        # print(f"Client locations: {client_locations}")
        # print(f"Assignments: {assignments}")

        plt.figure(figsize=(10, 8))
        if assignments is not None:
            for server_id, clients in assignments.items():
                if clients:
                    print(f"Server {server_id} has clients: {clients}")
                    client_coords = client_locations[clients]
                    # print(f"Client coordinates for server {server_id}: {client_coords}")
                    plt.scatter(
                        client_coords[:, 0],
                        client_coords[:, 1],
                        label=f"Server {server_id} Clients",
                    )
                    plt.text(
                        server_locations[server_id, 0],
                        server_locations[server_id, 1] + 2,
                        f"Server {server_id}",
                        fontsize=12,
                        ha="center",
                        va="center",
                        color="red",
                    )
        else:
            for i in range(len(server_locations)):
                plt.text(
                    float(server_locations[i, 0]),
                    float(server_locations[i, 1]),
                    f"Server {i}",
                    fontsize=12,
                    ha="center",
                    va="center",
                    color="red",
                )
        # plt.scatter(client_locations[:, 0], client_locations[:, 1], c="blue", label="Clients")
        plt.scatter(
            server_locations[:, 0],
            server_locations[:, 1],
            c="red",
            marker="s",
            s=100,
            label="Servers",
        )
        plt.xlim(min_width, max_width)
        plt.ylim(min_height, max_height)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Clients and Servers in 2D Space")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_name}.png")
        plt.show()
        print(f"Location map saved as {plot_name}.png")

    def run(self) -> None:
        print("RUNNING...")

        self.client_allocation_tracker = ClientAllocationTracker()

        # Calculate global label count
        self.global_label_count = self.get_all_label_count()
        print(f"Global label count: {self.global_label_count}")
        for server in self.servers:
            server.global_label_count = self.global_label_count

        #     # print(server.calculate_server_noniidness())
        #     tvd_value = server.calculate_server_noniidness(global_label_count)
        #     print(f"Server {server.server_id} iidness: {tvd_value}")
        if self.config.get("server_async"):
            # self.run_global_async()
            # raise NotImplementedError("Server async is not enabled at the moment.")
            self.run_global_async_token()
        elif not self.leader:
            self.run_global_sync()
        else:
            raise NotImplementedError("Hierarchical training is not enabled at the moment.")
            self.run_hier()

        if not os.path.exists(f"./results/{self.config.get('result_file')}"):
            os.makedirs(f"./results/{self.config.get('result_file')}")
        histories = [0] * self.num_clients
        for server_id in range(self.config.get("num_servers")):
            server = self.servers[server_id]
            for client, freq in server.history.items():
                histories[client] = freq
        with open(
            f"./results/{self.config.get('result_file')}/{self.config.get('name')}/history.pkl",
            "wb",
        ) as f:
            pickle.dump(histories, f)

    def run_hier(self) -> None:
        round = 0
        while round < int(self.config.get("num_rounds")):
            for server in self.servers:
                server.update_local_sync()
            # if round != 0 and round % self.period == 0: # it is synchronized
            #     print(f"server synchronizing during {t} - {t+self.config.get('fedavg_delay')}"
            round += 1
            if round % self.hier_period == 0:  # should send to the global server
                print("TIME FOR CLOUD AGGREGATION")
                slowest = -1
                for server in self.servers:
                    copied_model_state_dict = {
                        k: v.clone().detach().to(device=server.device)
                        for k, v in server.aggregator.model.state_dict().items()
                    }
                    self.leader.receive_server(copied_model_state_dict)  # type: ignore
                    # server.age = 0
                    finish_time = server.cur_time + 2 * float(self.config.get("leader_comm"))
                    slowest = max(slowest, finish_time)
                self.leader.deliver_model(leader=True)

                for server in self.servers:
                    server.cur_time = slowest + float(self.config.get("hier_delay"))

        # calculate the max cur_time of all the servers

        print(
            colors[-1],
            "===========================================================================",
        )
        for server in self.servers:
            print(
                colors[server.server_id % len(colors)],
                f"Server {server.server_id}: ended training at time {server.cur_time} with a history {list(server.history.values())}",
                colors[-1],
            )

        print(self.servers[0].cur_time)

    def run_global_sync(self) -> None:
        """
        Polling the server to update the global model;
        Periodically synchronize the global model;
        In asynchronous systems, num_rounds is the upper bound of the number of server updating;
        For local sync, the synchronize period refers to # rounds
        For local async, the synchronize period refers to time
        """

        # global_label_count = self.get_all_label_count()
        # print(f"Global label count: {global_label_count}")
        round_idx = 0
        max_rounds = self.config.get("num_rounds")
        import time

        # create_flag = True
        while round_idx < max_rounds:
            client_assignments = self.calculate_client_assignments_as_np()
            self.client_allocation_tracker.add_allocation(client_assignments, round_idx)

            # if round_idx > 4 and create_flag:
            #     print(f"Round {round_idx} of {max_rounds} started")

            #     print(f"Introducing new server at round {round_idx}")
            #     self.introduce_new_server()
            #     create_flag = False

            round_s = time.time()
            print(f"Round {round_idx} of {max_rounds} started")
            if self.config.get("client_async"):
                # ready2sync = True
                for server in self.servers:
                    # Check for events

                    while ev := self.event_system.next_event(server.cur_time, round_idx):
                        self.handleEvent(ev)

                    # print(f"Server {server.server_id} updating local model asynchronously")
                    if server.update_local_async_global_sync(current_round=round_idx, event_system=self.event_system):
                        # ready2sync = False # not all the servers are waiting
                        round_idx += 1
                    # print(f"Server {server.server_id} updated local model asynchronously at time {server.cur_time}")
                    # print(f"server synchronizing during {max_time} - {max_time + self.config.get('globalsync_delay')}")
                # print("Servers done")

            else:
                # Run local synchronous
                raise NotImplementedError("Local synchronous is not enabled at the moment.")
                t = self.servers[0].cur_time
                for server in self.servers:
                    server.update_local_sync()
                    t = max(t, server.cur_time)
                # if round != 0 and round % self.period == 0: # it is synchronized
                #     print(f"server synchronizing during {t} - {t+self.config.get('fedavg_delay')}")
                for server in self.servers:
                    server.cur_time = t
                round_idx += 1

            # Decide whether to move a client
            self.rebalance_clients()

            assigned_clients = self.calculate_client_assignements()
            assigned_clients["round"] = round_idx
            self.client_assignment_history.append(assigned_clients)

            # for server in self.servers:
            #     # print(server.calculate_server_noniidness())
            #     tvd_value = server.calculate_server_noniidness(self.global_label_count)
            #     print(f"Server {server.server_id} iidness: {tvd_value}")
            # print(f"Assigned clients in round {round_idx}:")
            # for server in self.servers:
            #     server.print_assigned_clients()
            round_e = time.time()

            round_duration = round_e - round_s

            print(f"Round {round_idx} ended, duration: {round_duration:.2f} seconds")

            if round_duration < 0.01:
                raise ValueError(
                    f"Round {round_idx} took less than 0.01 seconds, which is too fast. Check your configuration."
                )
        print(
            colors[-1],
            "===========================================================================",
        )
        for server in self.servers:
            print(
                colors[server.server_id % len(colors)],
                f"Server {server.server_id}: ended training at time {server.cur_time} with a history {list(server.history.values())}",
                colors[-1],
            )

        # print(self.servers[0].cur_time)

    def calculate_client_assignements(self) -> Dict[Union[int, str], Union[List[int], int]]:
        """
        Calculate the client assignments for each server.
        :return: A dictionary where keys are server IDs and values are lists of client IDs assigned to that server.
        """
        all_client_ids = [client.client_id for client in self.clients]
        assignments = {}
        for server in self.servers:
            server_key = f"server_{server.server_id}"
            assignments[server_key] = server.get_assigned_client_ids()
            # Remove all clients that are not assigned to any server
            all_client_ids = [client_id for client_id in all_client_ids if client_id in assignments[server_key]]

        assignments["global_pool"] = all_client_ids
        return assignments

    def calculate_client_assignments_as_np(self) -> np.ndarray:
        all_client_ids = [client.client_id for client in self.clients]
        assignments = np.ones(len(all_client_ids), dtype=int) * -1  # -1 means not assigned to any server
        for server in self.servers:
            assigned_client_ids = server.get_assigned_client_ids()
            for client_id in assigned_client_ids:
                if client_id in all_client_ids:
                    assignments[all_client_ids.index(client_id)] = server.server_id
        return assignments

    def move_client(self, client_id: int, source_server_id: int, target_server_id: int) -> None:
        """
        Move a client from source server to target server.
        :param client_id: The ID of the client to move.
        :param source_server_id: The ID of the server from which to move the client.
        :param target_server_id: The ID of the server to which to move the client.
        """
        source_server: Server = self.servers[source_server_id]
        target_server: Server = self.servers[target_server_id]

        if client_id not in source_server.clients:
            raise ValueError(f"Client {client_id} not found in source server {source_server_id}.")

        # Remove the client from the source server
        client, _next_time = source_server.remove_client(client_id)

        target_server.add_client(client)

    def calculate_latency_stats(
        self,
    ) -> Dict[int, Dict[str, Union[float, List[float]]]]:
        """Calculate and print latency statistics for each server."""
        print("Calculating latency statistics for each server...")

        stats = {}
        for server in self.servers:
            stats[server.server_id] = server.latency_stats()
        return stats

    def rebalance_by_latency(self) -> None:
        """For every server get all the latencies of the clients.
        Rebalance clients based on the latency of the clients.
        """
        print("Rebalancing clients by latency...")

        # Get the latencies of all clients in each server
        latency_stats = self.calculate_latency_stats()
        worst_client_id = -1
        worst_latency_diff = -1
        worst_latency = -1
        source_server_id = -1

        for server_id, stats in latency_stats.items():

            client_ids = list(self.servers[server_id].clients.keys())
            max_abs_diff_item_id = np.argmax(stats["abs_diff"])
            max_abs_diff_client_id = client_ids[max_abs_diff_item_id]
            max_abs_diff = np.max(stats["abs_diff"])
            if max_abs_diff > worst_latency_diff:
                if len(stats["latencies"]) < 2:
                    print(f"Server {server_id} has less than 2 clients, skipping...")
                    continue
                worst_latency_diff = max_abs_diff
                worst_client_id = max_abs_diff_client_id
                worst_latency = stats["abs_diff"][max_abs_diff_item_id]
                source_server_id = server_id

        # print(
        #     f"Client to move: Client {worst_client_id} with worst latency difference {worst_latency_diff:.2f} ms and latency {worst_latency:.2f} ms from Server {source_server_id}"
        # )

        _current_avg_std_latency_all = np.mean([stats["std_latency"] for stats in latency_stats.values()])

        best_target_server_id = None
        best_new_std_latency = float("inf")

        # Find a target server to move the client to such that the std deviation of the latencies is minimized
        for server_id, stats in latency_stats.items():
            if server_id == source_server_id:
                continue
            new_std_latency = np.std(stats["latencies"] + [worst_latency])
            # print(
            #     f"Server {server_id} - New Std Latency if moved client: {new_std_latency:.2f} ms, old Std Latency: {stats['std_latency']:.2f} ms"
            # )
            _new_avg_std_latency_all = np.mean(
                [stats["std_latency"] for stats in latency_stats.values() if stats["std_latency"] != 0]
                + [new_std_latency]
            )
            # print(
            #     f"New average std latency across all servers if moved client: {new_avg_std_latency_all:.2f} ms, current average std latency: {current_avg_std_latency_all:.2f} ms"
            # )
            if new_std_latency < best_new_std_latency:
                best_new_std_latency = new_std_latency
                best_target_server_id = server_id
        if best_target_server_id is None:
            print("No suitable target server found for rebalancing based on latency.")
            # return
        else:
            print(
                f"Best target server for rebalancing: Server {best_target_server_id} with new std latency {best_new_std_latency:.2f} ms"
            )
            # for server in self.servers:
            #     # Print clients in each server
            #     print(f"Server {server.server_id} clients: {list(server.clients.keys())}")
            # Move the client to the new target server
            source_server = self.servers[source_server_id]
            target_server = self.servers[best_target_server_id]
            client, next_time = source_server.remove_client(worst_client_id)
            client.move_ban = 5
            target_server.add_client(client, next_time=next_time)

    def rebalance_clients_random(self) -> None:
        print("Rebalancing clients randomly...")

        # Get random server

        server = random.choice(self.servers)

        # Make sure the server has at least two clients to rebalance
        while len(server.clients) < 2:
            print(f"Server {server.server_id} has less than 2 clients, selecting another server.")
            server = random.choice(self.servers)

        # Get random client from the server
        if not server.clients:
            print(f"Server {server.server_id} has no clients to rebalance.")
            return
        client_id = random.choice(list(server.clients.keys()))

        # Get random target server that is not the current server
        target_server = random.choice([s for s in self.servers if s.server_id != server.server_id])

        print(f"Rebalancing client {client_id} from server {server.server_id} to server {target_server.server_id}")

        # Move the client to the target server
        client, next_time = server.remove_client(client_id)
        client.move_ban = 5
        target_server.add_client(client, next_time=next_time)

    def update_move_bans(self) -> None:
        for server in self.servers:
            for client in server.clients.values():
                client.move_ban = client.move_ban - 1
                if client.move_ban < 0:
                    client.move_ban = 0

    def rebalance_non_iid_clients(self) -> None:
        """
        Rebalance clients based on the non-IIDness of the label distribution across servers.
        This method will find the server with the largest deviation from the global label count
        and move clients to balance the label distribution.
        """
        print("Rebalancing non-IID clients...")

        global_label_count = self.get_global_label_count()
        # print("Global label counts:", global_label_count)
        global_label_count = global_label_count / len(self.servers)

        candidates: Dict[Any] = {}
        for server in self.servers:
            total_weighted_tvd, new_total_weighted_tvd, client_to_remove = server.tvd_and_propose_client(
                global_label_count
            )
            tvd_differnce = total_weighted_tvd - new_total_weighted_tvd
            num_remaining_clients = len(server.clients) - 1

            if client_to_remove["client_id"] >= 0 and tvd_differnce >= 0 and num_remaining_clients > 2:
                candidates[server.server_id] = {
                    **client_to_remove,
                    "tvd_difference": tvd_differnce,
                    "total_weighted_tvd": total_weighted_tvd,
                    "new_total_weighted_tvd": new_total_weighted_tvd,
                    "source_server_id": server.server_id,
                }
                print(
                    f"Server {server.server_id} can remove client {client_to_remove['client_id']} to reduce TVD by {tvd_differnce:.4f}. "
                )
            else:
                print(
                    f"Server {server.server_id} has no clients to remove or removing client {client_to_remove['client_id']} would not reduce TVD."
                )

        if not candidates:
            print("No candidates found for rebalancing. All servers are balanced.")
            return

        # Find the server with the maximum TVD difference
        max_tvd_difference = -1
        source_server_id: int = None
        best_candidate = None
        for server_id, candidate in candidates.items():
            if candidate["tvd_difference"] > max_tvd_difference:
                max_tvd_difference = candidate["tvd_difference"]
                source_server_id = server_id
                best_candidate = candidate
        if source_server_id is None:
            print("No target server found for rebalancing based on non-IIDness.")
            return
        print(
            f"Source server for rebalancing: Server {source_server_id} with max TVD difference {max_tvd_difference:.4f}"
        )
        # Find a ta5rget server to move the client to such that the TVD is minimized
        target_server_id = None
        best_new_total_weighted_tvd = float("inf")

        for server in self.servers:
            server_id = server.server_id
            if server_id == source_server_id:
                continue
            print(f"Evaluating server {server_id} for rebalancing...")
            client_id = best_candidate["client_id"]
            client_latency = self.frequency_matrix[client_id][server_id]
            client_distribution = best_candidate["client_distribution"]

            new_total_weighted_tvd = server.estimate_augmented_tvd(
                global_label_count,
                client_id,
                client_distribution,
                client_latency,
                action="add",
            )

            if new_total_weighted_tvd < best_new_total_weighted_tvd:
                best_new_total_weighted_tvd = new_total_weighted_tvd
                target_server_id = server_id
            else:
                print(
                    f"Server {server_id} - New Total Weighted TVD {new_total_weighted_tvd:.4f} is not better than current best {best_new_total_weighted_tvd:.4f}"
                )
        if target_server_id is None:
            print("No suitable target server found for rebalancing based on non-IIDness.")
            return
        print(
            f"[Move Decision] Best target server for rebalancing: Server {target_server_id} with new total weighted TVD {best_new_total_weighted_tvd:.4f}"
        )
        target_server: Server = self.servers[target_server_id]
        source_server: Server = self.servers[source_server_id]
        # Move the client to the target server
        client, next_time = source_server.remove_client(client_id)
        client.move_ban = 5
        target_server.add_client(client, next_time=next_time)

    def rebalance_clients(self) -> None:

        rebalance_policy = self.config.get("rebalance_policy")
        # @TODO: Implement a more sophisticated method to capture the current time
        curr_time = self.servers[0].cur_time
        # global_label_count = self.get_global_label_count()
        # print("Global label counts:", global_label_count)
        # global_label_count = global_label_count / len(self.servers)
        # for server in self.servers:
        #     _total_weighted_tvd, _client_distributions, _client_weights, _client_ids = server.compute_server_tvd(
        #         global_label_count
        #     )
        rebalance = False
        if rebalance_policy == "random":
            # Only rebalance randomly 1% of the time
            if random.random() < 0.10:
                self.rebalance_clients_random()
                rebalance = True
        elif rebalance_policy == "latency":
            if self.cur_round % 5 == 0:
                global_label_count = self.get_global_label_count()
                print("Global label counts:", global_label_count)
                global_label_count = global_label_count / len(self.servers)
                for server in self.servers:
                    _total_weighted_tvd, _client_distributions, _client_weights, _client_ids = (
                        server.compute_server_tvd(global_label_count)
                    )
                # Rebalance clients based on latency statistics
                self.rebalance_by_latency()
                rebalance = True
        elif rebalance_policy == "non_iid":
            if self.cur_round % 1 == 0:
                # Rebalance clients based on non-IIDness of the label distribution
                self.rebalance_non_iid_clients()
                rebalance = True
        else:
            # No rebalancing policy specified, or an unsupported policy
            # print(f"No rebalancing policy specified or unsupported policy: {rebalance_policy}. Skipping rebalancing.")
            pass

        if rebalance:
            self.update_move_bans()
            self.event_history.add_event(
                MEvent(
                    curr_time,
                    "system",
                    "rebalance_clients",
                    round=self.cur_round,
                    description=f"policy={rebalance_policy}",
                )
            )

        return

    def move_back(self) -> None:
        for target_server_id in list(self.selected_clients.keys()):
            # target_server_id = list(self.selected_clients.keys())[0]
            client_id_in_server = self.selected_clients[target_server_id][0]
            previous_server_id = self.selected_clients[target_server_id][1]
            previous_server = self.servers[previous_server_id]
            target_server = self.servers[target_server_id]
            client: Client = target_server.clients.pop(target_server.clients_num - 1)
            target_server.clients_num = target_server.clients_num - 1
            client.server.pop(0)
            client.server.append(previous_server_id)
            client.client_id = client_id_in_server
            new_server_clients = {}
            for index, client in enumerate(previous_server.clients.values()):
                if index < client_id_in_server:
                    client.client_id = index
                    new_server_clients[index] = client
                else:
                    client.client_id = index + 1
                    new_server_clients[index + 1] = client
            new_server_clients[client_id_in_server] = client
            previous_server.history[client_id_in_server] = target_server.history[target_server.clients_num]
            previous_server.clients_num += 1
            previous_server.client_train_datasets.extend(
                target_server.client_train_datasets.pop(target_server.clients_num)
            )
            del target_server.history[target_server.clients_num]
            previous_server.clients.clear()
            previous_server.clients.update(new_server_clients)

        for server in self.servers:
            self.clients_speed_per_server[server.server_id] = server.set_clients_speed()

    def share_after_move(self) -> None:
        # The chosen client first move to new server
        # After some determined rounds, it will communicate with previous server
        # The condition is first moving then sharing
        for target_server_id in list(self.selected_clients.keys()):
            # target_server_id = list(self.selected_clients.keys())[0]
            _client_id_in_server = self.selected_clients[target_server_id][0]
            previous_server_id = self.selected_clients[target_server_id][1]
            previous_server = self.servers[previous_server_id]
            target_server = self.servers[target_server_id]
            client = target_server.clients[target_server.clients_num - 1]
            client.server.append(previous_server_id)
            # client.server.append(target_server_id)
            client.client_id = target_server.clients_num - 1
            previous_server.clients[client.client_id] = client
            previous_server.history[client.client_id] = target_server.history[client.client_id]
            previous_server.clients_num += 1
            previous_server.client_train_datasets.extend([target_server.client_train_datasets[client.client_id]])

        for server in self.servers:
            self.clients_speed_per_server[server.server_id] = server.set_clients_speed()

    def run_global_async_token(self) -> None:
        round = 0
        move_flag = False
        _alternate_new_flag = False
        update_model = False
        # update the initial client's model
        # if self.config.get("move_a_client"):
        #     # self.update_client_model()
        #     if self.config.get("avg_model"):
        #         self.alternate_init()
        #     else:
        #         self.move_client()
        global_label_count = self.get_global_label_count()
        print("Global label counts:", global_label_count)
        global_label_count = global_label_count / len(self.servers)
        for server in self.servers:
            _total_weighted_tvd, _client_distributions, _client_weights, _client_ids = server.compute_server_tvd(
                global_label_count
            )
        while round < int(self.config.get("num_rounds")):
            client_assignments = self.calculate_client_assignments_as_np()
            self.client_allocation_tracker.add_allocation(client_assignments, round)
            # Print the current round if % 50
            if round % 20 == 0:
                print(
                    f"Round {round} of {self.config.get('num_rounds')} started. Percentage: {round / int(self.config.get('num_rounds')) * 100:.2f}%"
                )
            if self.config.get("move_a_client") and not self.config.get("alternate_new"):
                if not move_flag:
                    print("----------------------------START MOVING-----------------------------")
                    self.move_client()
                    move_flag = True
            elif self.config.get("avg_model"):
                if not update_model:
                    print("----------------------------UPDATE——CLIENT——MODEL-----------------------------")
                    self.update_client_model()
                    update_model = True
            # elif self.config.get("move_a_client") and self.config.get("alternate_new"):
            #     if round < int(self.config.get("num_rounds")) / 4:
            #         if not move_flag:
            #             print("----------------------------START MOVING-----------------------------")
            #             self.move_client()
            #             move_flag = True
            #     else:
            #         if not alternate_new_flag:
            #             print("----------------------------START SHARING-----------------------------")
            #             self.alternate_init()
            #             # self.share_after_move()
            #             alternate_new_flag = True
            # # elif (not self.config.get("move_a_client")) and self.config.get("alternate_new"):
            # #     if not alternate_new_flag:
            # #         print("----------------------------START SHARING-----------------------------")
            # #         self.alternate_init()
            # #         alternate_new_flag = True
            # # elif self.config.get("alternate_late"):
            # #     if round > int(self.config.get("num_rounds")) / 4:
            # #         if not alternate_new_flag:
            # #             print("----------------------------START SHARING-----------------------------")
            # #             self.alternate_init()
            # #             alternate_new_flag = True
            # # elif self.config.get("move_late"):
            # #     if round > int(self.config.get("num_rounds")) / 4:
            # #         if not move_flag:
            # #             print("----------------------------START MOVING-----------------------------")
            # #             self.move_client()
            # #             move_flag = True
            # every time find the next server to update
            latest_server = self.servers[0]
            # print("heap_last_server", latest_server.heap)
            # print("heap_last_server", self.servers[1].heap)
            for server in self.servers:
                if server.heap[0][0] < latest_server.heap[0][0]:
                    latest_server = server
            latest_server.update_local_async_global_async()
            round += 1
            self.cur_round = round

            # Decide whether to move a client
            self.rebalance_clients()

            assigned_clients = self.calculate_client_assignements()
            assigned_clients["round"] = round
            self.client_assignment_history.append(assigned_clients)
            for server in self.servers:
                self.rebalance_data.append(
                    (server.cur_time, round, server.server_id, server.current_total_weighted_tvd)
                )

        for server in self.servers:
            print(
                colors[server.server_id % len(colors)],
                f"Server {server.server_id}: ended training at time {server.cur_time} with a history {list(server.history.values())}",
                colors[-1],
            )

    # take average models of the servers that client should communicate with
    def average_models(self) -> Dict[int, List[int]]:
        clients_averaged_model = {}
        for target_server_id in list(self.selected_clients.keys()):
            target_server_id = int(target_server_id)
            move_from_server_id = self.selected_clients[str(target_server_id)][1]
            client_id_in_server = self.selected_clients[str(target_server_id)][0]
            model1 = self.servers[target_server_id].aggregator.model.state_dict()
            model2 = self.servers[move_from_server_id].aggregator.model.state_dict()
            averaged_model = {}
            for key in model1.keys():
                averaged_model[key] = (model1[key] + model2[key]) / 2.0
            clients_averaged_model[client_id_in_server] = [
                target_server_id,
                move_from_server_id,
                averaged_model,
            ]
        return clients_averaged_model

    def alternate_init(self) -> None:
        # The chosen client should train one server's model in one round
        # and train another server's model in next round
        for target_server_id in list(self.selected_clients.keys()):
            client_id_in_server = self.selected_clients[target_server_id][0]
            previous_server_id = self.selected_clients[target_server_id][1]
            previous_server = self.servers[previous_server_id]
            previous_server.share_client_id = client_id_in_server
            target_server = self.servers[int(target_server_id)]
            print(
                "Client {} of Server{} will communicate also with Server {}".format(
                    client_id_in_server, previous_server, target_server_id
                )
            )
            client = previous_server.clients[client_id_in_server]
            client.server.append(int(target_server_id))
            target_server.clients[target_server.clients_num] = client
            target_server.history[target_server.clients_num] = previous_server.history[client_id_in_server]
            if self.config.get("move_a_client") or self.config.get("alternate_late"):
                del self.clients_speed_per_server[previous_server.server_id]
                del self.clients_speed_per_server[target_server.server_id]
            # print("previous_server heap 0",previous_server.heap)

            self.clients_speed_per_server[previous_server.server_id] = previous_server.set_clients_speed()
            # print("previous_server heap 1",previous_server.heap)
            client.new_id = target_server.clients_num
            # print('target_server heap0',target_server.heap)
            target_server.push_heap_item[target_server.clients_num] = previous_server.share_heap_item[
                client_id_in_server
            ]
            target_server.clients_num += 1
            self.clients_speed_per_server[target_server.server_id] = target_server.set_clients_speed()
            # print('target_server heap1',target_server.heap)
            self.config.write_config(target_server.updates)
            self.config.write_config(previous_server.updates)
        # for i in range(len(self.servers)):
        #     int_keys = [int(key) for key in self.selected_clients.keys()]
        #     for keys in int_keys:
        #         if i != keys and i!= self.selected_clients[str(keys)][1] :
        #             server = self.servers[i]
        #             server.set_clients_speed()
        #             self.config.write_config(server.updates)

    def _unused_move_client(self) -> None:
        for target_server_id in list(self.selected_clients.keys()):
            # target_server_id = list(self.selected_clients.keys())[0]
            target_server_id = int(target_server_id)
            client_id_in_server = self.selected_clients[str(target_server_id)][0]
            previous_server_id = self.selected_clients[str(target_server_id)][1]
            previous_server: Server = self.servers[previous_server_id]
            target_server: Server = self.servers[target_server_id]
            target_server.recieved_moved = True
            print(
                "Moving client {} of Server{} to Server {}".format(
                    client_id_in_server, previous_server, target_server_id
                )
            )
            previous_server.pop_client_id = client_id_in_server
            client = previous_server.clients.pop(client_id_in_server)
            client.server.pop(0)
            client.server.append(target_server_id)
            client.client_id = target_server.clients_num
            target_server.clients[target_server.clients_num] = client
            target_server.history[target_server.clients_num] = 0
            target_server.clients_num += 1
            target_server.client_train_datasets.extend([previous_server.client_train_datasets.pop(client_id_in_server)])
            del previous_server.history[previous_server.clients_num - 1]
            new_server_clients = {}
            for new_index, c in enumerate(previous_server.clients.values()):
                c.client_id = new_index  # update each client' id
                new_server_clients[new_index] = c
            previous_server.clients.clear()
            previous_server.clients.update(new_server_clients)
            previous_server.clients_num -= 1
            self.clients_speed_per_server[previous_server.server_id] = previous_server.set_clients_speed()
            if self.config.get("move_late"):
                target_server.push_heap_item[target_server.clients_num] = previous_server.popped_heap_item[
                    client_id_in_server
                ]
            self.clients_speed_per_server[target_server.server_id] = target_server.set_clients_speed()
            self.config.write_config(target_server.updates)
            self.config.write_config(previous_server.updates)
            if self.config.get("alternate_new"):
                previous_selected_clients = self.selected_clients.copy()
                del self.selected_clients[str(target_server_id)]
                print("self.selected_clients", self.selected_clients)
                self.selected_clients[previous_server_id] = [
                    client.client_id,
                    target_server_id,
                ]
                print("later selected_clients", self.selected_clients)
                updates = {}
                updates["selected_clients"] = previous_selected_clients
                self.config.write_config(updates)

    def evaluate_server_level_iidness(self, server: Server) -> float:
        # the greater return value the bigger difference
        subsets = server.client_train_datasets
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
                label_encoder = LabelEncoder()
                flattened_data_int = label_encoder.fit_transform(flattened_data)
                subset_distribution = np.bincount(flattened_data_int, minlength=len(global_distribution))
                kl_divergence = entropy(subset_distribution, qk=global_distribution)
                kl_divergences.append(kl_divergence)
            return np.mean(kl_divergences)

        iidness = calculate_kl_divergence(subsets)
        return iidness

    def update_client_model(self) -> None:
        print("update_client_model")
        clients_averaged_models = self.average_models()
        print(clients_averaged_models)
        for client_id_in_server in clients_averaged_models.keys():
            target_server_id = int(clients_averaged_models[client_id_in_server][0])
            averaged_model = clients_averaged_models[client_id_in_server][2]
            previous_server = self.servers[clients_averaged_models[client_id_in_server][1]]
            print("client_id_in_server", client_id_in_server)
            previous_server.share_client_id = client_id_in_server
            client = previous_server.clients[client_id_in_server]
            client.server.append(target_server_id)
            # copied_model_state_dict = {k: v.clone().detach().to(device=self.device) for k, v in averaged_model.state_dict().items()}
            client.model.load_state_dict(averaged_model)
            # print("before target_server client",self.servers[target_server_id].clients)
            target_server = self.servers[target_server_id]
            target_server.clients[target_server.clients_num] = client
            target_server.history[target_server.clients_num] = 0
            self.clients_speed_per_server[previous_server.server_id] = previous_server.set_clients_speed()
            # print("previous_server heap 1",previous_server.heap)
            client.new_id = target_server.clients_num
            # print('target_server heap0',target_server.heap)
            target_server.push_heap_item[target_server.clients_num] = previous_server.share_heap_item[
                client_id_in_server
            ]
            target_server.clients_num += 1
            self.clients_speed_per_server[target_server.server_id] = target_server.set_clients_speed()
            # print('target_server heap1',target_server.heap)
            self.config.write_config(target_server.updates)
            self.config.write_config(previous_server.updates)
        for target_server_id in list(self.selected_clients.keys()):
            client_id_in_server = self.selected_clients[target_server_id][0]
            previous_server_id = self.selected_clients[target_server_id][1]
            previous_server = self.servers[previous_server_id]
            previous_server.share_client_id = client_id_in_server
            target_server = self.servers[int(target_server_id)]
            print(
                "Client {} of Server{} will communicate also with Server {}".format(
                    client_id_in_server, previous_server, target_server_id
                )
            )
            client = previous_server.clients[client_id_in_server]
            client.server.append(int(target_server_id))
            target_server.clients[target_server.clients_num] = client
            target_server.history[target_server.clients_num] = previous_server.history[client_id_in_server]
            if self.config.get("move_a_client") or self.config.get("alternate_late"):
                del self.clients_speed_per_server[previous_server.server_id]
                del self.clients_speed_per_server[target_server.server_id]

    def save_client_stats(self, save_plot: bool = False) -> None:
        """
        Save client statistics to a CSV file.
        This method collects the statistics of each client and saves them to a CSV file in the base result directory.
        """
        print("Saving client statistics to disk...")
        client_stats = []
        for server in self.servers:
            for client in server.clients.values():

                for idx, (t, loss_values, lr) in enumerate(client.training_history):
                    client_stats.append(
                        {
                            "client_id": client.client_id,
                            "time": t,
                            "loss": loss_values,
                            "learning_rate": lr,
                        }
                    )

        df = pd.DataFrame(client_stats)
        # Make column client_id categorical
        # df["client_id"] = df["client_id"].astype("category")
        client_stats_path = self.base_result_dir / "client_stats.csv"
        df.to_csv(client_stats_path, index=False)

        if save_plot:
            import matplotlib.pyplot as plt

            # Make column client_id categorical
            # df["client_id"] = df["client_id"].astype("category")

            plt.figure(figsize=(14, 6))

            # First subplot: Loss
            plt.subplot(2, 1, 1)
            sns.lineplot(df, x="time", y="loss", hue="client_id", marker="o")
            plt.xlabel("Time")
            plt.ylabel("Loss")
            plt.title("Client Loss History")
            plt.legend()
            plt.grid()

            # Second subplot: Learning Rate
            plt.subplot(2, 1, 2)
            sns.lineplot(df, x="time", y="learning_rate", hue="client_id", marker="o")
            plt.xlabel("Time")
            plt.ylabel("Learning Rate")
            plt.title("Client Learning Rate History")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plot_path = self.base_result_dir / "client_loss_history.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Client loss history plot saved to {plot_path}")

    def plot_rebalance_history(self, df: pd.DataFrame) -> None:
        """
        Plot the rebalance history from the DataFrame.
        This method creates a plot showing the total variation distance (TVD) over time for each server.
        """

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="time", y="tvd", hue="server_id", marker="o")
        plt.title("Rebalance History - Total Variation Distance (TVD)")
        plt.xlabel("Time")
        plt.ylabel("Total Variation Distance (TVD)")
        plt.legend(title="Server ID")
        plt.grid()
        plt.tight_layout()
        plot_path = self.base_result_dir / "rebalance_history.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Rebalance history plot saved to {plot_path}")

    def report(self) -> None:

        msg = "* Event History *"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))
        for ev in self.event_history.get_events():
            print(f"[{ev.timestamp:.2f} {ev.entity}] {ev.event_type}")

        print("*" * len(msg))

        print(self.event_history.to_dataframe())

        self.base_result_dir.mkdir(parents=True, exist_ok=True)
        event_history_path = self.base_result_dir / "event_history.csv"
        self.event_history.save_to_disk(event_history_path)

        rebalance_path = self.base_result_dir / "rebalance_history.csv"
        rebalance_df = pd.DataFrame(self.rebalance_data, columns=["time", "round", "server_id", "tvd"])
        rebalance_df.to_csv(rebalance_path, index=False)
        self.plot_rebalance_history(rebalance_df)

        self.save_client_stats(True)

        # Save the client assignments
        print("Saving client assignments to disk...")
        cat_df = self.client_allocation_tracker.get_as_dataframe()
        client_assignment_path = self.base_result_dir / "client_assignment_history.csv"
        cat_df.to_csv(client_assignment_path, index=False)
        client_assignment_history_path = self.base_result_dir / "client_assignment_history.pkl"
        # Save the client assignment history
        with open(
            client_assignment_history_path,
            "wb",
        ) as f:
            pickle.dump(self.client_assignment_history, f)

        # Config files
        conf_dir = self.config.path_as_path().parent
        config_target_path = self.base_result_dir / "config"
        config_target_path.mkdir(parents=True, exist_ok=True)
        print(f"Copying config files from {conf_dir} to {config_target_path}")
        # Copy the conf_dir to results using shutil
        shutil.copytree(
            conf_dir,
            config_target_path,
            dirs_exist_ok=True,
        )

        self.logger.final_logs()
        self.logger.save()  # save pkl
        print(line("REPORT"))
        print("Server\t|Fast\t|Medium\t|Slow\t|Model Age\t|Staleness")
        for server in self.servers:
            server.report()

        ################## SAVING TXT FILE ############################
        ending_time = self.logger.final_report["time"]
        avg_acc = self.logger.final_report["acc"]["global"]  # type: ignore
        std_acc = self.logger.final_report["acc"]["std"]  # type: ignore
        alternate_new_age = 0.0
        avg_staleness = 0.0
        for server in self.servers:
            alternate_new_age += server.age
            avg_staleness += server.sum_staleness // max(1, server.age)
        alternate_new_age /= float(len(self.servers))
        avg_staleness /= float(len(self.servers))

        # Save the results in the file with a data and time
        txt_path = self.base_result_dir.parent / f"{self.base_result_dir.name}.txt"
        with open(
            txt_path,
            "a",
        ) as f:
            f.write(line("Configuration") + "\n")
            # the NAME
            name_content = "NAME: "
            if self.config.get("client_async"):
                name_content += "Local Async"
            else:
                name_content += "Local Sync"
            if self.config.get("server_async"):
                name_content += "Global Async"
            else:
                name_content += "Global Sync"
            name_content += f"{self.config.get('num_servers')} x {self.num_clients_per_server} clients for each server"
            f.write(content(name_content) + "\n")
            if self.config.get("move_a_client"):
                f.write(
                    content(
                        f"Choose Server {list(self.selected_clients.keys())}'s Client {list(self.selected_clients.values())[0][0]} to communicate with Server {list(self.selected_clients.values())[0][1]}"
                    )
                )
            # the DATASET
            f.write(content(f"DATASET: {self.config.get('dataset')}") + "\n")
            if not self.config.get("iid"):
                f.write(content("DATA DISTRIBUTION: Non IID") + "\n")
            else:
                f.write(
                    content(
                        f"DATA DISTRIBUTION: NON-IID, each client has at most {self.config.get('num_label_per_client')} labels"
                    )
                    + "\n"
                )

            f.write(
                content(
                    f"{self.config.get('num_local_epochs')} local epochs, {self.config.get('batch_size')} batch size"
                )
                + "\n"
            )
            f.write(content(f"# of local updates: {self.config.get('num_rounds')}") + "\n")
            f.write(content(f"Weight_decay : {self.config.get('weight_decay')}") + "\n")
            f.write(
                content(
                    f"Local learning rate: {self.config.get('local_lr')}, global learning rate: {self.config.get('global_lr')}"
                )
                + "\n"
            )
            f.write(content(f"L2 norm mu: {self.config.get('l2_mu')}") + "\n")
            f.write(content(f"l2_norm : {self.config.get('l2_mu')} ") + "\n")

            f.write(content(f"gaussianclient : {self.config.get('gaussianclient')} ") + "\n")
            if not self.config.get("gaussianclient"):
                f.write(
                    content(
                        f"slow: medium: fast = {self.config.get('slow_ratio')} : {self.config.get('medium_ratio')} : {self.config.get('fast_ratio')}"
                    )
                    + "\n"
                )
            f.write(f"server_heter: {self.config.get('server_heter')} \n")

            if self.config.get("server_async"):
                f.write(content(f"Peer learning rate: {self.config.get('global_lr_peer')}") + "\n")
                f.write(content(f"Peer activation rate: {self.config.get('activation_rate')}") + "\n")
                f.write(content(f"# of token-based broadcasts: {self.servers[0].token_sequence}") + "\n")
                f.write(content(f"token threshold: {self.config.get('token_threshold')}") + "\n")
            elif self.config.get("client_async"):
                f.write(content(f"global_sync_period : {self.period} ") + "\n")
                sync_rounds = self.servers[0].sync_round
                f.write(content(f"# of global sync rounds: {sync_rounds}") + "\n")

            f.write(line("Delay Simulation") + "\n")
            # f.write(content(f"training delay : {self.config.get('training_delay')} ") + "\n")
            for i in range(len(self.servers)):
                f.write(
                    content(f"training delay for server {i} : {self.config.get('server_'+str(i)+'_training_delay')} ")
                    + "\n"
                )
            f.write(content(f"globalsync delay : {self.config.get('globalsync_delay')} ") + "\n")
            f.write(content(f"fedavg delay : {self.config.get('fedavg_delay')} ") + "\n")
            f.write(content(f"fedAsync delay : {self.config.get('fedasync_delay')} ") + "\n")
            f.write(content(f"client communication delay: {self.config.get('comm_delay')}") + "\n")

            for server_index, client_speeds in self.clients_speed_per_server.items():
                for client_index, speed in enumerate(client_speeds):
                    f.write(
                        content(f"Server {server_index}: client {client_index}'s training time (speed) is {speed}\n")
                    )

            f.write(line("Result") + "\n")
            f.write(content(f"Global Accuracy: {100. * avg_acc:.2f}%") + "\n")
            for server_id, acc in self.logger.final_report["acc"].items():  # type: ignore
                if isinstance(server_id, int):
                    f.write(content(f" -- Server {server_id} Accuracy: {100. * acc:.2f}%") + "\n")
            f.write(content(f"Accuracy Std-dev: {std_acc:.6f}") + "\n")
            f.write(content(f"Running Time: {ending_time}") + "\n")
            f.write(content(f"Average Staleness: {avg_staleness}") + "\n")
            f.write(content(f"Server ages: {[server.age for server in self.servers]}") + "\n")
            f.write(content(f"Average queue length: {self.logger.final_report['q_len']}") + "\n")

        ############# DELAYS #################
        self.servers[0].aggregator.report()
        self.clients[0].report()

        # Plot each server's learning weight
        # for server in self.servers:
