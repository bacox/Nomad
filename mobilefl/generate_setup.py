import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from mobilefl.utils import calculate_delay_matrix, calculate_distances


def generate_clients(
    num_clients: int,
    max_width: int,
    max_height: int,
    delay_mean: float = 30,
    delay_std: float = 4,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    client_locations = np.random.rand(num_clients, 2) * [max_width, max_height]
    client_delays = np.random.normal(loc=delay_mean, scale=delay_std, size=num_clients)
    return client_locations, client_delays


def generate_servers(num_servers: int, max_width: int, max_height: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    server_locations = np.random.rand(num_servers, 2) * [max_width, max_height]
    return server_locations


def generate_alternative_server_locations(
    num_servers: int, max_width: int, max_height: int, seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    alternative_server_locations = np.random.rand(num_servers, 2) * [max_width, max_height]
    return alternative_server_locations


def assign_random(
    num_clients: int,
    num_servers: int,
    client_locations: np.ndarray,
    server_locations: np.ndarray,
    num_start_clients: int,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(num_clients)
    np.random.shuffle(indices)
    assignments: Dict[int, List[int]] = {i: [] for i in range(num_servers)}
    assignments[-1] = []  # For the unassigned clients
    for i, client_idx in enumerate(indices):
        server_id = i % num_servers
        if i < num_start_clients:
            assignments[server_id].append(int(client_idx))
        else:
            assignments[-1].append(int(client_idx))
    return assignments


def assign_distance_best(
    num_clients: int,
    num_servers: int,
    client_locations: np.ndarray,
    server_locations: np.ndarray,
    num_start_clients: int,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    # Assign clients to servers based on the closest server
    # Make sure servers get the same number of clients
    if seed is not None:
        np.random.seed(seed)
    max_clients_per_server = num_clients // num_servers
    assignments: Dict[int, List[int]] = {i: [] for i in range(num_servers)}
    # Initialize assignments for unassigned clients
    assignments[-1] = []  # For the unassigned clients
    distances = calculate_distances(client_locations, server_locations)

    for client_id in range(num_clients):
        if client_id >= num_start_clients:
            # If the client is not in the starting set, assign to unassigned clients
            assignments[-1].append(client_id)
            continue
        # Find the closest server
        closest_server_id = int(np.argmin(distances[client_id]))
        # Check if the server can take more clients
        if len(assignments[closest_server_id]) < max_clients_per_server:
            assignments[closest_server_id].append(client_id)
        else:
            # If the closest server is full, assign to the next available server
            for server_id in range(num_servers):
                if len(assignments[(closest_server_id + server_id) % num_servers]) < max_clients_per_server:
                    assignments[(closest_server_id + server_id) % num_servers].append(client_id)
                    break
    return assignments


def assign_distance_worst(
    num_clients: int,
    num_servers: int,
    client_locations: np.ndarray,
    server_locations: np.ndarray,
    num_start_clients: int,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    # Assign clients to servers based on the farthest server
    # Make sure servers get the same number of clients
    if seed is not None:
        np.random.seed(seed)
    max_clients_per_server = num_clients // num_servers
    assignments: Dict[int, List[int]] = {i: [] for i in range(num_servers)}
    # Initialize assignments for unassigned clients
    assignments[-1] = []  # For the unassigned clients
    distances = calculate_distances(client_locations, server_locations)
    for client_id in range(num_clients):
        if client_id >= num_start_clients:
            # If the client is not in the starting set, assign to unassigned clients
            assignments[-1].append(client_id)
            continue
        # Find the farthest server
        farthest_server_id = int(np.argmax(distances[client_id]))
        # Check if the server can take more clients
        if len(assignments[farthest_server_id]) < max_clients_per_server:
            assignments[farthest_server_id].append(client_id)
        else:
            # If the farthest server is full, assign to the next available server
            for server_id in range(num_servers):
                if len(assignments[(farthest_server_id + server_id) % num_servers]) < max_clients_per_server:
                    assignments[(farthest_server_id + server_id) % num_servers].append(client_id)
                    break
    return assignments


def assign_clients_to_servers(
    num_clients: int,
    num_servers: int,
    client_locations: np.ndarray,
    server_locations: np.ndarray,
    seed: Optional[int] = None,
    generator_type: str = "random",
    num_start_clients: Optional[int] = None,
) -> Dict[int, List[int]]:

    if num_start_clients is None:
        num_start_clients = num_clients

    if generator_type == "random":
        return assign_random(num_clients, num_servers, client_locations, server_locations, num_start_clients, seed=seed)
    elif generator_type == "distance_best":
        return assign_distance_best(
            num_clients, num_servers, client_locations, server_locations, num_start_clients, seed=seed
        )
    elif generator_type == "distance_worst":
        return assign_distance_worst(
            num_clients, num_servers, client_locations, server_locations, num_start_clients, seed=seed
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def plot_grid(
    client_locations: np.ndarray,
    server_locations: np.ndarray,
    max_width: float,
    max_height: float,
    plot_name: str,
    assignments: Optional[Dict[int, List[int]]] = None,
    file_path: Path = Path("."),
) -> None:
    plt.figure(figsize=(10, 8))

    if assignments is not None:
        if -1 in assignments:
            # Add [0,0] to server_locations for unassigned clients
            server_locations = np.vstack([server_locations, [0, 0]])
    # Determine colors based on the number of servers
    if len(server_locations) > 10:
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(server_locations)))
    else:
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(server_locations)))

    if assignments is not None:
        # if -1 in assignments:
        #     # Add [0,0] to server_locations for unassigned clients
        #     server_locations = np.vstack([server_locations, [0, 0]])
        # Do different style if -1 in assignments

        for server_id, clients in assignments.items():
            if clients:
                client_coords = client_locations[clients]
                alpha_value = 0.25 if server_id == -1 else 1.0
                plt.scatter(
                    client_coords[:, 0],
                    client_coords[:, 1],
                    label=f"Server {server_id} Clients",
                    c=colors[server_id % len(colors)],
                    alpha=alpha_value,
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
        plt.text(
            float(server_locations[:, 0]),
            float(server_locations[:, 1]),
            "-".join([f"Server {i}" for i in range(len(server_locations))]),
            fontsize=12,
            ha="center",
            va="center",
            color="red",
        )
    # plt.scatter(client_locations[:, 0], client_locations[:, 1], c="blue", label="Clients")
    # Make sure to use a consistent color for clients and servers
    # print(f"Server locations: {server_locations}")
    server_ids = np.arange(len(server_locations))
    plt.scatter(
        server_locations[:, 0],
        server_locations[:, 1],
        marker="s",
        s=100,
        label="Servers",
        c=colors[server_ids % len(colors)],
        edgecolor="black",
    )
    plt.xlim(0, max_width)
    plt.ylim(0, max_height)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Clients and Servers in 2D Space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path / f"{plot_name}.png")
    plt.show()
    print(f"Plot saved as {file_path / f'{plot_name}.png'}")


def save_config(
    filename: Union[str, Path],
    client_locations: np.ndarray,
    client_delays: np.ndarray,
    server_locations: np.ndarray,
    assignments: Dict[int, List[int]],
    client_server_latencies: np.ndarray,
    alternative_server_locations: Optional[np.ndarray] = None,
) -> None:

    # Invert assignments to match the expected format key: client_id, value: value of server_id
    client_assignments = [-1] * len(client_locations)
    for server_id, clients in assignments.items():
        for client_id in clients:
            client_assignments[client_id] = server_id
    # print(f"{client_assignments[0]=}")
    data = {
        "clients": [
            {
                "id": i,
                "x": float(loc[0]),
                "y": float(loc[1]),
                "delay": float(client_delays[i]),
                "assigned_server": client_assignments[i],
                "latencies": client_server_latencies[i].tolist(),
            }
            for i, loc in enumerate(client_locations)
        ],
        "servers": [
            {
                "id": i,
                "x": float(loc[0]),
                "y": float(loc[1]),
                "assigned_clients": assignments[i],
            }
            for i, loc in enumerate(server_locations)
        ],
    }
    if alternative_server_locations is not None:
        data["alternative_servers"] = [
            {
                "x": float(loc[0]),
                "y": float(loc[1]),
            }
            for i, loc in enumerate(alternative_server_locations)
        ]

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Configuration saved to {filename}")


def load_config(filename: Union[str, Path]) -> Dict[str, any]:  # type: ignore
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Configuration file {filename} not found.")
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def parse_num_entities(config_path: Path) -> Tuple[int, int]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as f:
        data = json.load(f)
    num_clients = data["num_clients"]
    num_servers = data["num_servers"]
    return num_clients, num_servers


def update_by_key_in_json_file(json_file_path: str, key: str, value: any) -> None:  # type: ignore
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Update the value
    data[key] = value

    # Write the updated data back to the file
    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {key} to {value} in {json_file_path}")


def main(args: argparse.Namespace) -> None:
    filename_base = "world_config"

    num_worlds = args.num_worlds
    if args.ignore_config:
        # If ignore_config is set, use the provided number of clients/servers
        num_clients = args.num_clients
        num_servers = args.num_servers

        assert num_clients > 0, "Number of clients must be greater than 0."
        assert num_servers > 0, "Number of servers must be greater than 0."
        exp_path = Path(".")
    else:
        # If ignore_config is not set, read the number of clients/servers from the config file

        exp_name = args.experiment_name
        # Check of experiment can be found
        exp_path = Path("configurations") / exp_name
        if not exp_path.exists():
            print(f"Experiment {exp_name} not found in configurations directory.")
            return

        num_clients, num_servers = parse_num_entities(exp_path / "config.json")
        update_by_key_in_json_file(exp_path / "config.json", "world_config", f"{filename_base}_0.json")

    base_cfg = load_config(exp_path / "config.json")
    num_servers = base_cfg.get("num_servers", 1)
    num_start_clients = base_cfg.get("num_start_clients", num_servers)

    for world_id in range(num_worlds):
        filename = f"{filename_base}_{world_id}"
        print(f"Generating world {filename} with {num_clients} clients and {num_servers} servers...")
        client_locations, client_delays = generate_clients(
            num_clients,
            args.width,
            args.height,
            delay_mean=30,
            delay_std=4,
            seed=args.seed,
        )
        server_locations = generate_servers(num_servers, args.width, args.height, seed=args.seed)

        num_alternative_servers = np.max([num_servers, 10])
        alternative_server_locations = generate_alternative_server_locations(
            num_alternative_servers, args.width, args.height, seed=args.seed
        )
        assignments = assign_clients_to_servers(
            num_clients,
            num_servers,
            client_locations,
            server_locations,
            seed=args.seed,
            generator_type=args.generator_type,
            num_start_clients=num_start_clients,
        )

        client_server_distances = calculate_distances(client_locations, server_locations)
        client_server_delays = calculate_delay_matrix(client_delays, client_locations, server_locations)
        print("Client-Server Distances:\n", client_server_distances)
        print("Client-Server Delays:\n", client_server_delays)
        print(assignments)
        plot_grid(
            client_locations,
            server_locations,
            args.width,
            args.height,
            plot_name=filename,
            assignments=assignments,
        )
        print(f"Plot saved as {filename}.png")
        config_path = exp_path / f"{filename}.json"
        save_config(
            config_path,
            client_locations,
            client_delays,
            server_locations,
            assignments,
            client_server_delays,
            alternative_server_locations=alternative_server_locations,
        )
        save_config(
            f"{filename}.json",
            client_locations,
            client_delays,
            server_locations,
            assignments,
            client_server_delays,
            alternative_server_locations=alternative_server_locations,
        )

    if args.expand:
        # If --expand is set, expand the configurations based on mutations
        mutations = {"rebalance_policy": ["non_iid", "random", "latency", "none"]}
        expand_configurations(exp_path / "config.json", num_worlds, mutations)


def plot_grid_from_config(
    config_path: Path,
    plot_name: str,
    overide_assignments: Optional[Dict[int, List[int]]] = None,
    file_path: Path = Path("."),
) -> None:
    with open(config_path, "r") as f:
        data = json.load(f)

    client_locations = np.array([[c["x"], c["y"]] for c in data["clients"]])
    server_locations = np.array([[s["x"], s["y"]] for s in data["servers"]])
    assignments = {s["id"]: s["assigned_clients"] for s in data["servers"]}

    if overide_assignments is not None:
        # Override the assignments with the provided ones
        assignments = overide_assignments
    print(f"assignments: {assignments}")

    plot_grid(
        client_locations,
        server_locations,
        100.0,
        100.0,
        plot_name,
        assignments=assignments,
        file_path=file_path,
    )


def expand_configurations(base_config: Path, num_worlds: int, mutations: dict) -> None:

    print(f"Mutating {base_config} {num_worlds} times with {mutations}")

    base_path = base_config.parent
    list_of_folder_names = []
    for world_id in range(num_worlds):
        for policy in mutations["rebalance_policy"]:
            exp_name = base_path.stem.replace("_template_w0", f"_{policy}_w{world_id}")

            print(f"Generating world '{exp_name}' with policy {policy}...")

            short_exp_name = f"{policy}_w{world_id}"
            new_path = base_path.parent / exp_name
            new_path.mkdir(parents=True, exist_ok=True)
            # Use shutill to copy the folder structure
            import shutil

            shutil.copytree(base_path, new_path, dirs_exist_ok=True)

            # Update the config file with the new policy
            config_file = new_path / "config.json"
            if not config_file.exists():
                print(f"Config file {config_file} does not exist. Skipping...")
                continue
            with open(config_file, "r") as f:
                config_data = json.load(f)
            config_data["rebalance_policy"] = policy
            config_data["world_config"] = f"world_config_{world_id}.json"
            config_data["name"] = short_exp_name
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            list_of_folder_names.append(exp_name)

    slurm_str = ["declare -a exps=("] + [f'"{name}"' for name in list_of_folder_names] + [")"]

    print("Generated SLURM array string:")
    print("\n".join(slurm_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client-Server Space Simulation")
    # Add positional arg with the experiment name
    parser.add_argument("experiment_name", type=str, help="Name of the experiment.")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=50,
        help="Number of clients. This is ignored when a config file is provided",
    )
    parser.add_argument(
        "--num_servers",
        type=int,
        default=5,
        help="Number of servers. This is ignored when a config file is provided",
    )
    parser.add_argument("--width", type=float, default=100.0, help="Max width of space")
    parser.add_argument("--height", type=float, default=100.0, help="Max height of space")
    # Add flag to ignore the config file
    parser.add_argument(
        "--ignore_config",
        "-i",
        action="store_true",
        help="Ignore the config file and use the provided number of clients/servers",
    )
    # Argument for number of worlds to generate
    parser.add_argument(
        "--num_worlds",
        "-n",
        type=int,
        default=1,
        help="Number of worlds to generate. This is ignored when a config file is provided",
    )

    # Add argument for the generator type "random" (default), "distance_best", "distance_worst"
    parser.add_argument(
        "--generator_type",
        "-g",
        type=str,
        choices=["random", "distance_best", "distance_worst"],
        default="random",
        help="Type of generator to use for client-server assignments",
    )

    parser.add_argument(
        "--expand",
        "-e",
        action="store_true",
        help="Expand configurations based on mutations",
    )
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)
