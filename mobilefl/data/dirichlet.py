from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from mobilefl.config import Config
from mobilefl.data.subset import CustomSubset
def split_noniid(train_labels, alpha, n_clients):
    """Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha and returns the labels corresponding to each client's data.
    Args:
        train_labels: ndarray of train_labels.
        alpha: the parameter of Dirichlet distribution.
        n_clients: number of clients.
    Returns:
        client_idcs: a list containing sample idcs of clients.
        client_lbls: a list containing labels corresponding to each client's data.
    """
    assert n_clients > 0, "Number of clients must be greater than 0."
    n_classes = np.array(train_labels).max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    print(f"Splitting data into {n_clients} clients with Dirichlet distribution (alpha={alpha})")
    client_idcs = [[] for _ in range(n_clients)]
    client_lbls = [[] for _ in range(n_clients)]
    for c_idcs, fracs, lbl in zip(class_idcs, label_distribution, range(n_classes)):
        for i, idcs in enumerate(np.split(c_idcs, (np.cumsum(fracs)[:-1] * len(c_idcs)).astype(int))):
            client_idcs[i] += [idcs]
            client_lbls[i] += [lbl] * len(idcs)  
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    client_lbls = [np.array(lbls) for lbls in client_lbls]
    return ensure_min_data_per_client(client_idcs, client_lbls, 10)
def ensure_min_data_per_client(client_idcs, client_lbls, min_samples_per_client):
    client_sample_counts = np.array([len(indices) for indices in client_idcs])
    under_threshold_clients = np.where(client_sample_counts < min_samples_per_client)[0]
    over_threshold_clients = np.where(client_sample_counts > min_samples_per_client)[0]
    donor_clients_sorted = over_threshold_clients[np.argsort(client_sample_counts[over_threshold_clients])[::-1]]
    for under_client in under_threshold_clients:
        needed_samples = min_samples_per_client - client_sample_counts[under_client]
        for over_client in donor_clients_sorted:
            if needed_samples <= 0:
                break
            transferable_samples = client_sample_counts[over_client] - min_samples_per_client
            if transferable_samples > 0:
                transfer_samples = min(transferable_samples, needed_samples)
                transfer_indices = client_idcs[over_client][:transfer_samples]
                client_idcs[over_client] = client_idcs[over_client][transfer_samples:]
                client_idcs[under_client] = np.append(client_idcs[under_client], transfer_indices)
                transfer_labels = client_lbls[over_client][:transfer_samples]
                client_lbls[over_client] = client_lbls[over_client][transfer_samples:]
                client_lbls[under_client] = np.append(client_lbls[under_client], transfer_labels)
                client_sample_counts[over_client] -= transfer_samples
                client_sample_counts[under_client] += transfer_samples
                needed_samples -= transfer_samples
    return client_idcs, client_lbls
def get_dataset(
    train_data,
    test_data,
    alpha,
    n_clients,
    config: Config,
    server_id=None,
    out_path: Path = Path(__file__).parent.parent,
) -> Tuple[List[CustomSubset], List[CustomSubset], Dict[str, any]]:
    """
    Splits a dataset into non-IID partitions for federated learning and prepares client-specific
    training and testing datasets.
    Args:
        train_data (Dataset): The training dataset containing data and labels.
        test_data (Dataset): The testing dataset containing data and labels.
        alpha (float): The Dirichlet distribution parameter controlling the degree of non-IID-ness.
        n_clients (int): The number of clients to split the data among.
        config (Config): Configuration object containing settings and file paths.
        server_id (int, optional): The server ID for which the client indices and labels are generated.
                                   If None, data is split for all clients.
    Returns:
        tuple: A tuple containing:
            - client_train_datasets (list): A list of training datasets for each client.
            - client_test_datasets (list): A list of testing datasets for each client.
            - data_info (dict): A dictionary containing metadata about the dataset, including:
                - "classes": The list of class names.
                - "num_classes": The number of unique classes.
    Raises:
        ValueError: If the required client indices or labels are not found in the configuration file
                    when `var_control` is enabled.
    Notes:
        - `client_idcs` refers to the indices of the data samples assigned to each client. These indices
          are used to partition the dataset into client-specific subsets.
        - The function supports both generating new splits and loading pre-existing splits from a
          configuration file when `var_control` is enabled.
        - The data distribution across clients is displayed using the `display_data_distribution` function.
    """
    data_info = {}
    data_info["classes"] = train_data.classes
    data_info["num_classes"] = len(train_data.classes)
    labels = np.concatenate([np.array(train_data.targets), np.array(test_data.targets)], axis=0)
    train_data_classes = train_data.classes
    test_data_classes = test_data.classes
    dataset = ConcatDataset([train_data, test_data])
    updates = {}
    if server_id is not None:
        if not config.get("var_control"):
            client_idcs, client_labels = split_noniid(labels, alpha, n_clients)
            updates[f"server_{server_id}_client_idcs"] = [indexes.tolist() for indexes in client_idcs]
            updates[f"server_{server_id}_client_labels"] = [labels.tolist() for labels in client_labels]
            new_path = config.file_as_str().replace(".json", "_client_idcs.json")
            config.write_config(updates, new_path)
        else:
            new_path = config.file_as_str().replace(".json", "_client_idcs.json")
            new_config = Config(new_path)
            if not new_config.has_key(f"server_{server_id}_client_idcs"):
                raise ValueError(f"server_{server_id}_client_idcs not found in the config file.")
            if not new_config.has_key(f"server_{server_id}_client_labels"):
                raise ValueError(f"server_{server_id}_client_labels not found in the config file.")
            client_idcs = [np.array(indexes) for indexes in new_config.get(f"server_{server_id}_client_idcs")]
            client_labels = [np.array(labels) for labels in new_config.get(f"server_{server_id}_client_labels")]
    else:
        if config.get("var_control"):
            new_path = config.file_as_str().replace(".json", "_client_idcs.json")
            new_config = Config(new_path)
            client_idcs = [np.array(indexes) for indexes in new_config.get("server_all_server_level_idcs")]
            client_labels = [np.array(labels) for labels in new_config.get("server_all_server_level_labels")]
        else:
            client_idcs, client_labels = split_noniid(labels, alpha, n_clients)
            updates["server_all_server_level_idcs"] = [indexes.tolist() for indexes in client_idcs]
            updates["server_all_server_level_labels"] = [labels.tolist() for labels in client_labels]
            new_path = config.file_as_str().replace(".json", "_client_idcs.json")
            config.write_config(updates, new_path)
    display_data_distribution(client_idcs, labels, data_info["num_classes"], n_clients, alpha, out_path)
    client_train_idcs, client_test_idcs, client_train_lbls, client_test_lbls = (
        [],
        [],
        [],
        [],
    )
    for idcs, lbls in zip(client_idcs, client_labels):
        train_idcs, test_idcs, train_lbls, test_lbls = train_test_split(idcs, lbls, train_size=0.9, random_state=42)
        client_train_idcs.append(train_idcs)
        client_test_idcs.append(test_idcs)
        client_train_lbls.append(train_lbls)
        client_test_lbls.append(test_lbls)
    client_train_datasets, client_test_datasets = [], []
    for train_idcs, test_idcs, train_labels, test_labels in zip(
        client_train_idcs, client_test_idcs, client_train_lbls, client_test_lbls
    ):
        client_train_datasets.append(CustomSubset(dataset, train_idcs, train_labels, train_data_classes))
        client_test_datasets.append(CustomSubset(dataset, test_idcs, test_labels, test_data_classes))
    return client_train_datasets, client_test_datasets, data_info
def display_data_distribution(
    client_idcs, train_labels, num_classes, n_clients, alpha, out_path: Path = Path(__file__).parent.parent
):
    dataset_split_method = "Dirichlet"
    param_descr = "alpha={}".format(alpha)
    file_path = out_path / f"data_distribution_{dataset_split_method}_{param_descr}.png"
    plt.figure(figsize=(20, 6))
    label_distribution = [[] for _ in range(num_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[train_labels[idx]].append(c_id)
    plt.hist(
        label_distribution,
        stacked=True,
        bins=np.arange(-0.5, n_clients + 1.5, 1),
        label=["Class {}".format(i) for i in range(num_classes)],
        rwidth=0.5,
    )
    plt.xticks(np.arange(n_clients), ["Client {}".format(c_id) for c_id in range(n_clients)])
    plt.ylabel("Number of samples")
    plt.xlabel("Client ID")
    plt.legend()
    plt.title("Federated " + " Display ({}, {})".format(dataset_split_method, param_descr))
    plt.savefig(file_path)
    print(f"Data distribution plot saved as {file_path}")
    plt.show()
