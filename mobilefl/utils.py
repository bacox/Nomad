from collections import Counter
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import (
    cdist,
    cityblock,  # for TVD
)


def sum_dicts(dict1, dict2) -> dict:
    """
    Sum two dictionaries with the same keys.
    """
    c1 = Counter(dict1)
    c2 = Counter(dict2)
    c1.update(c2)
    return dict(c1)


def np_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


# Function to compute the Total Variation Distance (TVD) between two probability distributions
# TVD is a measure of the difference between two probability distributions and is defined as:
# TVD(p, q) = 0.5 * sum(|p_i - q_i|) for all i
# Here, we use the `cityblock` function from `scipy.spatial.distance`, which computes the Manhattan distance,
# to calculate the sum of absolute differences between the two distributions.
def tvd(p, q):
    return 0.5 * cityblock(p, q)


def emd(p, q):
    """
    Earth Mover's Distance (EMD) between two distributions p and q.
    EMD is a measure of the distance between two probability distributions over a region D.
    It is defined as the minimum cost of transforming one distribution into another.
    """
    # Ensure both distributions are normalized
    p = np_normalize(p)
    q = np_normalize(q)

    # Calculate the EMD using the cityblock distance
    return cityblock(p, q)


def total_variation_distance(p_counts: list, q_counts: list, label_space):
    # Normalize to probability vectors over global label space

    # Make sure both p_counts and q_counts are dictionaries, use label_space to ensure all labels are present
    if isinstance(p_counts, list):
        p_counts = {label: count for label, count in zip(label_space, p_counts)}
    if isinstance(q_counts, list):
        q_counts = {label: count for label, count in zip(label_space, q_counts)}

    p = np.array([p_counts.get(label, 0) for label in label_space], dtype=float)
    q = np.array([q_counts.get(label, 0) for label in label_space], dtype=float)

    p /= p.sum()
    q /= q.sum()

    return 0.5 * np.abs(p - q).sum()


# def compute_weighted_tvd(
#     client_latencies: np.ndarray,
#     client_ids: List[int],
#     server_label_count: np.ndarray,
#     global_label_count: np.ndarray,
# ):
# #     # latencies = self.get_clients_latencies()
# #     latencies = client_latencies
# #     # server_label_count, client_ids = self.collect_client_label_count()
# #     # print(f"Server {self.server_id} label count: {server_label_count}")

# #     global_label_count = np_normalize(global_label_count)

# #     # Normalize label counts to distributions per client
# #     client_distributions = server_label_count / server_label_count.sum(axis=1, keepdims=True)

# #     # print(f"Client distributions: {client_distributions}")
# #     # Compute inverse latency weights (lower latency = higher influence)
# #     client_weights = 1.0 / latencies
# #     client_weights /= client_weights.sum()  # Normalize weights to sum to 1

# #     # Per-client, per-label absolute difference
# #     tvd_matrix = np.abs(client_distributions - global_label_count)

# #     # Scalar weighted TVD per client
# #     scalar_tvds = 0.5 * np.sum(tvd_matrix, axis=1)
# #     # print("Scalar TVDs per client:", scalar_tvds)
# #     weighted_scalar_tvds = client_weights * scalar_tvds

# #     # print("Weighted Scalar TVDs per client:", weighted_scalar_tvds)

# #     total_weighted_tvd = np.sum(weighted_scalar_tvds)
# #     # print(f"Weighted TVD per client: {weighted_scalar_tvds}")
# #     # print(f"Total weighted TVD for server: {total_weighted_tvd:.4f}")
# #     # print(f"Client IDs: {client_ids}")
# #     # print(f"Client distributions:\n{client_distributions}")
# #     # exit()

# #     return total_weighted_tvd, client_distributions, client_weights, client_ids

#         # Normalize global label distribution
#     global_label_distribution = np_normalize(global_label_count)

#     # Normalize each client's label histogram to get a distribution
#     client_distributions = server_label_count / server_label_count.sum(axis=1, keepdims=True)

#     # Compute per-client TVD against global label distribution
#     tvd_matrix = np.abs(client_distributions - global_label_distribution)
#     scalar_tvds = 0.5 * np.sum(tvd_matrix, axis=1)  # shape: (num_clients,)

#     # Weight by number of samples per client
#     client_weights = client_sample_counts / np.sum(client_sample_counts)

#     # Weighted sum of TVDs
#     weighted_scalar_tvds = client_weights * scalar_tvds
#     total_weighted_tvd = np.sum(weighted_scalar_tvds)

#     return total_weighted_tvd, client_distributions, client_weights, client_ids


def compute_weighted_tvd(
    # client_sample_counts: np.ndarray,  # shape: (num_clients,)
    client_latencies: np.ndarray,  # shape: (num_clients,)
    client_ids: List[int],  # list of client IDs
    server_label_count: np.ndarray,  # shape: (num_clients, num_classes)
    global_label_count: np.ndarray,  # shape: (num_classes,)
) -> Tuple[float, np.ndarray, np.ndarray, List[int]]:
    """
    Compute weighted TVD where weight = (sample_count / latency), normalized.

    Returns:
        total_weighted_tvd (float)
        client_distributions (np.ndarray)
        client_weights (np.ndarray)
        client_ids (List[int])
    """
    # 1. Derive sample counts from label counts
    client_sample_counts = server_label_count.sum(axis=1)

    # Normalize global label distribution
    global_label_distribution = np_normalize(global_label_count)

    # Normalize client histograms to probability distributions
    client_distributions = server_label_count / server_label_count.sum(axis=1, keepdims=True)

    # Compute per-client TVD
    tvd_matrix = np.abs(client_distributions - global_label_distribution)
    scalar_tvds = 0.5 * np.sum(tvd_matrix, axis=1)  # shape: (num_clients,)

    # Combined weight = sample_count / latency
    combined_weights = client_sample_counts / client_latencies
    client_weights = combined_weights / combined_weights.sum()

    # Weighted TVD
    weighted_scalar_tvds = client_weights * scalar_tvds
    total_weighted_tvd = np.sum(weighted_scalar_tvds)

    return total_weighted_tvd, client_distributions, client_weights, client_ids


def calculate_distances(client_locations: np.ndarray, server_locations: np.ndarray) -> np.ndarray:
    """
    Calculate the distance matrix between clients and servers.
    :param client_locations: Array of client locations (shape: num_clients x 2).
    :param server_locations: Array of server locations (shape: num_servers x 2).
    :return: Distance matrix (shape: num_clients x num_servers).
    """
    return cdist(client_locations, server_locations, metric="euclidean")  # type: ignore


def calculate_delay_matrix(
    client_delays: np.ndarray,
    client_locations: np.ndarray,
    server_locations: np.ndarray,
) -> np.ndarray:
    """
    Calculate the delay matrix between clients and servers based on client delays and distances.
    :param client_delays: Array of client delays (shape: num_clients).
    :param client_locations: Array of client locations (shape: num_clients x 2).
    :param server_locations: Array of server locations (shape: num_servers x 2).
    :return: Delay matrix (shape: num_clients x num_servers).
    """
    distances = calculate_distances(client_locations, server_locations)
    delay_matrix = np.zeros(distances.shape)
    for i in range(len(client_delays)):
        delay_matrix[i, :] = client_delays[i] + 1 * distances[i, :]
    return delay_matrix
