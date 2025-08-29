from collections import Counter
from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import (
    cdist,
    cityblock,  
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
def tvd(p, q):
    return 0.5 * cityblock(p, q)
def emd(p, q):
    """
    Earth Mover's Distance (EMD) between two distributions p and q.
    EMD is a measure of the distance between two probability distributions over a region D.
    It is defined as the minimum cost of transforming one distribution into another.
    """
    p = np_normalize(p)
    q = np_normalize(q)
    return cityblock(p, q)
def total_variation_distance(p_counts: list, q_counts: list, label_space):
    if isinstance(p_counts, list):
        p_counts = {label: count for label, count in zip(label_space, p_counts)}
    if isinstance(q_counts, list):
        q_counts = {label: count for label, count in zip(label_space, q_counts)}
    p = np.array([p_counts.get(label, 0) for label in label_space], dtype=float)
    q = np.array([q_counts.get(label, 0) for label in label_space], dtype=float)
    p /= p.sum()
    q /= q.sum()
    return 0.5 * np.abs(p - q).sum()
def compute_weighted_tvd(
    client_latencies: np.ndarray,  
    client_ids: List[int],  
    server_label_count: np.ndarray,  
    global_label_count: np.ndarray,  
) -> Tuple[float, np.ndarray, np.ndarray, List[int]]:
    """
    Compute weighted TVD where weight = (sample_count / latency), normalized.
    Returns:
        total_weighted_tvd (float)
        client_distributions (np.ndarray)
        client_weights (np.ndarray)
        client_ids (List[int])
    """
    client_sample_counts = server_label_count.sum(axis=1)
    global_label_distribution = np_normalize(global_label_count)
    client_distributions = server_label_count / server_label_count.sum(axis=1, keepdims=True)
    tvd_matrix = np.abs(client_distributions - global_label_distribution)
    scalar_tvds = 0.5 * np.sum(tvd_matrix, axis=1)  
    combined_weights = client_sample_counts / client_latencies
    client_weights = combined_weights / combined_weights.sum()
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
    return cdist(client_locations, server_locations, metric="euclidean")  
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
