from __future__ import absolute_import, division, print_function
import numpy as np


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    Constructs an adjacency matrix from a DataFrame containing distances between sensors.

    :param distance_df: DataFrame with three columns: [from, to, distance].
    :param sensor_ids: List of sensor ids.
    :param normalized_k: Entries that become lower than normalized_k after normalization
                         are set to zero for sparsity.
    :return: Tuple of (sensor_ids, sensor_id_to_ind, adjacency_matrix).
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.full((num_sensors, num_sensors), np.inf, dtype=np.float32)

    # Build sensor id to index map.
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}

    # Fill the distance matrix.
    for from_sensor, to_sensor, distance in distance_df.values:
        if from_sensor in sensor_id_to_ind and to_sensor in sensor_id_to_ind:
            dist_mx[sensor_id_to_ind[from_sensor], sensor_id_to_ind[to_sensor]] = distance

    # Calculate the standard deviation as theta.
    valid_distances = dist_mx[~np.isinf(dist_mx)]
    std = valid_distances.std()

    # Compute the adjacency matrix using a Gaussian kernel.
    adj_mx = np.exp(-np.square(dist_mx / std))

    # Set entries lower than normalized_k to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0

    return sensor_ids, sensor_id_to_ind, adj_mx
