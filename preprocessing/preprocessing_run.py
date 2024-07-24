import os
import pickle

import pandas as pd
import cv2

from DataProcessor import DataProcessor
from gen_adj_mx import get_adjacency_matrix

# Coordinate definitions
box_coordinates = [34.188469, -118.509482, -118.439572, 34.132489]
box_coordinates_bigger = [34.243283, -118.594800, -118.253022, 34.127267]

# Data options: [road_distance_small, sensor_ids_file, dataset_file, coordinates, dataset_name, coordinates_bigger, distances_filename]
metrla = ["METR-LA", "metr_ids.txt", "metr-la", box_coordinates, "METR-LA", box_coordinates_bigger,
          "distances_la_2012.csv"]

sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 1 / 3, 0.25]
suffixes = ["100", "075", "050", "030", "025"]

data_option = metrla
h5_filename = data_option[2]
distances_filename = data_option[6]


def save_adj_mx(filename, option):
    with open(f"ids/{option[4]}/{filename}.txt") as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(f"sensor_graph/{option[6]}", dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)

    output_dir = f"../preprocessing/Datasets/sensor_graph/adj_mxs/{option[4]}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/{filename}.pkl", 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)


def concat_images(option):
    large_images = []
    small_images = []
    for size, suffix in zip(sizes, suffixes):
        large_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-large-{suffix}.png"))
        small_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-small-{suffix}.png"))
    large_image = cv2.vconcat(large_images)
    small_image = cv2.vconcat(small_images)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-large.png", large_image)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-small.png", small_image)


processor = DataProcessor(metrla, sensor_locations_file)
within_box, in_comparison_box, outside_of_box = processor.process_data()
processor.save_filtered_data(within_box, "metr-la-half-100")
save_adj_mx(f"{metrla[2]}-half-100", metrla)
processor.plot_data("metr-la-half-100", within_box, in_comparison_box, in_comparison_box, outside_of_box)
