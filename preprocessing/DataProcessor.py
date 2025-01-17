import io
import itertools
import os

import folium
import pandas as pd
from PIL import Image
from haversine import haversine, Unit


def read_ids_to_array(file_path):
    with open(file_path, 'r') as file:
        id_list = file.read().strip().split(',')
    return id_list


def get_in_box(sensor_locs, coordinates_of_box):
    return sensor_locs[(sensor_locs.latitude >= coordinates_of_box[3])
                       & (sensor_locs.latitude <= coordinates_of_box[0])
                       & (sensor_locs.longitude >= coordinates_of_box[1])
                       & (sensor_locs.longitude <= coordinates_of_box[2])]

def pairwise_distances(points_df, indices):
    distances = {}
    for i, j in itertools.combinations(indices, 2):
        distance = haversine((points_df.at[i, 'latitude'], points_df.at[i, 'longitude']),
                             (points_df.at[j, 'latitude'], points_df.at[j, 'longitude']),
                             unit=Unit.KILOMETERS)
        distances[(i, j)] = distance
    return distances

class DataProcessor:
    def __init__(self, data_option, sensor_locations_file):
        self.sensor_locations_file = sensor_locations_file
        # self.dataset_name = data_option[0]
        self.sensor_ids_file = data_option[1]
        self.dataset_file = data_option[2]
        self.coordinates = data_option[3]
        self.dataset_name = data_option[4]
        self.coordinates_bigger = data_option[5]
        self.this_map = folium.Map(prefer_canvas=True, zoom_start=50)

    def process_data(self):
        sensor_locations = pd.read_csv("Datasets/" + self.dataset_name + "/" + self.sensor_locations_file,
                                       index_col=0)

        in_box = get_in_box(sensor_locations, self.coordinates)
        in_comp_box = get_in_box(sensor_locations, self.coordinates_bigger)

        out_of_box = sensor_locations[(sensor_locations.latitude < self.coordinates[3])
                                      | (sensor_locations.latitude > self.coordinates[0])
                                      | (sensor_locations.longitude < self.coordinates[1])
                                      | (sensor_locations.longitude > self.coordinates[2])]

        return in_box, in_comp_box, out_of_box

    def save_filtered_data(self, in_box, filename):
        data = pd.read_hdf("../datasets/raw_data/" + self.dataset_name + "/" + self.dataset_file + ".h5")

        ids = in_box.sensor_id.tolist()  # the sensors we train and test with
        indices = in_box.index.tolist()  # the indices of the sensors we train and test with

        ids_dir = f"ids/{self.dataset_name}"
        indices_dir = f"indices/{self.dataset_name}"

        os.makedirs(ids_dir, exist_ok=True)
        os.makedirs(indices_dir, exist_ok=True)

        # save new subset of data
        filtered_dataset = data.iloc[:, indices]
        filtered_dataset.to_hdf("../preprocessing/Datasets/" + self.dataset_name + "/" + filename+".h5", key='subregion_test', mode='w')

        with open("ids/" + self.dataset_name + "/" + filename + ".txt", 'w') as file:
            file.write(','.join(map(str, ids)))

        with open("indices/" + self.dataset_name + "/" + filename + ".txt", 'w') as file:
            file.write(f"{len(indices)}\n")
            file.write(','.join(map(str, indices)))


    def reset_map(self):
        self.this_map = folium.Map(prefer_canvas=True, zoom_start=50)

    def plotDot(self, point, color):
        folium.CircleMarker(location=[point.latitude, point.longitude], radius=8, color=color, stroke=False, fill=True,
                            fill_opacity=0.8, opacity=1, popup=point.sensor_id, fill_color=color).add_to(self.this_map)

    def plot_data(self, name, in_box, in_comp_box, in_bigger, out_of_box):
        out_of_box.apply(self.plotDot, axis=1, args=("#000000",))
        in_comp_box.apply(self.plotDot, axis=1, args=("#0000FF",))
        in_bigger.apply(self.plotDot, axis=1, args=("#32cd32",))
        in_box.apply(self.plotDot, axis=1, args=("#FF0000",))
        self.this_map.fit_bounds(self.this_map.get_bounds())

        # TO SAVE
        os.makedirs("html", exist_ok=True)
        os.makedirs("images", exist_ok=True)

        self.this_map.save("html/" + name + ".html")
        img_data = self.this_map._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img.save("images/" + name + ".png")
        self.reset_map()
