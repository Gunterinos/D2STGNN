import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths to your HDF5 files
file_paths = ['datasets/raw_data/METR-LA-one-intersection-20/metr-la-one-intersection-20.h5',
              'datasets/raw_data/METR-la-straight-35/metr-la-straight-35.h5',
              'datasets/raw_data/METR-LA-three-intersections-55/metr-la-three-intersections-55.h5',
              'datasets/raw_data/METR-LA-two-intersections-40/metr-la-two-intersections-40.h5']

# Number of points to consider from each dataset
num_points = 500000

# Initialize lists to store the results
means = []
variances = []
std_devs = []
zero_percentages = []  # Store the percentage of zeros
zero_sequences_counts = []  # Store the number of zero sequences
largest_zero_sequences = []  # Store the largest zero sequences
dataset_names = []

# Function to count sequences of zeros in an array
def count_zero_sequences(arr):
    count = 0
    in_sequence = False
    for val in arr:
        if val == 0 and not in_sequence:
            in_sequence = True
            count += 1
        elif val != 0:
            in_sequence = False
    return count

# Function to find the largest sequence of zeros in an array
def largest_zero_sequence(arr):
    max_len = 0
    current_len = 0
    for val in arr:
        if val == 0:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
        else:
            current_len = 0
    return max_len

# Iterate over the file paths
for file_path in file_paths:
    # Extract the dataset name from the file path
    dataset_name = os.path.basename(os.path.dirname(file_path))
    dataset_names.append(dataset_name)

    with h5py.File(file_path, 'r') as h5_file:
        dataset_path = 'subregion_test/block0_values'  # Assuming this is constant for all files, as per your code

        if dataset_path in h5_file:
            data = h5_file[dataset_path][-num_points:]  # Take only the last `num_points` points
            zero_percentage_per_node = []  # Store the percentage of zeros for each node
            zero_sequence_count_per_node = []  # Store the number of zero sequences for each node
            largest_zero_sequence_per_node = []  # Store the largest zero sequence for each node
            mean_per_node = []  # Store the mean for each node
            variance_per_node = []  # Store the variance for each node
            std_dev_per_node = []  # Store the standard deviation for each node

            for node_data in data.T:  # Iterate over nodes
                non_zero_data = node_data[node_data != 0]
                zero_count = len(node_data) - len(non_zero_data)  # Calculate the number of zeros
                zero_percentage = (zero_count / len(node_data)) * 100  # Calculate the percentage of zeros
                zero_percentage_per_node.append(zero_percentage)

                zero_sequence_count = count_zero_sequences(node_data)  # Calculate the number of zero sequences
                zero_sequence_count_per_node.append(zero_sequence_count)

                largest_zero_seq = largest_zero_sequence(node_data)  # Find the largest zero sequence
                largest_zero_sequence_per_node.append(largest_zero_seq)

                if non_zero_data.size > 0:
                    mean = np.mean(non_zero_data)
                    variance = np.var(non_zero_data)
                    std_dev = np.sqrt(variance)
                    mean_per_node.append(mean)
                    variance_per_node.append(variance)
                    std_dev_per_node.append(std_dev)
                else:
                    print(f'The last {num_points} points of a node in the dataset {dataset_path} in file {file_path} contain only zero values.')
                    mean_per_node.append(np.nan)
                    variance_per_node.append(np.nan)
                    std_dev_per_node.append(np.nan)

            zero_percentages.append(np.mean(zero_percentage_per_node))
            zero_sequences_counts.append(np.mean(zero_sequence_count_per_node))
            largest_zero_sequences.append(np.mean(largest_zero_sequence_per_node))
            means.append(np.mean(mean_per_node))
            variances.append(np.mean(variance_per_node))
            std_devs.append(np.mean(std_dev_per_node))
        else:
            print(f'Dataset {dataset_path} not found in file {file_path}.')
            means.append(np.nan)
            variances.append(np.nan)
            std_devs.append(np.nan)
            zero_percentages.append(np.nan)
            zero_sequences_counts.append(np.nan)
            largest_zero_sequences.append(np.nan)

# Create a DataFrame with the results
data = {
    'Dataset': dataset_names,
    'Mean': np.round(means, 2),  # Round mean values to 2 decimal places
    'Variance': np.round(variances, 2),  # Round variance values to 2 decimal places
    'Standard Deviation': np.round(std_devs, 2),  # Round standard deviation values to 2 decimal places
    'Average Percentage of Zeros per Node': np.round(zero_percentages, 2),  # Round zero percentages to 2 decimal places
    'Average Number of Zero Sequences per Node': np.round(zero_sequences_counts, 2),  # Round zero sequences counts to 2 decimal places
    'Average Largest Zero Sequence per Node': np.round(largest_zero_sequences, 2)  # Round largest zero sequences to 2 decimal places
}
df = pd.DataFrame(data)

# Create a table plot
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Increase font size for better readability
table.auto_set_font_size(False)
table.set_fontsize(12)

# Save the table as an image
plt.savefig('dataset_statistics.png', bbox_inches='tight', dpi=300)

# Display the table
plt.show()