# Thesis Project: Graph Neural Networks for Long-Term Traffic Forecasting


This project was completed as part of the Research Project course in the Computer Science and Engineering Bachelor's program at TU Delft.

Paper Link: "[Graph Neural Networks for Long-Term Traffic Forecasting](https://repository.tudelft.nl/record/uuid:4512a2ff-f0a1-4aac-89f6-c46e1944e7e8)"

The GNN model used is D2STGNN, which can be found here: https://github.com/zezhishao/d2stgnn?tab=readme-ov-file

## 1. Table of Contents

```text
configs         ->  training Configs and model configs for each dataset
dataloader      ->  pytorch dataloader
datasets        ->  raw data and processed data
model           ->  model implementation and training pipeline
output          ->  model checkpoint
preprocessing   ->  custom dataset creation with the use of coordinates
statistics      ->  dataset statistics, such as variation and standard deviation of data
error plots     ->  error plotting per each timestep
```

## 2. Requirements

```bash
pip install -r requirements.txt
```

## 3. Data Preparation

### 3.1 Download Data

The original datasets used in the model can be found here: [Google Drive](https://drive.google.com/drive/folders/1H3nl0eRCVl5jszHPesIPoPu1ODhFMSub?usp=sharing).

The custom datasets used for the experiments can be found here: [Google Drive](https://drive.google.com/drive/folders/1oFc0otdV3REoJiUJPHhCLx6kLtnq1vQ3?usp=sharing)

They should be downloaded to the code root dir and replace the `raw_data` and `sensor_graph` folder in the `datasets` folder by using this script or manually:

```bash
cd /path/to/project
unzip raw_data.zip -d ./datasets/
unzip sensor_graph.zip -d ./datasets/
rm {sensor_graph.zip,raw_data.zip}
mkdir log output
```

Alterbatively, the datasets can be found here:

- METR-LA and PEMS-BAY: These datasets were released by DCRNN[1]. Data can be found in its [GitHub repository](https://github.com/chnsh/DCRNN_PyTorch), where the sensor graphs are also provided.

- PEMS04 and PEMS08: These datasets were released by ASTGCN[2] and ASTGNN[3]. Data can also be found in its [GitHub repository](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).

### 3.2 Data Process


To create a custom dataset, you need to input the coordinates of the rectangle where you want the sensors to be located.
These coordinates can be entered into `box_coordinates` in the `preprocessing_run.py` file.

```bash
cd ./preprocessing; python preprocessing_run.py
```
Afterward you need to move the newly created h5 file from the Datasets folder in preprocessing to the datasets folder in the main project. 
Also, pair it with a compatible executable for generating training data.


Then to generate the necessary training data the following command will be run.

```bash
python datasets/raw_data/$DATASET_NAME/generate_training_data.py
```

Replace `$DATASET_NAME` with one of `METR-LA`, `PEMS-BAY`, `PEMS04`, `PEMS08`, or the custom dataset name.

The processed data is placed in `datasets/$DATASET_NAME`.

## 4. Training the D2STGNN Model

```bash
python main.py --dataset=$DATASET_NAME
```

E.g., `python main.py --dataset=METR-LA`.

## 5 Loading a Pretrained D2STGNN Model

Check the config files of the dataset in `configs/$DATASET_NAME`, and set the startup args to test mode.

Download the pre-trained model files in for the original datasets [Google Drive](https://drive.google.com/drive/folders/18nkluGajYET2F9mxz3Kl6jcFVAAUGfpc?usp=sharing) into the `output` folder and run the command line in `4`.

## 6 Plotting and statistics

After each epoch a detailed log will be printed out. 
In order to plot the errors over time, you need to input the outputted log into the data variable inside `error_plots.py`.

```bash
python error_plots.py
```

To show dataset statistic you need to input the path of the h5 files of the datasets into the `file_paths` variable inside `dataset_statistics.py`.

```bash
python dataset_statistics.py
```

## 7 Results and Visualization

**Errors converge** to a sort of **logarithmic growth** as the number of epochs is increased.

![log_convergence.gif](..%2Ffinal_plots_latex%2Fgif_prezentare.gif)

Logarithmic curves fitted to the errors.

![logs_plotted.png](..%2Ffinal_plots_latex%2FFigure_1.png)

**Prediction performance differs** from the **type of curve** a traffic jams creates. 
The blue is the real data, orange is predicted. 
Also, the data shown here is the traffic speed of vehicles passing through sensors.
The sensor ids are 767350 and 717499, which can be found in the `LA_AND_BAY_traffic_sensors_map.html` highlited in yellow and red respectively.


![Horizon 1 (10).png](..%2F..%2F..%2FDownloads%2FHorizon%201%20%2810%29.png)
## References

[1] Atwood J, Towsley D. Diffusion-convolutional neural networks[J]. Advances in neural information processing systems, 2016, 29: 1993-2001.

[2] Guo S, Lin Y, Feng N, et al. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 922-929.

[3] Guo S, Lin Y, Wan H, et al. Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting[J]. IEEE Transactions on Knowledge and Data Engineering, 2021.
