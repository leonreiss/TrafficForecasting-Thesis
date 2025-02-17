# -*- coding: utf-8 -*-

############### Installing Pytorch Geometric ###############
!pip install torch_geometric
import torch
import torch_geometric
print("PyTorch Version:", torch.__version__)
print("PyTorch Geometric Version:", torch_geometric.__version__)
#PyTorch Version: 2.2.1+cu121
#PyTorch Geometric Version: 2.5.2

############ Installing Pytorch Geometric Temporal ############
! pip install torch-geometric-temporal
! pip freeze | grep torch-geometric-temporal
#Result: torch-geometric-temporal==0.54.0

############ Updating Import Path in tsagcn.py ################

# Note: This script updates the import path in 'tsagcn.py' for compatibility with the latest 'torch_geometric' library.
# It creates a backup of the original file before replacing the old import statement with the new one.
# The path to the file

file_path = '/usr/local/lib/python3.10/dist-packages/torch_geometric_temporal/nn/attention/tsagcn.py'

# Create a backup file
!cp {file_path} {file_path}.bak

# Reads the contents of the file
with open(file_path, 'r') as file:content = file.readlines()

# Change the specific row
content = [line.replace("from torch_geometric.utils.to_dense_adj import to_dense_adj", "from torch_geometric.utils import to_dense_adj") for line in content]

# Write the changed content back into the file
with open(file_path, 'w') as file: file.writelines(content)


######################Installing all other libraries ###############

#!pip install pmdarima
#!pip install matplotlib
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from torch_geometric_temporal.dataset import PedalMeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import json
import urllib.request from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import GConvLSTM
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

######################## Data cleansing ########################

def trim_to_quartiles(data, lower_q=0.25, upper_q=0.75):
    lower_quartile = np.quantile(data, lower_q)
    upper_quartile = np.quantile(data, upper_q)
    trimmed_data = np.clip(data, lower_quartile, upper_quartile)
    return trimmed_data

def correct_specific_outliers(features, targets):
    # The last target is trimmed to lower quartile
    trimmed_target = trim_to_quartiles(targets, lower_q=0.25, upper_q=0.25)
    targets[-1] = trimmed_target[-1]

    # The first feature is trimmed to upper quartile
    trimmed_features = trim_to_quartiles(features[:, 0], lower_q=0.75, upper_q=0.75)
    features[0, 0] = trimmed_features[0]

    return features, targets

def correct_dataset_outliers(dataset):
    number_of_stations = len(dataset.features[0])
    for station_index in range(number_of_stations):
        station_features = np.array([features[station_index] for features in dataset.features])
        station_targets = np.array([targets[station_index] for targets in dataset.targets])

        corrected_features, corrected_targets = correct_specific_outliers(station_features, station_targets)

        for idx in range(len(dataset.features)):
            dataset.features[idx][station_index] = corrected_features[idx]
            dataset.targets[idx][station_index] = corrected_targets[idx]
    return dataset

# plot style attributes
linestyles = ['-', '--', ':', '-.']
greyscales = ['#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3']
number_of_stations = len(dataset.features[0])

def plot_station_data(dataset, station_index, linestyles, greyscales):
    station_features = np.array([features[station_index] for features in dataset.features])
    station_targets = np.array([targets[station_index] for targets in dataset.targets])
    plt.figure(figsize=(15, 5))
    for lag_index in range(station_features.shape[1]):
        plt.plot(station_features[:, lag_index], label=f"Features (Lag {lag_index})",
                 linestyle=linestyles[lag_index % len(linestyles)], color=greyscales[lag_index % len(greyscales)])
    plt.plot(station_targets, label="Target (Demand)", color='red', linewidth=2.5)
    plt.title(f"Time series of demand for Station {station_index + 1}")
    plt.xlabel("Time (in days)")
    plt.ylabel("Number of deliveries")
    plt.legend()
    plt.show()

# main process
loader = PedalMeDatasetLoader()
dataset = loader.get_dataset()
dataset = correct_dataset_outliers(dataset)
#for station_index in range(number_of_stations):
#    plot_station_data(dataset, station_index, linestyles, greyscales)

###################### ARIMA ######################

# Preparation of time series data for each station for ARIMA by breaking up the graph data into single time series data
station_series_dfs = []
for station_index in range(len(dataset.targets[0])):
    station_series = [target[station_index] for target in dataset.targets]
    df = pd.DataFrame(station_series, columns=['Demand'])
    station_series_dfs.append(df)

# Definition of the function for checking stationarity and application of differentiation
def check_stationarity_and_difference(series):
    result = adfuller(series.dropna())
    if result[1] > 0.05:  # Non-stationary
        series = series.diff().dropna()
    return series

# Prepare dictionaries to store predictions and actuals per station
arima_train_predictions = {}
arima_test_predictions = {}

# Metrics per Station
arima_train_metrics = {}
arima_test_metrics = {}

#Overall metrics
total_arima_train_actuals = []
total_arima_train_predictions = []
total_arima_test_actuals = []
total_arima_test_predictions = []

#Defining number of timesteps for testset
test_size = 5

for i, df in enumerate(station_series_dfs):
    print(f"Station {i+1}:")
    time_series = df['Demand']
    stationary_series = check_stationarity_and_difference(time_series)

    # Split in training and testing
    train = stationary_series.iloc[:len(station_series_dfs[0]) - test_size]
    test = stationary_series.iloc[len(station_series_dfs[0]) - test_size:]
    print(f"Station {i+1}: Number of train data for ARIMA: {len(train)}")
    print(f"Station {i+1}: Number of test data for ARIMA: {len(test)}")

    # Fit the ARIMA model
    model = auto_arima(train, seasonal=False, start_p=0, start_q=0, max_p=5, max_q=5, stepwise=True, trace=False)
    train_predictions = model.predict_in_sample()
    test_predictions = model.predict(n_periods=test_size)

    # Store predictions per station
    arima_train_predictions[f"Station_{i+1}"] = train_predictions
    arima_test_predictions[f"Station_{i+1}"] = test_predictions

    # Overall performance metrics
    total_arima_train_actuals.extend(train)
    total_arima_train_predictions.extend(train_predictions)
    total_arima_test_actuals.extend(test)
    total_arima_test_predictions.extend(test_predictions)

    # Calculate and store metrics
    train_mse = mean_squared_error(train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train, train_predictions)
    arima_train_metrics[f"Station_{i+1}"] = (train_mse, train_rmse, train_mae)

    test_mse = mean_squared_error(test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test, test_predictions)
    arima_test_metrics[f"Station_{i+1}"] = (test_mse, test_rmse, test_mae)

    print(f"Train MSE ARIMA: {train_mse}, Train RMSE ARIMA: {train_rmse}, Train MAE ARIMA: {train_mae}")
    print(f"Test MSE ARIMA: {test_mse}, Test RMSE ARIMA: {test_rmse}, Test MAE ARIMA: {test_mae}")

# Calculate overall train metrics
overall_train_mse = mean_squared_error(total_arima_train_actuals, total_arima_train_predictions)
overall_train_rmse = np.sqrt(overall_train_mse)
overall_train_mae = mean_absolute_error(total_arima_train_actuals, total_arima_train_predictions)
print(f"Overall ARIMA - Train MSE: {overall_train_mse}, Train RMSE: {overall_train_rmse}, Train MAE: {overall_train_mae}")

# Calculate overall test metrics
overall_test_mse = mean_squared_error(total_arima_test_actuals, total_arima_test_predictions)
overall_test_rmse = np.sqrt(overall_test_mse)
overall_test_mae = mean_absolute_error(total_arima_test_actuals, total_arima_test_predictions)
print(f"Overall ARIMA - Test MSE: {overall_test_mse}, Test RMSE: {overall_test_rmse}, Test MAE: {overall_test_mae}")

##################### Graph based data split ###########################

# Splitting of data using prebuild function of pytorch spatio temporal
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.85)
num_test_steps = len(test_dataset.features)
print(f"Number of test steps for GCRN models: {num_test_steps}")
num_train_timesteps = len(train_dataset.features)
print(f"Number of timesteps in train dataset: {num_train_timesteps}")

#####################GConvGRU###########################

# Define the RecurrentGCN model
class RecurrentGCN_GRU(nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN_GRU, self).__init__()
        self.recurrent1 = GConvGRU(node_features, filters, 2)
        self.dropout = nn.Dropout(p=0.3)  # Adjust Dropout here!
        self.linear = nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear(h)
        return h

# Initialize the model and optimizer
model_gru = RecurrentGCN_GRU(node_features=4, filters=32)
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=0.01)

# Prepare to store loss data for plotting
train_losses_gru = []
test_losses_gru = []

# Initialize lists to store metrics for each station
train_actuals_gru = {i: [] for i in range(len(dataset.targets[0]))}
train_predictions_gru = {i: [] for i in range(len(dataset.targets[0]))}
test_actuals_gru = {i: [] for i in range(len(dataset.targets[0]))}
test_predictions_gru = {i: [] for i in range(len(dataset.targets[0]))}

# Initialize lists to store overall metrics
total_train_actuals_gru = []
total_train_predictions_gru = []
total_test_actuals_gru = []
total_test_predictions_gru = []

# Train the model
for epoch in tqdm(range(30)):
    epoch_train_loss = 0
    num_snapshots = 0  # Initialize snapshot counter

    model_gru.train() #start training mode

    # Clear the lists for each epoch
    for i in range(len(dataset.targets[0])):
        train_actuals_gru[i].clear()
        train_predictions_gru[i].clear()

    for time, snapshot in enumerate(train_dataset):
        optimizer_gru.zero_grad()
        y_hat = model_gru(snapshot.x, snapshot.edge_index, snapshot.edge_weight).squeeze()
        cost = torch.mean((y_hat - snapshot.y)**2)
        cost.backward()
        optimizer_gru.step()
        epoch_train_loss += cost.item()
        num_snapshots += 1

        # Store train predictions and actuals
        for i in range(len(dataset.targets[0])):
            train_predictions_gru[i].append(y_hat[i].item())
            train_actuals_gru[i].append(snapshot.y[i].item())

    train_losses_gru.append(epoch_train_loss / num_snapshots if num_snapshots > 0 else 0)

    # Evaluate the model
    model_gru.eval()
    epoch_test_loss = 0
    num_test_snapshots = 0  # Initialize snapshot counter for testing

    # Clear the lists for each epoch
    for i in range(len(dataset.targets[0])):
        test_actuals_gru[i].clear()
        test_predictions_gru[i].clear()

    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset):
            y_hat = model_gru(snapshot.x, snapshot.edge_index, snapshot.edge_weight).squeeze()
            if y_hat.dim() > 1:
                y_hat = y_hat.flatten()
            if snapshot.y.dim() > 1:
                snapshot.y = snapshot.y.flatten()
            for i in range(len(dataset.targets[0])):
                test_predictions_gru[i].append(y_hat[i].item())
                test_actuals_gru[i].append(snapshot.y[i].item())
            cost = torch.mean((y_hat - snapshot.y)**2)
            epoch_test_loss += cost.item()
            num_test_snapshots += 1

    test_losses_gru.append(epoch_test_loss / num_test_snapshots if num_test_snapshots > 0 else 0)
    model_gru.train()

for i in range(len(dataset.targets[0])):
    total_train_actuals_gru.extend(train_actuals_gru[i])
    total_train_predictions_gru.extend(train_predictions_gru[i])
    total_test_actuals_gru.extend(test_actuals_gru[i])
    total_test_predictions_gru.extend(test_predictions_gru[i])

# Calculate train metrics for each station for GConvGRU
gru_train_metrics = {}
for i in range(len(dataset.targets[0])):
    train_mse = mean_squared_error(train_actuals_gru[i], train_predictions_gru[i])
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_actuals_gru[i], train_predictions_gru[i])
    gru_train_metrics[f"Station_{i+1}"] = (train_mse, train_rmse, train_mae)
    print(f"Station {i+1} GConvGRU - Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train MAE: {train_mae}")

# Calculate test metrics for each station for GConvGRU
gru_test_metrics = {}
for i in range(len(dataset.targets[0])):
    mse = mean_squared_error(test_actuals_gru[i], test_predictions_gru[i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actuals_gru[i], test_predictions_gru[i])
    gru_test_metrics[f"Station_{i+1}"] = (mse, rmse, mae)
    print(f"Station {i+1} GConvGRU - Test MSE: {mse}, Test RMSE: {rmse}, Test MAE: {mae}")

# Calculate overall train metrics
overall_train_mse = mean_squared_error(total_train_actuals_gru, total_train_predictions_gru)
overall_train_rmse = np.sqrt(overall_train_mse)
overall_train_mae = mean_absolute_error(total_train_actuals_gru, total_train_predictions_gru)
print(f"Overall GConvGRU - Train MSE: {overall_train_mse}, Train RMSE: {overall_train_rmse}, Train MAE: {overall_train_mae}")

# Calculate overall test metrics
overall_test_mse = mean_squared_error(total_test_actuals_gru, total_test_predictions_gru)
overall_test_rmse = np.sqrt(overall_test_mse)
overall_test_mae = mean_absolute_error(total_test_actuals_gru, total_test_predictions_gru)
print(f"Overall GConvGRU - Test MSE: {overall_test_mse}, Test RMSE: {overall_test_rmse}, Test MAE: {overall_test_mae}")

##################GConvLSTM########################

# Define the RecurrentGCN model
class RecurrentGCN_LSTM(nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN_LSTM, self).__init__()
        self.recurrent1 = GConvLSTM(node_features, filters, 2)
        self.dropout = nn.Dropout(p=0.3)  # Adjust Dropout here!
        self.linear = nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h, _ = self.recurrent1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear(h)
        return h

# Initialize the model and optimizer
model_lstm = RecurrentGCN_LSTM(node_features=4, filters=32)
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.01)

# Prepare to store loss data for plotting
train_losses_lstm = []
test_losses_lstm = []

# Initialize lists to store metrics for each station
train_actuals_lstm = {i: [] for i in range(len(dataset.targets[0]))}
train_predictions_lstm = {i: [] for i in range(len(dataset.targets[0]))}
test_actuals_lstm = {i: [] for i in range(len(dataset.targets[0]))}
test_predictions_lstm = {i: [] for i in range(len(dataset.targets[0]))}

# Initialize lists to store overall metrics
total_train_actuals_lstm = []
total_train_predictions_lstm = []
total_test_actuals_lstm = []
total_test_predictions_lstm = []

# Train the model
for epoch in tqdm(range(30)):
    epoch_train_loss = 0
    num_snapshots = 0  # Initialize snapshot counter

    model_lstm.train() #start training mode

    for i in range(len(dataset.targets[0])):
        train_actuals_lstm[i].clear()
        train_predictions_lstm[i].clear()

    for time, snapshot in enumerate(train_dataset):
        optimizer_lstm.zero_grad()
        y_hat = model_lstm(snapshot.x, snapshot.edge_index, snapshot.edge_weight).squeeze()
        cost = torch.mean((y_hat - snapshot.y)**2)
        cost.backward()
        optimizer_lstm.step()
        epoch_train_loss += cost.item()
        num_snapshots += 1

        for i in range(len(dataset.targets[0])):
            train_predictions_lstm[i].append(y_hat[i].item())
            train_actuals_lstm[i].append(snapshot.y[i].item())

    train_losses_lstm.append(epoch_train_loss / num_snapshots if num_snapshots > 0 else 0)

    # Evaluate the model
    model_lstm.eval()
    epoch_test_loss = 0
    num_test_snapshots = 0  # Initialize snapshot counter for testing

    for i in range(len(dataset.targets[0])):
        test_actuals_lstm[i].clear()
        test_predictions_lstm[i].clear()

    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset):
            y_hat = model_lstm(snapshot.x, snapshot.edge_index, snapshot.edge_weight).squeeze()
            if y_hat.dim() > 1:
                y_hat = y_hat.flatten()
            if snapshot.y.dim() > 1:
                snapshot.y = snapshot.y.flatten()
            for i in range(len(dataset.targets[0])):
                test_predictions_lstm[i].append(y_hat[i].item())
                test_actuals_lstm[i].append(snapshot.y[i].item())
            cost = torch.mean((y_hat - snapshot.y)**2)
            epoch_test_loss += cost.item()
            num_test_snapshots += 1

    test_losses_lstm.append(epoch_test_loss / num_test_snapshots if num_test_snapshots > 0 else 0)
    model_lstm.train()

for i in range(len(dataset.targets[0])):
    total_train_actuals_lstm.extend(train_actuals_lstm[i])
    total_train_predictions_lstm.extend(train_predictions_lstm[i])
    total_test_actuals_lstm.extend(test_actuals_lstm[i])
    total_test_predictions_lstm.extend(test_predictions_lstm[i])

# Calculate train metrics for each station for GConvLSTM
lstm_train_metrics = {}
for i in range(len(dataset.targets[0])):
    train_mse = mean_squared_error(train_actuals_lstm[i], train_predictions_lstm[i])
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_actuals_lstm[i], train_predictions_lstm[i])
    lstm_train_metrics[f"Station_{i+1}"] = (train_mse, train_rmse, train_mae)
    print(f"Station {i+1} GConvLSTM - Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train MAE: {train_mae}")

# Calculate test metrics for each station for GConvLSTM
lstm_test_metrics = {}
for i in range(len(dataset.targets[0])):
    mse = mean_squared_error(test_actuals_lstm[i], test_predictions_lstm[i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actuals_lstm[i], test_predictions_lstm[i])
    lstm_test_metrics[f"Station_{i+1}"] = (mse, rmse, mae)
    print(f"Station {i+1} GConvLSTM - Test MSE: {mse}, Test RMSE: {rmse}, Test MAE: {mae}")

# Calculate overall train metrics
overall_train_mse = mean_squared_error(total_train_actuals_lstm, total_train_predictions_lstm)
overall_train_rmse = np.sqrt(overall_train_mse)
overall_train_mae = mean_absolute_error(total_train_actuals_lstm, total_train_predictions_lstm)
print(f"Overall GConvLSTM - Train MSE: {overall_train_mse}, Train RMSE: {overall_train_rmse}, Train MAE: {overall_train_mae}")

# Calculate overall test metrics
overall_test_mse = mean_squared_error(total_test_actuals_lstm, total_test_predictions_lstm)
overall_test_rmse = np.sqrt(overall_test_mse)
overall_test_mae = mean_absolute_error(total_test_actuals_lstm, total_test_predictions_lstm)
print(f"Overall GConvLSTM - Test MSE: {overall_test_mse}, Test RMSE: {overall_test_rmse}, Test MAE: {overall_test_mae}")

################### Plotting of selected stations ###################

selected_stations = [0, 2, 14]
num_historical_data = 10

# Sicherstellen, dass die Vorhersagen und tatsächlichen Werte korrekt abgerufen und geplottet werden.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharex=False, sharey=True)
fig.suptitle('Comparison of Actual Targets and Predictions for Selected Stations')

for i, station_index in enumerate(selected_stations):
    historical_targets = [target[station_index] for target in dataset.targets[-(num_test_steps + num_historical_data):-num_test_steps]]
    test_targets = [target[station_index] for target in dataset.targets[-num_test_steps:]]

    # Anzahl der Trainingsteps vor historischen
    num_train_steps = num_train_timesteps - num_test_steps - num_historical_data
    train_steps = list(range(num_train_steps))

    # Sicherstellen, dass die Vorhersagen korrekt abgerufen werden
    arima_preds = arima_test_predictions.get(f"Station_{station_index + 1}", [None] * num_test_steps)
    gconvgru_preds = test_predictions_gru[station_index][-num_test_steps:]
    gconvlstm_preds = test_predictions_lstm[station_index][-num_test_steps:]

    historical_steps = list(range(num_train_steps, num_train_steps + len(historical_targets)))
    test_steps = list(range(num_train_steps + len(historical_targets), num_train_steps + len(historical_targets) + len(test_targets)))

    ax = axes[i]
    ax.plot(train_steps + historical_steps + test_steps,
            [None]*num_train_steps + historical_targets + test_targets, label='Actual Values', marker='o')
    ax.plot(test_steps, arima_preds, label='ARIMA Predictions', marker='x', linestyle="--")
    ax.plot(test_steps, gconvgru_preds, label='GConvGRU Predictions', marker='*', linestyle="--")
    ax.plot(test_steps, gconvlstm_preds, label='GConvLSTM Predictions', marker='^', linestyle="--")

    # Set the title to the current station
    ax.set_title(f'Region {station_index + 1}')

    # Set x-axis ticks in increments of 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Add legend to each subplot
    ax.legend()

    # Set labels
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Number of Requests')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

###################### Plotting of Learning Curves ######################

# GConvGRU Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses_gru, label='Train Loss GConvGRU', marker='o')
plt.plot(test_losses_gru, label='Test Loss GConvGRU', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('GConvGRU Learning Curve')
plt.legend()
plt.show()

# GConvLSTM Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses_lstm, label='Train Loss GConvLSTM', marker='o')
plt.plot(test_losses_lstm, label='Test Loss GConvLSTM', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('GConvLSTM Learning Curve')
plt.legend()
plt.show()

###################### Training Metrics Summary ######################

print("\n================== Metrics Summary for selected Regions ==================\n")

# Ausgewählte Stationen (in Null-basierter Indexierung)
selected_stations = [0, 2, 14]  # Indices für Stationen 1, 9 und 15

# Funktion zum Formatieren des Stationsnamens
def format_station_name(index):
    return f"Station_{index + 1}"

# ARIMA Training and Testing Metrics
print("ARIMA Metrics:")
for station_index in selected_stations:
    station = format_station_name(station_index)
    train_mse, train_rmse, train_mae = arima_train_metrics[station]
    test_mse, test_rmse, test_mae = arima_test_metrics[station]
    print(f"{station}:")
    print(f"  Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    print(f"  Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

print("\nGConvGRU Metrics:")
for station_index in selected_stations:
    station = format_station_name(station_index)
    train_mse, train_rmse, train_mae = gru_train_metrics[station]
    test_mse, test_rmse, test_mae = gru_test_metrics[station]
    print(f"{station}:")
    print(f"  Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    print(f"  Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

print("\nGConvLSTM Metrics:")
for station_index in selected_stations:
    station = format_station_name(station_index)
    train_mse, train_rmse, train_mae = lstm_train_metrics[station]
    test_mse, test_rmse, test_mae = lstm_test_metrics[station]
    print(f"{station}:")
    print(f"  Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    print(f"  Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

print("\n=============================================================\n")

