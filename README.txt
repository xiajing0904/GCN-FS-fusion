## Project Structure Overview

- `dataset_FC` folder: Contains Functional Connectome X.npy dataset for 30 samples for testing.
- `dataset_SC` folder: Contains Structural Connectome X.npy dataset for 30 samples for testing.
- `gae` folder: Contains the Models for classification, and utility scripts for data preparation, training and testing.
- `results_fusion` folder: Contains classification results.

## Data directory
- FC dataset: Located in /dataset_FC/*/X.npy
- SC dataset: Located in /dataset_SC/*/X.npy
- Label dataset: Located in /dataset_SC/*/Y.npy

## Usage
- Run train_fusion_end.py for training and testing

## Environment

pytorch                   1.12.0          py3.8_cuda11.3_cudnn8.3.2_0    pytorch
torch-cluster             1.6.0+pt112cu113          pypi_0    pypi
torch-geometric           2.3.1                    pypi_0    pypi
torch-scatter             2.1.0+pt112cu113          pypi_0    pypi
torch-sparse              0.6.16+pt112cu113          pypi_0    pypi
torch-spline-conv         1.2.1+pt112cu113          pypi_0    pypi

Other details can refer to 'conda_list.txt'
