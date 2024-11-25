# TRUCK Fewshot Segmentation
This repository contains the code for the paper "Few-shot Structure-Informed Machinery Part Segmentation with Foundation Models and Graph Neural Networks".

There is the possiblity to try the code on a few images of the synthetic truck dataset(anonymized) found in ''data/test_data''.

<video controls src="img/Video.mp4" title="Title"></video>

## Installation
There is a convinient way of using Docker Devcontainers to run the code in a safe envorioment. Make sure to have installed docker(including nvidia docker) and the vscode extension 'Dev Containers' and open the devcontainer. It will install all necessary tools and scripts.

Otherwise you can just install the requirements in your local machine.

Due to submit limitations all three weights for superpoint need do be copied manually to the 'foundation_graph_segmentation/interest_point_detectors/superpoint/weights' from their repo (https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights)

## Usage
There are three trained checkpoints in the ''checkpoints'' folder. Each checkpoint features a different granularity: TRUCK, TRUCK CRANE and LOW. The checkpoints are trained on the synthetic truck dataset.

Each granularity has a config file in the ''config'' folder. The config file specifies the parameters for the training and testing. Feel free to try around :)
To run the code on the test images, run the following command. Depending on the granularity you want to test, change the config file to the corresponding one. 
```python3.10 test.py --config_file config/parameters_test_TRUCK.yaml```
```python3.10 test.py --config_file config/parameters_test_TRUCK_CRANE.yaml```
```python3.10 test.py --config_file config/parameters_test_LOW.yaml```: 

The results will be saved in the ''results'' folder. 

## Information
Due to anonymization, the dataset is not fully published. A small part of the training and more test data will be published after the review process.

