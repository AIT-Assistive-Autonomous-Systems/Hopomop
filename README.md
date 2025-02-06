[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# HOPOMOP (Hundreds Of Points Over Millions Of Pixels)

[contributors-shield]: https://img.shields.io/github/contributors/AIT-Assistive-Autonomous-Systems/Hopomop.svg?style=for-the-badge
[contributors-url]: https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AIT-Assistive-Autonomous-Systems/Hopomop.svg?style=for-the-badge
[forks-url]: https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop/network/members
[stars-shield]: https://img.shields.io/github/stars/AIT-Assistive-Autonomous-Systems/Hopomop.svg?style=for-the-badge
[stars-url]: https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop/stargazers
[issues-shield]: https://img.shields.io/github/issues/AIT-Assistive-Autonomous-Systems/Hopomop.svg?style=for-the-badge
[issues-url]: https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop/issues
[license-shield]: https://img.shields.io/github/license/AIT-Assistive-Autonomous-Systems/Hopomop.svg?style=for-the-badge
[license-url]: https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop/blob/master/LICENSE.txt

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href=""https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop">
    <img src="images/logo.png" alt="Logo">
  </a>

  <p align="center">
    <a href="https://arxiv.org/abs/2501.10080">Paper</a>
    ·
    <a href="https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop">Video</a>
    ·
    <a href="https://github.com/AIT-Assistive-Autonomous-Systems/Hopomop">Try</a>
  </p>
</div>


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

## Results
### Semi-Supervised Video Segmentation
Using Davis Dataset. Trained on First, Middle and Last Frame.
<img src="images/davis.gif"/>


## Information
Due to anonymization, the dataset is not fully published. A small part of the training and more test data will be published after the review process.



## Meet the Authors 👩‍🔬
This work was conducted at the [AIT Austrian Institute of Technology](https://www.ait.ac.at/) 🇦🇹 in the [Center for Vision, Automation & Control](https://www.ait.ac.at/en/about-the-ait/center/center-for-vision-automation-control) 🏗️.

- 🖥️ **Michael Schwingshackl** [🔗 Research Profile](https://publications.ait.ac.at/de/persons/michael.Schwingshackl)
- 🖥️ **Fabio Francisco Oberweger** [🔗 Research Profile](https://publications.ait.ac.at/de/persons/fabio.oberweger)
- 🖥️ **Markus Murschitz**  [🔗 Research Profile](https://publications.ait.ac.at/de/persons/markus.murschitz)

## Citing Hopomop (Accepted / Submission pending!)
If you use Hopomop in your research, please use the following BibTeX entry.

```
@inproceedings{Schwingshackl2025,
  author       = {Michael Schwingshackl and Fabio Oberweger and Markus Murschitz},
  title        = {Few-shot Structure-Informed Machinery Part Segmentation with Foundation Models and Graph Neural Networks},
  booktitle    = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year         = {2025},
  organization = {IEEE},
  note         = {Presented at WACV 2025},
}
```

