# Self-Collision Aware Reaching and Pose Control in Large Workspaces using Reinforcement Learning

## Table of Contents: 
- [About](#about)
- [Citation](#citation)
- [Installation](#installation)
- [Usage](#usage)
## About
- This repository contains the official implementation of the research paper titled **"Self-Collision Aware Reaching and Pose Control in Large Workspaces using Reinforcement Learning"**. You can find the paper [here](https://github.com/Tumucin/DeepRL-Pose-Control).
- The codebase is developed and tested using Python, along with the following libraries:
  - [stable_baselines3](https://github.com/DLR-RM/stable-baselines3)
  - [PyBullet](https://github.com/bulletphysics/bullet3)
  - [panda-gym](https://github.com/qgallouedec/panda-gym)

The development and testing were performed on an Ubuntu 18.04 system.
## Citation
If you find this study useful, please cite using the following citation:
```bibtex
 @article{...,
   title = {Self-Collision Aware Reaching and Pose Control in Large Workspaces using Reinforcement Learning},
   author = {$author$},
   year = {2023},
   booktitle = {$booktitle$},
   url = {https://github.com/Tumucin/DeepRL-Pose-Control},
   pdf = {$pdf$}
 }
```
## Installation
- Clone this repository to your local machine
```setup
git clone https://github.com/Tumucin/DeepRL-Pose-Control.git
```
- Navigate to panda_gym directory and create the conda environment
```setup
conda env create environment.yml
conda activate DeepRL-Pose-Control
```
- Navigate to the parent directory of this project. Run the installation script to install the required packages and pre-trained models.
```setup
bash install.sh
```
## Usage

To use this package, follow these steps to update the necessary directory paths in the code and configuration files:
- Modify line 27 and line 29 in the **"panda_reach.py"** file, located within the **"DeepRL-Pose-Control/panda_gym/envs/panda_tasks"** directory, to match the folder locations specific to your setup.
- Navigate to the **"DeepRL-Pose-Control/panda_gym/envs/configFiles"** directory. Within this directory,  update the paths specified in the first six lines of each file to reflect the correct directory locations for your configuration.

### Training
#### Agent1(Traditional Baseline)
Please note that **"Agent1"** serves as a traditional baseline in this context. There is no training component associated with Agent1; it functions as a baseline reference.
#### Agent Training
The training and evaluation procedures for Agents 2, 3, 4, and 5 are similar. The following command line uses Agent2 as an example. Please note that this command line performs both training and evaluation. Evaluation is conducted with 1000 random initial robot configurations and random target poses.
To train **"Agent2"**, which functions as a Learning Baseline without the Pseudo-inverse module, follow these steps:
```setup
python3 trainTest.py --mode True --expNumber 1 --configName "Agent2_Panda.yaml"
```
The trained model, log files, and information about failed samples will be saved to the directory specified in the corresponding YAML files. Metric results will be recorded and saved in the metrics$expNumber$.txt file, where expNumber corresponds to the experiment number.

### Evaluation
After completing the training procedure, you can evaluate the trained models to obtain metric results using the PyBullet simulator. The evaluation process includes using 1000 random initial robot configurations and random target poses. The results are saved to a .txt file as explained in the Training section.
```setup
python3 trainTest.py --expNumber 1 --configName "Agent2_Panda.yaml" --render True
```


