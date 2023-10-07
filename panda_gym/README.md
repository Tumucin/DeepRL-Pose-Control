# Self-Collision Aware Reaching and Pose Control in Large Workspaces using Reinforcement Learning

## Table of Contents: 
- [About](#about)
- [Citation](#citation)
- [Installation](#installation)
- [Tutorials](#tutorials) 
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
- GNavigate to panda_gym directory and create the conda environment
```setup
conda env create environment.yml
conda activate DeepRL-Pose-Control
```
- Navigate to the parent directory of this project. Run the installation script to install the required packages and pre-trained models.
```setup
bash install.sh
```
## Tutorials
