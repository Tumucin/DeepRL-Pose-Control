# Self-Collision Aware Reaching and Pose Control in Large Workspaces using Reinforcement Learning
![](https://github.com/Tumucin/DeepRL-Pose-Control/blob/PoseControlConda/panda_gym/robots.gif)
## Table of Contents: 
- [ABOUT](#about)
- [CITATION](#citation)
- [INSTALLATION](#installation)
- [USAGE](#usage)
- [AGENTS](#agents)
- [TRAINING](#training)
  - [Agent1](#agent1)
  - [Other Agents:Case1](#otherAgentsCase1)
  - [Other Agents:Case2](#otherAgentsCase2)
  - [Other Agents:Case3](#otherAgentsCase3)
  - [Curriculum Learning](#curriculumLearning)
    - [Curriculum Learning without Considering Self-Collisions](#curriculumLearningwithoutConsideringSelf-Collisions)
    - [Curriculum Learning with Considering Self-Collisions](#curriculumLearningwithconsideringSelf-Collisions)
  - [Reward Function Design](#rewardFunctionDesign)
- [EVALUATION](#evaluation)
- [SWITCHING](#switching)
- [PRE-TRAINED MODELS](#preTrainedModels)
## ABOUT
- This repository contains the official implementation of the research paper titled **"Self-Collision Aware Reaching and Pose Control in Large Workspaces using Reinforcement Learning"**. You can find the paper [here](https://github.com/Tumucin/DeepRL-Pose-Control).
- The codebase is developed and tested using Python, along with the following libraries:
  - [stable_baselines3](https://github.com/DLR-RM/stable-baselines3)
  - [PyBullet](https://github.com/bulletphysics/bullet3)
  - [panda-gym](https://github.com/qgallouedec/panda-gym)

The development and testing were performed on an Ubuntu 18.04 system.
## CITATION
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
## INSTALLATION
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
## USAGE

To use this package, follow these steps to update the necessary directory paths in the code and configuration files:
- Modify line 27 and line 29 in the **"panda_reach.py"** file, located within the **"DeepRL-Pose-Control/panda_gym/envs/panda_tasks"** directory, to match the folder locations specific to your setup.
- Navigate to the **"DeepRL-Pose-Control/panda_gym/envs/configFiles"** directory. Within this directory,  update the paths specified in the first six lines of each file to reflect the correct directory locations for your configuration.

## AGENTS
**"Agent1:"** Direct RL training without the pseudo-inverse component and curriculum scheduling.
**"Agent2:"** Curriculum training of RL without the pseudo-inverse component. This agent is trained as the combination of **"Agent1:"** + Curriculum Learning.
**"Agent3:"** Both the pseudo-inverse and the RL output is used without curriculum scheduling.
**"Agent4:"** Both the pseudo-inverse and the RL output is used with curriculum scheduling. This agent is trained as the combination of **"Agent4:"** + Curriculum Learning.
**"Agent5:"** Raw pseudo-inverse joint velocities, treated as a traditional baseline. It is important to note that this agent is not trained since there is no learning part in it.
### TRAINING
#### Agent1
Please note that **"Agent1"** serves as a traditional baseline in this context. There is no training component associated with Agent1; it functions as a baseline reference.
#### Other Agents:Case1
This example does not account for orientation at the target pose and also does not consider self-collisions.
The training and evaluation procedures for Agents 2, 3, 4, and 5 are similar. The following command line uses Agent2 as an example. Please note that this command line performs both training and evaluation. Evaluation is conducted with 1000 random initial robot configurations and random target poses.
To train **"Agent2"**, which functions as a Learning Baseline without the Pseudo-inverse module, follow these steps:
```setup
# No Orientation, no collision
python3 trainTest.py --mode True --expNumber 1 --configName "Agent2_Panda.yaml"
```
The trained model, log files, and information about failed samples will be saved to the directory specified in the corresponding YAML files. Metric results will be recorded and saved in the metrics$expNumber$.txt file, where expNumber corresponds to the experiment number.

#### Other Agents:Case2
- For this time, set the **"addOrientation"** variable in the config yaml files to True to consider orientation at the target pose.
```setup
# Orientation, no collision
python3 trainTest.py --mode True --expNumber 1 --configName "Agent2_Panda.yaml"
```
#### Other Agents:Case3
- To incorporate both orientation and self-collision considerations during training, set the **"addOrientation"** and **"enableSelfCollision"** variables in the config yaml files to True.
```setup
# Orientation, collision
python3 trainTest.py --mode True --expNumber 1 --configName "Agent2_Panda.yaml"
```
#### Curriculum Learning
##### Curriculum Learning without Considering Self-Collisions
Agents 3 and 5 employ curriculum learning during training. We move on to the next region if the mean absolute error in position and the mean norm of the joint velocities at the final state are below thresholds if collisions are not taken into account. The following command line setups the curriculum learning:
```setup
python3 trainTest.py --mode True --expNumber 1 --configName "Agent3_Panda.yaml" --maeThreshold 0.05 --avgJntVelThreshold 0.15 --evalFreqOnTraining 3000000 --testSampleOnTraining 500
```
In this configuration, the training progress is evaluated every 3 million training steps, utilizing 500 samples for evaluation. If the mean absolute error in position is less than 5 cm and the mean norm of the joint velocities is less than 0.15 rad/s, the training process proceeds to the next region.
##### Curriculum Learning with Considering Self-Collisions
In this configuration, episodes are terminated upon collision. It's crucial to understand that the mean absolute error and mean norm of joint velocities metrics are not meaningful in this context.  Since **"maeThreshold"** and **"avgJntVelThreshold"** thresholds can not exceed 10, they are set to 10. However, these thresholds can be adjusted to other values, such as 15 or 20. This adjustment ensures the next curriculum region will be added to the current workspace. Thus, the number of training episodes is fixed when collision are taken into account. 
```setup
python3 trainTest.py --mode True --expNumber 1 --configName "Agent3_Panda.yaml" --maeThreshold 10 --avgJntVelThreshold 10 --evalFreqOnTraining 3000000
```
#### Reward Function Design
It is possible to change the parameters of the reward function. This project uses the following reward function type to train an agent:
$`r(s, a)= \exp \left(-\lambda_{pos}\|(\delta x_{})\|^2\right)  -\frac{\lambda_{velocity}\|\dot{\theta}\|}{1+\|\delta x_{}\|} - \lambda_{coll} + \exp \left(-\lambda_{ori}\beta_q^2\right)`$

To modify the reward function, you can make adjustments using the following arguments:
```setup
python3 trainTest.py --mode True --expNumber 1 --configName "Agent2_Panda.yaml" --lambdaErr 100.0 --velocityConstant 0.1 --orientationConstant 50 --collisionConstant 10.0
```
Please keep in mind that the provided code assumes that the algorithm will account for self-collisions and orientation at the target pose. As mentioned earlier, ensure to set the **"addOrientation"** and **"enableSelfCollision"** variables in the config YAML files to True accordingly.

### EVALUATION
After completing the training procedure, you can evaluate the trained models to obtain metric results using the PyBullet simulator. The evaluation process includes using 1000 random initial robot configurations and random target poses. The trained model, log files, and information about failed samples will be saved to the directory specified in the corresponding YAML files. Metric results will be recorded and saved in the metrics$expNumber$.txt file, where expNumber corresponds to the experiment number as explained in the [TRAINING](#training) section.
```setup
python3 trainTest.py --expNumber 1 --configName "Agent2_Panda.yaml" --render True
```
### SWITCHING
We switch to raw pseudo-inverse control when the Euclidean distance between the end effector and the target position is below a threshold (i.e. only utilize $`\theta_{pinv}`$). It is important to note that this is only employed during inference and not during training.
If you want to implement the switch modification, set the **"switching"** variable in the config yaml files to True. It is important to note that this is only employed during inference and not during training. Additionally, the switching mode can be executed solely within the hybrid model.

### PRE-TRAINED MODELS
No Collision             |  Collision
:-------------------------:|:-------------------------:
![noCollision](https://github.com/Tumucin/DeepRL-Pose-Control/blob/PoseControlConda/panda_gym/noCollisionPretrained.png)  |  ![Collision](https://github.com/Tumucin/DeepRL-Pose-Control/blob/PoseControlConda/panda_gym/CollisionPretrained.png)

The figure above illustrates the pre-trained models corresponding to different cases and experiment numbers. These pre-trained models have been automatically downloaded into the repo during the installation process, as described in the [INSTALLATION](#installation) section. All metric results have been documented in the paper. To evaluate the performance of the pre-trained model, simply execute the following command:
```setup
# Model 891 uses hybrid agent (Agent5) without considering self-collisions and orientation
# Make sure to set the 'addOrientation' and 'enableSelfCollision' flags to False
python3 trainTest.py --expNumber 891 --configName "Agent5_Panda.yaml" --render True
```

```setup
# Model 1082 employs the Hybrid Agent (Agent5) while taking into account both self-collisions and orientation considerations
# Make sure to set the 'addOrientation' and 'enableSelfCollision' flags to True
python3 trainTest.py --expNumber 1082 --configName "Agent5_Jaco7dof.yaml" --render True
```

