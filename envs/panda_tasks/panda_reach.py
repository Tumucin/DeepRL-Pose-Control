import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.robots.myRobot import MYROBOT
from panda_gym.envs.tasks.reach import Reach
from panda_gym.pybullet import PyBullet
import yaml
from yaml.loader import SafeLoader

class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """
    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "joints") -> None:
        control_type = "joints"
        reward_type = "dense"
        with open('/kuacc/users/tbal21/.conda/envs/stableBaselines/panda-gym/panda_gym/envs/3.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
        #with open('/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/21.yaml') as f:
         #   config = yaml.load(f, Loader=SafeLoader)
        sim = PyBullet(render=render, config=config)
        robot = MYROBOT(sim, block_gripper=True, base_position=np.array(config['base_position']), control_type=control_type, config=config)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, goal_range=config['goal_range'],config=config)
        super().__init__(robot, task)
