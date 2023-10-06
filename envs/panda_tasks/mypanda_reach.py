from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.robots.myRobot import MYROBOT
from panda_gym.envs.tasks.reach import Reach
from panda_gym.pybullet import PyBullet
import numpy as np
from yaml.loader import SafeLoader
import yaml
#from kinematics import KINEMATICS
#import PyKDL

class MYPandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "joints") -> None:
        control_type = "joints"
        sim = PyBullet(render=render)
        robot = MYROBOT(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)

"""
kinematics = KINEMATICS('panda.urdf')
env = MYPandaReachEnv(render=True)

episodes = 1000
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        error = obs['desired_goal'] - obs['achieved_goal']
        #print("achieved_goal:", obs['achieved_goal'])
        errorMagnitude = np.linalg.norm(error)
        q_in = PyKDL.JntArray(kinematics.numbOfJoints)
        q_in[0], q_in[1], q_in[2], q_in[3] = obs['observation'][0], obs['observation'][1], obs['observation'][2], obs['observation'][3]
        q_in[4], q_in[5], q_in[6] = obs['observation'][4], obs['observation'][5], obs['observation'][6]
        v_in = PyKDL.Twist(PyKDL.Vector(error[0],error[1],error[2]), PyKDL.Vector(0.00,0.00,0.00))
        q_dot_out = PyKDL.JntArray(kinematics.numbOfJoints)
        kinematics.ikVelKDL.CartToJnt(q_in, v_in, q_dot_out)
        
        action = np.array([0.00, 0.00, 0.00, -0.00, 0.00, 0.00, 0.00])
        action[0], action[1], action[2], action[3] = q_dot_out[0], q_dot_out[1], q_dot_out[2], q_dot_out[3] 
        action[4], action[5], action[6] = q_dot_out[4], q_dot_out[5], q_dot_out[6]
        action = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        obs, reward, done, info = env.step(action)
        #obs = env.reset()
        

    
"""
