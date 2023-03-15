from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.myRobot import MYROBOT
import numpy as np
sim = PyBullet(render=True)
robot = MyRobot(sim)

for _ in range(50):
    robot.set_action(np.array([1.0]))
    sim.step()