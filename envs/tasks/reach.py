from typing import Any, Dict, Union
import numpy as np
import math
from panda_gym.envs.core import Task
from panda_gym.utils import distance
from yaml.loader import SafeLoader
import yaml
from stable_baselines3.common.utils import safe_mean

import PyKDL
from ..utils.kinematics import KINEMATICS
class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.01,
        goal_range=0.3,
        config: dict=None,
    ) -> None:
        super().__init__(sim)
        self.model = None
        self.config=config
        self.kinematics = KINEMATICS(self.config['urdfPath'])
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goalFrame = None
        self.quaternionAngleError = 0.0
        self.quaternionDistanceError = 0.00
        """
        self.jointLimitLow = np.array([-math.pi/2, -0.09-math.pi/4, 0.00, -1.85, 0.00, 2.26, 0.79])
        self.jointLimitHigh = np.array([+math.pi/2, -0.09+math.pi/4,  0.00, -1.85, 0.00, 2.26, 0.79])
        """
        #"""
        self.jointLimitLow = np.array([-math.pi/2, 0.00, 0.00, -1.85, 0.00, 2.26, 0.79])
        self.jointLimitHigh = np.array([+math.pi/2, math.pi/2,  0.00, -1.85, 0.00, 2.26, 0.79])
        #"""

        """
        self.jointLimitLow = np.array([-math.pi/2,    0.00,       0.00,  0.00, -2.9, 0.00, -2.9])
        self.jointLimitHigh = np.array([+math.pi/2,   math.pi/2,  0.00, -math.pi, 2.9, 3.8, 2.9])
        """
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.previousJointVelocities = 0

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        #self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        #print("reset in reach.py")
        self.goal = self._sample_goal()
        goalOrientation = np.asarray(self.goalFrame.M.GetQuaternion())
        self.sim.set_base_pose('target', self.goal, goalOrientation)
        #self.sim.set_base_pose('target', self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        
    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal_range_low = np.array([-self.config['goal_range'] / 2, -self.config['goal_range'] / 2, 0])
        goal_range_high = np.array([self.config['goal_range'] / 2, self.config['goal_range'] / 2, self.config['goal_range']])
        goal = self.np_random.uniform(goal_range_low, goal_range_high)
        if self.config['sampleJointAnglesGoal']==True:
            sampledAngles = self.np_random.uniform(self.jointLimitLow, self.jointLimitHigh)
            q_in = PyKDL.JntArray(self.kinematics.numbOfJoints)
            q_in[0], q_in[1], q_in[2], q_in[3] =sampledAngles[0], sampledAngles[1], sampledAngles[2], sampledAngles[3]
            q_in[4], q_in[5], q_in[6] = sampledAngles[4], sampledAngles[5], sampledAngles[6]
            goalFrame = self.kinematics.forwardKinematicsPoseSolv(q_in)
            goalFrame.p[0] = goalFrame.p[0] #+0.6
            goal[0], goal[1], goal[2] = goalFrame.p[0], goalFrame.p[1], goalFrame.p[2]
        #goal[0], goal[1], goal[2] = 0.3, -0.3, 0.3
        self.goalFrame = goalFrame
        
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self,achieved_goal,desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        #achieved_goal = obs['achieved_goal']
        
        d = distance(achieved_goal, desired_goal)
        orientation = self.sim.get_link_orientation(self.sim.body_name, 0)

        currentJointVelocities = np.array([self.sim.get_joint_velocity(self.sim.body_name,joint=i) for i in range(7)])
        
        get_joint_angle = np.array([self.sim.get_joint_angle(self.sim.body_name,joint=i) for i in range(7)])
        currentJointAccelerations = (currentJointVelocities - self.previousJointVelocities)/(self.sim.timestep)
        self.previousJointVelocities = currentJointVelocities
        lambdaErr = self.config['lambdaErr']
        accelerationConstant = self.config['accelerationConstant']
        velocityConst = self.config['velocityConstant']
        velocityNormThreshold = self.config['velocityNormThreshold']
        thresholdConstant = self.config['thresholdConstant']
        alpha = self.config['alpha']
        orientationConstant = self.config['orientationConstant']
        currentJointVelocitiesNorm = np.linalg.norm(currentJointVelocities)
        self.sim.drawInfosOnScreen(d, currentJointVelocitiesNorm, self.quaternionAngleError)
        #print(self.quaternionAngleError)
        if self.reward_type == "sparse":
            return np.exp(-(lambdaErr)*(d*d)) - accelerationConstant*currentJointVelocitiesNorm
        else:
            return np.exp(-(lambdaErr)*(d*d)) - accelerationConstant*np.linalg.norm(currentJointAccelerations) - (velocityConst*currentJointVelocitiesNorm)/(1+alpha*d)+ \
                   thresholdConstant*np.array(d < self.distance_threshold, dtype=np.float64)*np.array(currentJointVelocitiesNorm < velocityNormThreshold, dtype=np.float64)+\
                   np.exp(-(orientationConstant)*(self.quaternionAngleError**2))     
