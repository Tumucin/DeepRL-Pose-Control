from typing import Any, Dict, Union
import numpy as np
import math
from panda_gym.envs.core import Task
from panda_gym.utils import distance
from yaml.loader import SafeLoader
import yaml
from stable_baselines3.common.utils import safe_mean
import os
import PyKDL
from ..utils.kinematics import KINEMATICS
import gym.utils.seeding


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
        self.kinematics = KINEMATICS(self.config['urdfPath'], config['baseLinkName'], config['eeLinkName'])
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goalFrame = None
        self.quaternionAngleError = 0.0
        self.quaternionDistanceError = 0.00
        self.lambdaErr = self.config['lambdaErr']
        self.accelerationConstant = self.config['accelerationConstant']
        self.velocityConst = self.config['velocityConstant']
        self.velocityNormThreshold = self.config['velocityNormThreshold']
        self.thresholdConstant = self.config['thresholdConstant']
        self.alpha = self.config['alpha']
        self.orientationConstant = self.config['orientationConstant']
        self.np_random_reach, _ = gym.utils.seeding.np_random()
        self.jointLimitLow = np.array(self.config['jointLimitLow'])
        self.jointLimitHigh = np.array(self.config['jointLimitHigh'])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.previousJointVelocities = 0

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1)
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
        goal = self.np_random_reach.uniform(goal_range_low, goal_range_high)
        if self.config['sampleJointAnglesGoal']==True:
            sampledAngles = self.np_random_reach.uniform(self.jointLimitLow, self.jointLimitHigh)
            q_in = PyKDL.JntArray(self.kinematics.numbOfJoints)
            for i in range(self.kinematics.numbOfJoints):
                q_in[i] = sampledAngles[i]
            goalFrame = self.kinematics.forwardKinematicsPoseSolv(q_in)
            goalFrame.p[0] = goalFrame.p[0] #+0.6
            goal[0], goal[1], goal[2] = goalFrame.p[0], goalFrame.p[1], goalFrame.p[2]
        self.goalFrame = goalFrame
        calculatedRadius = np.sqrt(self.goalFrame.p[0]**2 + self.goalFrame.p[1]**2)
        if not (calculatedRadius < 0.20 and self.goalFrame.p[2] < 0.5):
            pass
        else:
            print("here")
            self._sample_goal()
            
        
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self,achieved_goal,desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:        
        d = distance(achieved_goal, desired_goal)
        currentJointVelocities = np.array([self.sim.get_joint_velocity(self.sim.body_name,joint=i) for i in range(7)])
        currentJointAccelerations = (currentJointVelocities - self.previousJointVelocities)/(self.sim.timestep)
        self.previousJointVelocities = currentJointVelocities
        currentJointVelocitiesNorm = np.linalg.norm(currentJointVelocities)
        self.sim.drawInfosOnScreen(d, currentJointVelocitiesNorm, self.quaternionAngleError)
        
        if self.reward_type == "sparse":
            return np.exp(-(self.lambdaErr)*(d*d)) - self.accelerationConstant*currentJointVelocitiesNorm
        else:
            if self.config['addOrientation'] == True:

                return np.exp(-(self.lambdaErr)*(d*d)) - self.accelerationConstant*np.linalg.norm(currentJointAccelerations) - (self.velocityConst*currentJointVelocitiesNorm)/(1+self.alpha*d)+ \
                    self.thresholdConstant*np.array(d < self.distance_threshold, dtype=np.float64)*np.array(currentJointVelocitiesNorm < self.velocityNormThreshold, dtype=np.float64)+\
                    np.exp(-(self.orientationConstant)*(self.quaternionAngleError**2))     
            else:
                return np.exp(-(self.lambdaErr)*(d*d)) - self.accelerationConstant*np.linalg.norm(currentJointAccelerations) - (self.velocityConst*currentJointVelocitiesNorm)/(1+self.alpha*d)+ \
                    self.thresholdConstant*np.array(d < self.distance_threshold, dtype=np.float64)*np.array(currentJointVelocitiesNorm < self.velocityNormThreshold, dtype=np.float64)