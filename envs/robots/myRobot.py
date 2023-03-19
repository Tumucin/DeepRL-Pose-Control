import numpy as np
from gym import spaces
from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
import math
from panda_gym.envs.core import Task
import time
import gym.utils.seeding
from yaml.loader import SafeLoader
import yaml
import PyKDL
from ..utils.kinematics import KINEMATICS

    
class MYROBOT(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        control_type: str = "joints",
        config: dict = None
    ) -> None:
        self.config = config
        self.kinematic = KINEMATICS(self.config['urdfPath'])
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        self.wdlsAction = np.zeros(7)
        self.networkAction = np.zeros(7)
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
            
        self.jointLimitLow = np.array([-math.pi, 0, -2.9, -math.pi, -2.9, 0, -2.9, 0.00, 0.00])
        #self.jointLimitLow =np.array([0, math.pi/4-0.01, 0.00, 0.00, 0.00, 0, 0.00, 0.00, 0.00])
        self.jointLimitHigh = np.array([math.pi, math.pi/2, 2.9, 0.00, 2.9, 3.8, 2.9, 0.00,0.00])
        #self.jointLimitHigh = np.array([0, math.pi/4+0.01, 0.00, 0.00, 0.00, 0, 0.00, 0.00, 0.00])
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        #self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        #self.neutral_joint_values = np.array([0.00, math.pi/2, 0.00, -0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def calActionWDLS(self, obs):
        #time.sleep(0.5)
        wdlsAction = np.zeros(7)
        deltax = obs['desired_goal'] - obs['achieved_goal']

        q_in = PyKDL.JntArray(self.kinematic.numbOfJoints)
        q_in[0], q_in[1], q_in[2], q_in[3] = obs['observation'][0], obs['observation'][1], obs['observation'][2], obs['observation'][3]
        q_in[4], q_in[5], q_in[6] = obs['observation'][4], obs['observation'][5], obs['observation'][6]

        v_in = PyKDL.Twist(PyKDL.Vector(deltax[0],deltax[1],deltax[2]), PyKDL.Vector(0.0,0.00,0.00))
        v_norm = np.linalg.norm(deltax)
        if v_norm > 0.5:
            v_in/=v_norm*2

        ## v_in = v_in*threshold/norm(vi) 

        ## TODO v_in normalize et 
        q_dot_out = PyKDL.JntArray(self.kinematic.numbOfJoints)
        self.kinematic.ikVelKDL.CartToJnt(q_in, v_in, q_dot_out)

        j_kdl = PyKDL.Jacobian(self.kinematic.numbOfJoints)
        self.kinematic.jacSolver.JntToJac(q_in, j_kdl)
        #print("j_kdl:", j_kdl)
        for i in range(7):
            wdlsAction[i] = q_dot_out[i]

        return wdlsAction

    def set_action(self, action: np.ndarray, obs) -> None:
        action = action.copy()  # ensure action don't change
        ## TODO network output 10a böl
        #action = action/5
        #print("action:", action)
        if self.config['pseudoI']==True:
            action = self.calActionWDLS(obs) + action

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        #ee_position = np.array(self.get_ee_position())
        #ee_velocity = np.array(self.get_ee_velocity())
        currentJointAngles = [self.get_joint_angle(joint=i) for i in range(7) ]
        currentJoinVelocities = [self.get_joint_velocity(joint=i) for i in range(7) ]
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((currentJointAngles, currentJoinVelocities, [fingers_width]))
        else:
            obs = np.concatenate((currentJointAngles, currentJoinVelocities))

        #print("obs:", obs)
        return obs

    def reset(self) -> None:
        #self.goal = self._sample_goal()
        #self.sim.set_base_pose("target", np.array([0.5,0.5,0.5]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        if self.config['randomStart']==True:
            seed=None
            np_random, seed = gym.utils.seeding.np_random(seed)
            sampledAngles = np_random.uniform(self.jointLimitLow, self.jointLimitHigh)
            self.set_joint_angles(sampledAngles)
        else:
            self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
