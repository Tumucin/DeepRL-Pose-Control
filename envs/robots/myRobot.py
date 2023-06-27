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
from stable_baselines3.common.utils import safe_mean
import pybullet as p
from pyquaternion import Quaternion

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
        config: dict=None
    ) -> None:
        self.config = config
        self.model = None
        self.kinematic = KINEMATICS(self.config['urdfPath'], config['baseLinkName'], config['eeLinkName'])
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else self.kinematic.numbOfJoints  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        self.wdlsAction = np.zeros(7)
        self.networkAction = np.zeros(7)
        self.currentWSNumber = 1
        self.goalFrame = None
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.quaternionAngleError = 0.00
        self.quaternionDistanceError = 0.00
        self.quaternionError = Quaternion(1, 0, 0, 0)
        self.finalAction = np.zeros(7)
        self.np_random_start, _ = gym.utils.seeding.np_random()
        self.pseudoAction = np.zeros(self.kinematic.numbOfJoints)
        self.currentSampledAnglesStart = None
        if self.config['CurriLearning'] == True:
            self.datasetFileName = self.config['datasetPath'] + "/" + self.config['body_name'] + "_" + self.config['curriculumFirstWorkspaceId']+".csv"
        else:
            self.datasetFileName = self.config['datasetPath'] + "/" + self.config['body_name'] + "_" + self.config['finalWorkspaceID']+".csv"
        self.dataset = np.genfromtxt(self.datasetFileName, delimiter=',', skip_header=1)
        #self.q_in = PyKDL.JntArray(self.kinematic.numbOfJoints)
        self.j_kdl = PyKDL.Jacobian(self.kinematic.numbOfJoints)
        if self.config['body_name'] == 'j2n6s300':
            joint_indices = np.array([0, 1, 2, 3, 4, 5])
            joint_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0])
        else:
            joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
            joint_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
        super().__init__(
            sim,
            body_name=config['body_name'],
            file_name=config['file_name'],
            #body_name="j2s7s300",
            #file_name="/home/tumu/anaconda3/envs/stableBaselines/panda-gym/panda_gym/envs/robots/jaco_robotiq.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=joint_indices,
            joint_forces=joint_forces,
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array(self.config['neutral_joint_values'])
        self.ee_link = self.config['ee_link']
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    

    def set_action(self, action: np.ndarray, obs) -> None:
        action = action.copy()  # ensure action don't change
        self.calculateRPYErrorWithQuaternion()
        #action = 0*action
        if self.config['pseudoI']==True and self.config['networkOutput']==True:
            if self.config['addOrientation'] == True:
                self.pseudoAction = self.calculateqdotFullJac(obs)
                action =  self.pseudoAction+ action
            else:
                self.pseudoAction = self.calculateqdotOnlyPosition(obs)
                action =  self.pseudoAction+ action

        elif self.config['pseudoI']==True and self.config['networkOutput']==False:
            if self.config['addOrientation'] == True:
                self.pseudoAction = self.calculateqdotFullJac(obs)
                action = self.pseudoAction
            else:
                self.pseudoAction = self.calculateqdotOnlyPosition(obs)
                action = self.pseudoAction
        else:
            action = action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.finalAction = action
        if self.control_type == "ee":
            ee_displacement = self.finalAction[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = self.finalAction[:self.kinematic.numbOfJoints]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = self.finalAction[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        
        if self.config['body_name'] == 'j2n6s300':
            target_angles = target_angles[0:6]
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
        arm_joint_ctrl = arm_joint_ctrl * 0.01  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(self.kinematic.numbOfJoints)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        currentJointAngles = [self.get_joint_angle(joint=i) for i in range(self.kinematic.numbOfJoints) ]
        currentJoinVelocities = [self.get_joint_velocity(joint=i) for i in range(self.kinematic.numbOfJoints) ]
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((currentJointAngles, currentJoinVelocities, [fingers_width]))
        else:
            if self.config['addOrientation']==True:
                obs = np.concatenate((currentJointAngles, currentJoinVelocities, self.quaternionError.elements))
            else:
                obs = np.concatenate((currentJointAngles, currentJoinVelocities, self.pseudoAction))  
        return obs

    def reset(self) -> None:
        #self.goal = self._sample_goal()
        #self.sim.set_base_pose("target", np.array([0.5,0.5,0.5]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        #print("datasetFileName in myRobot.py:", self.datasetFileName)
        if self.config['randomStart']==True:
            random_indices = self.np_random_start.choice(self.dataset.shape[0], size=1, replace=False)
            sampledAngles = self.dataset[random_indices][0]
            self.currentSampledAnglesStart = sampledAngles
            self.set_joint_angles(sampledAngles)
        else:
            self.set_joint_angles(self.neutral_joint_values)
        
        startingPose = self.get_ee_position()
        #print("startingPose in myrobot.py:", startingPose)
        
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
    
    def calculateqdotOnlyPosition(self, obs):
        """
            This function calculates the required q_dot values for a given deltax (No orientation)
            Deltax is the difference between the end effector and the desired position.
            The equation for calculating q_dot is: q_dot = J_pinv * deltax
            args:
                obs: Observation of the robot
            returns:
                qdot: Desired joint velocities
        """
        q_in = PyKDL.JntArray(self.kinematic.numbOfJoints)

        # Compute the current Jacobian for a given current joint angles
        # And convert the calculated Jacobian to numpy array for matrix multiplication
        for i in range(self.kinematic.numbOfJoints):
            q_in[i] = obs['observation'][i]
        self.kinematic.jacSolver.JntToJac(q_in, self.j_kdl)
        #print("q_in in myRobot.py:",q_in)
        # Take the first three rows of the Jacobian because we are not interested in Orientation
        J = np.zeros((3, self.kinematic.numbOfJoints))
        for i in range(3):
            for j in range(self.kinematic.numbOfJoints):
                J[i,j] = self.j_kdl[i,j]
        
        # Compute the Pseudoinverse of the Jacobian
        J_pinv = np.linalg.pinv(J)
        
        # Calculate the desired q_dot for a given v_in
        v_in = obs['desired_goal'] - obs['achieved_goal']
        if np.linalg.norm(v_in) >0.5:
            v_in/=3
        qdot = np.dot(J_pinv, v_in)
        return qdot
    
    def calculateqdotFullJac(self, obs):
        """
            This function calculates the required q_dot values for a given velocity difference
            v_in is the difference(position+orientation) between the end effector and the desired position.
            args:
                obs: Observation of the robot
            returns:
                qdot: Desired joint velocities
        """
        q_in = PyKDL.JntArray(self.kinematic.numbOfJoints)
        
        qdot = np.zeros(self.kinematic.numbOfJoints)
        
        for i in range(self.kinematic.numbOfJoints):
            q_in[i] = obs['observation'][i]
        #print("q_in:", q_in)
        #print("desired pos:", obs['desired_goal'])
        #print("achieved_goal:", obs['achieved_goal'])
        # Calculate the error in position and orientation seperately and create v_in Twist vector
        # based on the errors
        positionError = obs['desired_goal'] - obs['achieved_goal']
        errorRPYFromQuaternion = self.calculateRPYErrorWithQuaternion()

        v_in = PyKDL.Twist(PyKDL.Vector(positionError[0],positionError[1],positionError[2]), 
                           PyKDL.Vector(errorRPYFromQuaternion[0],errorRPYFromQuaternion[1],errorRPYFromQuaternion[2]))
        q_dot_out = PyKDL.JntArray(self.kinematic.numbOfJoints)
        self.kinematic.ikVelKDL.CartToJnt(q_in, v_in, q_dot_out)

        for i in range(self.kinematic.numbOfJoints):
            qdot[i] = q_dot_out[i]

        return qdot
    
    def calculateRPYErrorWithRotationMatrix(self):
        """
            This function calculates the error in orientation using Rotation Matrix. The formula is given as:
            orientationError = R_d * inverse(R_c).
            It is the error between the desired and current rotation matrix
        """
        # Create Identity Rotation Matrix
        currentRotationMatrixPYKDL = PyKDL.Rotation.Identity()

        # Get the orientation of the End Effector as quaternion info
        currentQuaternion = self.sim.get_link_orientation(self.body_name, self.ee_link)

        # Convert the current orientation to current Rotation Matrix
        currentRotationMatrix = np.array(p.getMatrixFromQuaternion(currentQuaternion)).reshape(3, 3)
        for row in range(3):
            for column in range(3):
                currentRotationMatrixPYKDL[row, column] = currentRotationMatrix[row, column]
        
        # Get the desired orientation using Rotation Matrix
        desiredRotationMatrix = self.goalFrame.M

        # Calculate the orientation error in terms of Rotation Matrix
        rotationError = desiredRotationMatrix*currentRotationMatrixPYKDL.Inverse()

        # Calculate the orientation error in terms of Roll-Pitch-Yaw
        errorRPYFromRotationMatrix = np.asarray(rotationError.GetRPY())

        return errorRPYFromRotationMatrix
        
    def calculateRPYErrorWithQuaternion(self):
        """
            This function calculates the error in orientation using quaternions. The formula is given as:
            orientationError = quat_d * inverse(quat_c)
            It is the error between the desired and current rotation matrix
        """
        # Get the desired and current orientation as "Quaternion"
        # Note that the order of the values change 
        d1 = self.goalFrame.M.GetQuaternion()
        c1 = self.sim.get_link_orientation(self.body_name, self.ee_link)
        desiredQuaternion = Quaternion(d1[3], d1[0], d1[1], d1[2])
        currentQuaternion = Quaternion(c1[3], c1[0], c1[1], c1[2])
        #print("current Quaternion:", currentQuaternion)
        #print("desired guaternion:", desiredQuaternion)
        # Calculate quaternion error 
        # Note that the order of the values change
        self.quaternionError = desiredQuaternion * currentQuaternion.conjugate
        q = PyKDL.Rotation.Quaternion(self.quaternionError[1], self.quaternionError[2], 
                                      self.quaternionError[3], self.quaternionError[0])

        # Get the orientation error in Roll-Pitch-Yaw
        errorRPYFromQuaternion = q.GetRPY()
        self.quaternionAngleError = self.quaternionError.angle
        self.quaternionDistanceError = Quaternion.distance(desiredQuaternion, currentQuaternion)
        return errorRPYFromQuaternion
