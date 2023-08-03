import os
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import panda_gym.assets
import gym.utils.seeding
import PyKDL
import math

class PyBullet:
    """Convenient class to use PyBullet physics engine.

    Args:
        render (bool, optional): Enable rendering. Defaults to False.
        n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
        background_color (np.ndarray, optional): The background color as (red, green, blue).
            Defaults to np.array([223, 54, 45]).
    """

    def __init__(
        self, render: bool = False, n_substeps: int = 20, background_color: np.ndarray = np.array([223.0, 54.0, 45.0]), config=None
    ) -> None:
        self.background_color = background_color.astype(np.float64) / 255
        options = "--background_color_red={} \
                    --background_color_green={} \
                    --background_color_blue={}".format(
            *self.background_color
        )
        self.connection_mode = p.GUI if render else p.DIRECT
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 2000
        #self.timestep = 10.0/1000.0
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}
        self.xArrowGoal = None
        self.yArrowGoal = None
        self.zArrowGoal = None
        self.xArrowEE = None
        self.yArrowEE = None
        self.zArrowEE = None
        self.counter = 0
        self.positionErrorText = None
        self.velocityNormText = None
        self.timeStepText = None
        self.angleErrorText = None
        self.body_name = config['body_name']
        self.ee_link = config['ee_link']

        self.isCollision = False
        self.numberOfCollisionsBelow5cm = 0
        self.numberOfCollisionsAbove5cm = 0
        self.visualShape = None
        self.np_random_start, _ = gym.utils.seeding.np_random(100)
        self.xPointsForDataset = []
        self.yPointsForDataset = []
        self.zPointsForDataset = []
        self.anglesForDataset = []
        self.anglesForDatasetList = []
        if self.body_name == "panda":
            self.consecutive_link_pairs={(4,6):True,(9,10):True,(6,8):True, (4,8):True}
        
        if self.body_name == "j2s7s300":
            self.consecutive_link_pairs={(4,9):True,(11,15):True,(13,17):True,(3,5):True,(6,9):True, 
                                         (12,16):True, (9,13):True, (15,17):True, (15,16):True, (14,16):True, (14,17):True, (11,17):True}

        if self.body_name == "j2n6s300":
            self.consecutive_link_pairs={(14,19):True}

        if self.body_name == "ur5":
            self.consecutive_link_pairs={(1,-1):True, (2,4):True, (2,5):True}

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self) -> None:
        """Step the simulation."""
        if self.counter%100==0:
            self.drawFrameForCurrentPose()
            
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

        self.changeLinkColorOnCollision()

    def checkRandomSampleAngles(self, numberOfJoints):
        eePositionForDataset = self.get_link_position(self.body_name, self.ee_link)
        angleForDataset = [self.get_joint_angle(self.body_name, joint=i) for i in range(numberOfJoints) ]
        #print("current ee position in pybullet.py:", self.initialeePositionForDataset)
        calculatedRadius = np.sqrt(eePositionForDataset[0]**2 + eePositionForDataset[1]**2)
        theta = math.atan2(eePositionForDataset[1], eePositionForDataset[0])
        thetaDegree = math.degrees(theta)
        if (0.20<calculatedRadius < 1) and (abs(thetaDegree)<20) and (eePositionForDataset[2] > -0.1) and (eePositionForDataset[2] < 0.9) and self.isCollision==False:
            self.xPointsForDataset.append(eePositionForDataset[0])
            self.yPointsForDataset.append(eePositionForDataset[1])
            self.zPointsForDataset.append(eePositionForDataset[2])
            #print("angles for dataset:", self.anglesForDataset)
            self.anglesForDatasetList.append(angleForDataset)
            print("This random angles can be used for the dataset.")

    def changeLinkColorOnCollision(self):
        if self.visualShape == None:
            self.visualShape = p.getVisualShapeData(0)
        contactPoints =self.physics_client.getContactPoints(self._bodies_idx[self.body_name],self._bodies_idx[self.body_name])
        coloredLinksList=[]
        for contact in contactPoints:     
            if (contact[3],contact[4]) not in self.consecutive_link_pairs:
                #print("first link:", cont[3]+1)
                #print("second link:", cont[4]+1)
                coloredLinksList.append((contact[3],contact[4]))
                
                p.changeVisualShape(contact[1], contact[3], rgbaColor=[1.0, 0.0, 0.0, 1])
                p.changeVisualShape(contact[2], contact[4], rgbaColor=[1.0, 0.0, 0.0, 1])
                self.isCollision = True
        #self.checkRandomSampleAngles(6)
        #if self.isCollision == False:
        #    eePosition = self.get_link_position(self.body_name, self.ee_link)
        #    #print("eePosiiton after step:", self.initialeePositionForDataset)
        #    self.xPointsForDataset.append(self.initialeePositionForDataset[0])
        #    self.yPointsForDataset.append(self.initialeePositionForDataset[1])
        #    self.zPointsForDataset.append(self.initialeePositionForDataset[2])
        #    self.anglesForDatasetList.append(self.anglesForDataset)

        #p.changeVisualShape(0, 1, rgbaColor=[1.0, 1.0, 0.0, 1])
        #p.changeVisualShape(0, 5, rgbaColor=[1.0, 0.0, 0.0, 1])
        if len(coloredLinksList)>0:
            print("Collied pairs:", coloredLinksList)
            #time.sleep(5)
            pass

        for pairedLinks in coloredLinksList:
            p.changeVisualShape(0, pairedLinks[0], rgbaColor=[self.visualShape[0][7][0], self.visualShape[0][7][1], self.visualShape[0][7][2], self.visualShape[0][7][3]])
            p.changeVisualShape(0, pairedLinks[1], rgbaColor=[self.visualShape[0][7][0], self.visualShape[0][7][1], self.visualShape[0][7][2], self.visualShape[0][7][3]])

    def drawInfosOnScreen(self, positionError, currentJointVelocitiesNorm, angleError)-> None:
        text_pos1 = [0, 0, -0.1] # Position of the text in world coordinates
        text_pos2 = [0, 0, -0.15] # Position of the text in world coordinates
        text_pos3 = [0, 0, -0.2] # Position of the text in world coordinates
        text_pos4 = [0, 0, -0.25] # Position of the text in world coordinates
        text_color = [1, 0, -0.3] # Blue color
        text_size = 1 # Text size in pixels

        if self.counter%100 == 0:
            if self.velocityNormText!=None:
                p.removeUserDebugItem(self.timeStepText)
                p.removeUserDebugItem(self.positionErrorText)
                p.removeUserDebugItem(self.angleErrorText)
                p.removeUserDebugItem(self.velocityNormText)
            self.timeStepText = p.addUserDebugText("TIMESTEP:   "+str(self.counter), text_pos1, text_color, text_size)
            self.positionErrorText = p.addUserDebugText("POSITION ERROR [m]:   "+str(positionError), text_pos2, text_color, text_size)
            self.angleErrorText = p.addUserDebugText("ANGLE ERROR [rad]:   "+str(angleError), text_pos3, text_color, text_size)
            self.velocityNormText = p.addUserDebugText("VELOCITYNORM [rad/s]:   "+str(currentJointVelocitiesNorm), text_pos4, text_color, text_size)
            
        self.counter +=1
        if self.counter==1000:
            self.counter=0
    def close(self) -> None:
        """Close the simulation."""
        self.physics_client.disconnect()

    def render(
        self,
        mode: str = "human",
        width: int = 720,
        height: int = 480,
        target_position: np.ndarray = np.zeros(3),
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if mode == "human":
            self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING)
            time.sleep(self.dt)  # wait to seems like real speed
        if mode == "rgb_array":
            if self.connection_mode == p.DIRECT:
                warnings.warn(
                    "The use of the render method is not recommended when the environment "
                    "has not been created with render=True. The rendering will probably be weird. "
                    "Prefer making the environment with option `render=True`. For example: "
                    "`env = gym.make('PandaReach-v2', render=True)`.",
                    UserWarning,
                )
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, px, depth, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            return px

    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)

    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as quaternion (x, y, z, w).
        """
        orientation = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[1]
        return np.array(orientation)

    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        quaternion = self.get_base_orientation(body)
        if type == "euler":
            rotation = self.physics_client.getEulerFromQuaternion(quaternion)
            return np.array(rotation)
        elif type == "quaternion":
            return np.array(quaternion)
        else:
            raise ValueError("""type must be "euler" or "quaternion".""")

    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)

    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.physics_client.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The velocity.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[1]

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        self.drawFrameForRandomGoalPose(position, orientation)
        if len(orientation) == 3:
            orientation = self.physics_client.getQuaternionFromEuler(orientation)
        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )
    def drawFrameForRandomGoalPose(self, positionGoal, orientationGoal):
        # create an arrow
        if self.xArrowGoal!=None:
            p.removeUserDebugItem(self.xArrowGoal)
            p.removeUserDebugItem(self.yArrowGoal)
            p.removeUserDebugItem(self.zArrowGoal)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientationGoal)).reshape(3, 3)

        axis_length = 0.2

        # Create the lines for the frame
        x_axis_end = positionGoal + axis_length * rotation_matrix[:, 0]
        y_axis_end = positionGoal + axis_length * rotation_matrix[:, 1]
        z_axis_end = positionGoal + axis_length * rotation_matrix[:, 2]

        x_axis_color = [0, 0, 0]  # Red axis
        y_axis_color = [0, 1, 0]  # Green axis
        z_axis_color = [0, 0, 1]  # Blue axis
        self.xArrowGoal = p.addUserDebugLine(positionGoal, x_axis_end, x_axis_color, lineWidth=5)
        self.yArrowGoal = p.addUserDebugLine(positionGoal, y_axis_end, y_axis_color, lineWidth=5)
        self.zArrowGoal = p.addUserDebugLine(positionGoal, z_axis_end, z_axis_color, lineWidth=5)
    
    def drawFrameForCurrentPose(self):
        orientationGoal = self.get_link_orientation(self.body_name, self.ee_link)
        positionGoal = self.get_link_position(self.body_name, self.ee_link)
        #print("ee position:", positionGoal)
        #print("orientation 4:", self.get_link_orientation(self.body_name, 6))
        # create an arrow
        if self.xArrowEE!=None:
            p.removeUserDebugItem(self.xArrowEE)
            p.removeUserDebugItem(self.yArrowEE)
            p.removeUserDebugItem(self.zArrowEE)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientationGoal)).reshape(3, 3)

        axis_length = 0.5

        # Create the lines for the frame
        x_axis_end = positionGoal + axis_length * rotation_matrix[:, 0]
        y_axis_end = positionGoal + axis_length * rotation_matrix[:, 1]
        z_axis_end = positionGoal + axis_length * rotation_matrix[:, 2]

        x_axis_color = [0, 0, 0]  # Red axis
        y_axis_color = [0, 1, 0]  # Green axis
        z_axis_color = [0, 0, 1]  # Blue axis
        self.xArrowEE = p.addUserDebugLine(positionGoal, x_axis_end, x_axis_color, lineWidth=2)
        self.yArrowEE = p.addUserDebugLine(positionGoal, y_axis_end, y_axis_color, lineWidth=2)
        self.zArrowEE = p.addUserDebugLine(positionGoal, z_axis_end, z_axis_color, lineWidth=2)


    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        #angles = [0,0,0,0,0,0]
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)
        #print(self.get_link_position(self.body_name, self.ee_link))

        #self.createDataset(body, joints)

    def createDataset(self, body, joints):
        urdfFileName = 'ur5_robot.urdf'
        jointLimitLow = np.array([ -6.2831, -6.2831, -6.2831, -6.2831, -6.2831, -6.2831])
        jointLimitHigh = np.array([6.2831,  6.2831,  6.2831,  6.2831, 6.2831, 6.2831])
        goal = np.empty(3)
        self.anglesForDataset = self.np_random_start.uniform(jointLimitLow, jointLimitHigh)
        #self.anglesForDataset = [-1.3752308  , 0.97254075 , 4.25507331,  1.35478544 ,-1.97200273 ,-3.72854775]
        for joint, angle in zip(joints, self.anglesForDataset):
            self.set_joint_angle(body=body, joint=joint, angle=angle)
            #time.sleep(1)
        
                #print(self.anglesForDataset)
    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def inverse_kinematics(self, body: str, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        joint_state = self.physics_client.calculateInverseKinematics(
            bodyIndex=self._bodies_idx[body],
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
        )
        return np.array(joint_state)

    def place_visualizer(self, target_position: np.ndarray, distance: float, yaw: float, pitch: float) -> None:
        """Orient the camera used for rendering.

        Args:
            target (np.ndarray): Target position, as (x, y, z).
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

        links = [idx for idx in range(-1, 13)]
        #for link in links:
        #    self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], link, 0, 1)
        #    self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], link, 1, 1)
        #    self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], link, 2, 1)

        #for link in links:
        #    if link != 1:
        #        self.physics_client.setCollisionFilterPair(self._bodies_idx[body_name], self._bodies_idx[body_name], link, 1, enableCollision=False)

        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 13, 0, 0)
        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 12, 0, 0)
        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 11, 0, 0)
        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 10, 0, 0)
        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 9, 0, 0)
        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 8, 0, 0)
        #self.physics_client.setCollisionFilterGroupMask(self._bodies_idx[body_name], 7, 0, 0)

    def create_box( 
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.ones(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        if texture is not None:
            texture_path = os.path.join(panda_gym.assets.get_data_path(), texture)
            texture_uid = self.physics_client.loadTexture(texture_path)
            self.physics_client.changeVisualShape(self._bodies_idx[body_name], -1, textureUniqueId=texture_uid)

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.zeros(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.zeros(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_plane(self, z_offset: float) -> None:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.15, 0.15, 0.15, 1.0]),
        )

    def create_table(
        self,
        length: float,
        width: float,
        height: float,
        x_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offet in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, 0.0, -height / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
        )

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )
