import sys
import PyKDL
#import kdl_parser.kdl_parser_py.kdl_parser_py.urdf as URDF
#import urdf as URDF
import kdl_parser_py.urdf as URDF
import numpy as np
import math

class KINEMATICS():
    def __init__(self, urdfFileName):
        self.urdfFileName = urdfFileName
        (_,self.kdlTree) = URDF.treeFromFile(self.urdfFileName)
        self.armChain = self.kdlTree.getChain('panda_link0', 'panda_grasptarget')
        self.numbOfJoints = self.armChain.getNrOfJoints()
        self.numbOfSegments = self.armChain.getNrOfSegments()

        ### Solvers
        self.fkPoseKDL = PyKDL.ChainFkSolverPos_recursive(self.armChain)
        self.fkVelKDL = PyKDL.ChainFkSolverVel_recursive(self.armChain)

        self.ikVelKDL = PyKDL.ChainIkSolverVel_wdls(self.armChain)
        self.ikPoseKDL = PyKDL.ChainIkSolverPos_NR(self.armChain,self.fkPoseKDL,self.ikVelKDL)

        self.jacSolver = PyKDL.ChainJntToJacSolver(self.armChain)

    def forwardKinematicsPoseSolv(self, q_in=None):
        endFrame = PyKDL.Frame()

        if q_in==None:
            q_in = PyKDL.JntArray(self.numbOfJoints)
        
        self.fkPoseKDL.JntToCart(q_in, endFrame)

        return endFrame

    def inverseKinematicsPoseSolv(self, endFrame, q_in=None):
        
        q_out = PyKDL.JntArray(self.numbOfJoints)
        if q_in==None:
            q_in = PyKDL.JntArray(self.numbOfJoints)

        self.ikPoseKDL.CartToJnt(q_in, endFrame, q_out)

        return q_out

def printKDLChain(armChain):
    for idx in range(armChain.getNrOfSegments()):
            print('* ' + armChain.getSegment(idx).getName())

if __name__ == '__main__':
    kinematics = KINEMATICS('panda.urdf')
    print(kinematics.numbOfJoints)
    q_in = PyKDL.JntArray(kinematics.numbOfJoints)
    q_in[0], q_in[1], q_in[2], q_in[3], q_in[4], q_in[5], q_in[6] = 0.3, 0.5, 0.3, -0.2, 0.00, 0.00, 0.00
    print("Input to the FK: q_in", q_in)
    endFrame = kinematics.forwardKinematicsPoseSolv(q_in)
    print("Result of FK pose:endFrame",endFrame)

    v_in = PyKDL.Twist(PyKDL.Vector(0.00,0.00,0.00), PyKDL.Vector(0.1,0.1,0.1))
    #q_in = PyKDL.JntArray(kinematics.numbOfJoints)
    q_dot_out = PyKDL.JntArray(kinematics.numbOfJoints)
    kinematics.ikVelKDL.CartToJnt(q_in, v_in, q_dot_out)
    print("q_dot_out:", q_dot_out)
    """

    #endFrame = PyKDL.Frame(PyKDL.Rotation.RPY(0,0,0), PyKDL.Vector(0.3106,0.0056,0.5864))
    q_out = kinematics.inverseKinematicsPoseSolv(endFrame, q_in)
    print("Results of IK: q_out", q_out)

    print("Input to the FK: q_in", q_out)
    endFrame = kinematics.forwardKinematicsPoseSolv(q_out)
    print("Result of FK pose:endFrame",endFrame)
    """