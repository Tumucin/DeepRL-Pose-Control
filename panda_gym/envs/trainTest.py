import sys
import os 
#sys.path.append('/scratch/users/tbal21/.conda/envs/stableBaselines/lib/python3.8/site-packages')
#sys.path.append('/scratch/users/tbal21/.conda/envs/stableBaselines/panda-gym')
import gym
#import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import os 
import yaml
from yaml.loader import SafeLoader
from sb3_contrib import TQC
import time
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from CustomCallback import CUSTOMCALLBACK
from pyquaternion import Quaternion
import gym.utils.seeding
import csv
import PyKDL
import random
import seaborn as sns
import matplotlib.pyplot as plt
class TRAINTEST():
    def __init__(self, config):
        self.config = config
        # The name of the checkpoint file for example PandaReach-v2PPO_3
        self.checkPointNamePrefix = self.config['envName']+self.config['algorithm']+"_"+str(self.config['expNumber'])
        self.tbParentFolderToSave = self.config['logSavePath']
        self.tbFileNameToSave = self.tbParentFolderToSave+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+str(self.config['expNumber'])
        self.modelParentFolderToSave = self.config['modelSavePath']
        self.modelFileNameToSave = self.modelParentFolderToSave+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['expNumber'])
        self.datasetFileName = self.config['datasetPath'] + "/" + self.config['body_name'] + "_" + self.config['finalWorkspaceID']+".csv"
        self.dataset = np.genfromtxt(self.datasetFileName, delimiter=',', skip_header=1)

    def train(self,algorithm,env):
        checkpoint_callback = CheckpointCallback(save_freq=self.config['save_freq'], save_path=self.config['modelSavePath'],
                                            name_prefix=self.checkPointNamePrefix)
        
        if self.config['algorithm']=="PPO":
            
            print("NORMAL LEARNING FOR PPO")
            if  self.config['activation_fn'] == 1:
                print("Activation fnc is RELU")
                activation_fn = th.nn.ReLU
            else:
                print("Activation fnc is Tanh")
                activation_fn = th.nn.Tanh
            #policy_kwargs = dict(activation_fn=activation_fn, net_arch=[dict(pi=[self.config['hiddenUnits'], self.config['hiddenUnits']], 
            #                                                                vf=[self.config['hiddenUnits'],self.config['hiddenUnits']])])
            print(type(self.config['net_arch']))
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=self.config['net_arch'])
            model = algorithm(policy=self.config['policy'], env=env, n_steps=self.config['n_steps'],n_epochs=self.config['n_epochs'],verbose=self.config['verbose'], batch_size=self.config['batch_size'], learning_rate=self.config['learning_rate'],
                            tensorboard_log=self.tbFileNameToSave,policy_kwargs=policy_kwargs)
            
            for i in range(self.config['n_envs']):
                env.envs[i].task.model = model
                env.envs[i].robot.model = model
            
        if self.config['algorithm']=="TQC":

            print("NORMAL LEARNING FOR TQC")
            model = algorithm(policy=self.config['policy'], env=env, tensorboard_log=self.tbFileNameToSave,
                            verbose=self.config['verbose'], ent_coef=self.config['ent_coef'], batch_size=self.config['batch_size'], gamma=self.config['gamma'],
                            learning_rate=self.config['learning_rate'], learning_starts=self.config['learning_starts'],replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=self.config['replay_buffer_kwargs'], policy_kwargs=self.config['policy_kwargs'])
        
        CustomCallBack = CUSTOMCALLBACK(verbose=0, config=self.config)
        start_time = time.time()
        model.learn(total_timesteps=self.config['total_timesteps'], callback=[checkpoint_callback,CustomCallBack])
        print("Total time:", time.time()-start_time)
        model.save(self.modelFileNameToSave)
        
        if self.config['algorithm']=='TQC':
            print("REPLAY BUFFER IS SAVED--NORMAL LEARNING")
            model.save_replay_buffer(self.config['bufferPath']+str(self.config['expNumber']))   

    def saveMetrics(self, env, rmse, mae, successRate1, successRate5, avgJntVel, avgQuaternionDistance, avgQuaternionAngle,
                        successRate1030, successRate1020, successRate1010, successRate530, successRate520, successRate510):
        print(env.robot.datasetFileName + " has been used by robot.py")
        print(env.task.datasetFileName + " has been used by reach.py")
        #print("RMSE:", rmse)
        #print("MAE:", mae)
        #print("Success Rate 1 cm:", successRate1)
        #print("Success Rate 5 cm:", successRate5)
        #print("Average joint velocities:", avgJntVel)
        #print("Average quaternion distance:", avgQuaternionDistance)
        #print("Average quaternion angle:", avgQuaternionAngle)
        #print("numberOfCollisionBelow5cm:", env.sim.numberOfCollisionsBelow5cm)
        #print("numberOfCollisionAbove5cm:", env.sim.numberOfCollisionsAbove5cm)
        #print("successRate 10cm 30 degrees:", successRate1030)
        #print("successRate 10cm 20 degrees:", successRate1020)
        #print("successRate 10cm 10 degrees:", successRate1010)
        #print("successRate 5cm 30 degrees:", successRate530)
        #print("successRate 5cm 20 degrees:", successRate520)
        #print("successRate 5cm 10 degrees:", successRate510)
        with open("metrics"+str(self.config['expNumber'])+".txt", 'w') as f:
            f.write('rmse:{}\nmae:{}\n1cm SR:{}\n5cm SR:{}\nMJV:{}\n{}\n{}\n# of collisions below 5cm:{}\n# of collisions above 5cm:{}\n10cm30deg SR:{}\n10cm20deg SR:{}\n10cm10deg SR:{}\n5cm30deg SR:{}\n5cm30deg SR:{}\n5cm30deg SR:{}'.format(rmse, mae, successRate1, successRate5, avgJntVel, 
                                                        avgQuaternionDistance, avgQuaternionAngle, env.sim.numberOfCollisionsBelow5cm,
                                                        env.sim.numberOfCollisionsAbove5cm,
                                                        successRate1030, successRate1020, successRate1010,
                                                        successRate530, successRate520, successRate510))
        f.close()
        print("Metrics results have been saved!!!")

    def saveFailedSamples(self, failedAngles):
        with open(self.config['failedSamplesSavePath']+"/failedStart"+str(self.config['expNumber'])+".csv", 'w', newline='') as csvfile1:
            for startAngles in failedAngles['startAngle']:
                writer = csv.writer(csvfile1)
                writer.writerow(startAngles)
        
        with open(self.config['failedSamplesSavePath']+"/failedTarget"+str(self.config['expNumber'])+".csv", 'w', newline='') as csvfile2:
            for targetAngles in failedAngles['targetAngle']:
                writer2 = csv.writer(csvfile2)
                writer2.writerow(targetAngles)
        
        print("Failed samples have been saved!!!")

    def plotFailedSamples(self, failedAngles, env):
        fig = plt.figure()
        ax1_2d = fig.add_subplot(221)
        ax2_2d = fig.add_subplot(222)
        ax3_3d = fig.add_subplot(223, projection='3d')
        ax4_3d = fig.add_subplot(224, projection='3d')
        
        xfailedPointsStart = []
        yfailedPointsStart = []
        zfailedPointsStart = []
        xfailedPointsTarget = []
        yfailedPointsTarget = []
        zfailedPointsTarget = []
        
        for startAngle in failedAngles['startAngle']:
            q_in = PyKDL.JntArray(env.task.kinematics.numbOfJoints)
            for i in range(env.task.kinematics.numbOfJoints):
                q_in[i] = startAngle[i]
            goalFrame = env.task.kinematics.forwardKinematicsPoseSolv(q_in)
            xfailedPointsStart.append(goalFrame.p[0])
            yfailedPointsStart.append(goalFrame.p[1])
            zfailedPointsStart.append(goalFrame.p[2])
        unique_colors = sns.color_palette(n_colors=len(xfailedPointsStart))
        ax1_2d.scatter(xfailedPointsStart, yfailedPointsStart,c=unique_colors)
        ax3_3d.scatter(xfailedPointsStart, yfailedPointsStart, zfailedPointsStart,c=unique_colors)

        # Add the index as text for each point
        for i, (x, y) in enumerate(zip(xfailedPointsStart, yfailedPointsStart)):
            ax1_2d.text(x, y, str(i), ha='center', va='bottom', fontsize = 7)
        
        
        for targetAngle in failedAngles['targetAngle']:
            q_in = PyKDL.JntArray(env.task.kinematics.numbOfJoints)
            for i in range(env.task.kinematics.numbOfJoints):
                q_in[i] = targetAngle[i]
            goalFrame = env.task.kinematics.forwardKinematicsPoseSolv(q_in)
            xfailedPointsTarget.append(goalFrame.p[0])
            yfailedPointsTarget.append(goalFrame.p[1])
            zfailedPointsTarget.append(goalFrame.p[2])
        
        for i, (x, y) in enumerate(zip(xfailedPointsTarget, yfailedPointsTarget)):
            ax2_2d.text(x, y, str(i), ha='center', va='bottom', fontsize = 7)

        ax2_2d.scatter(xfailedPointsTarget, yfailedPointsTarget,c=unique_colors)
        ax4_3d.scatter(xfailedPointsTarget, yfailedPointsTarget, zfailedPointsTarget,c=unique_colors)

        ax1_2d.set_xlabel('X [m]')
        ax1_2d.set_ylabel('Y [m]')
        ax1_2d.set_title('Start Points')

        ax2_2d.set_xlabel('X [m]')
        ax2_2d.set_ylabel('Y [m]')
        ax2_2d.set_title('Target Points')

        ax3_3d.set_xlabel('X [m]')
        ax3_3d.set_ylabel('Y [m]')
        ax3_3d.set_zlabel('Z [m]')
        ax3_3d.set_title('Start Points')

        ax4_3d.set_xlabel('X [m]')
        ax4_3d.set_ylabel('Y [m]')
        ax4_3d.set_zlabel('Z [m]')
        ax4_3d.set_title('Target Points')

        ax1_2d.set_xlim([-0.2,1])
        ax1_2d.set_ylim([-1.2,1.2])
        ax2_2d.set_xlim([-0.2,1])
        ax2_2d.set_ylim([-1.2,1.2])

        fig.set_size_inches(18.5, 10.5)  # Adjust the size as needed
        fig.tight_layout()

        plt.savefig(self.config['failedSamplesSavePath']+"/failedSamplesPlot"+str(self.config['expNumber'])+".png")
        print("Failed samples have been plotted!!!")

    def evaluatePolicy(self, numberOfSteps, model, env):
        mae = 0.0
        squaredError = 0.0
        successRate1 = 0.0
        successRate5 = 0.0
        avgJntVel = 0.0
        avgQuaternionDistance = 0.0
        avgQuaternionAngle = 0.0
        successRate1030 = 0
        successRate1020 = 0
        successRate1010 = 0
        successRate530 = 0
        successRate520 = 0
        successRate510 = 0
        failedAngles = {'startAngle': [], 'targetAngle': []}
        for step in range(numberOfSteps):
            print("step: ", step)
            obs = env.reset()
            done = False
            countTimeStep = 0
            episode_reward = 0.0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                countTimeStep+=1
            error = abs(obs['achieved_goal'] - obs['desired_goal'])
            mae = np.linalg.norm(error) + mae
            squaredError += np.sum(error**2)
            currentJntVel = np.array([env.task.sim.get_joint_velocity(env.task.sim.body_name,joint=i) for i in range(7)])
            #print("current vel in traintest.py:", currentJntVel)
            avgJntVel = np.linalg.norm(currentJntVel) + avgJntVel
            d1 = env.robot.goalFrame.M.GetQuaternion()
            c1 = env.sim.get_link_orientation(env.sim.body_name, self.config['ee_link'])
            desiredQuaternion = Quaternion(d1[3], d1[0], d1[1], d1[2])
            currentQuaternion = Quaternion(c1[3], c1[0], c1[1], c1[2])
            avgQuaternionDistance+=Quaternion.distance(desiredQuaternion, currentQuaternion)
            quaternionError = desiredQuaternion*currentQuaternion.conjugate
            avgQuaternionAngle+=quaternionError.angle
            print("last timestep:", countTimeStep)
            if np.linalg.norm(error) <=0.01:
                successRate1+=1
            
            if np.linalg.norm(error) <=0.05:
                successRate5+=1
            if np.linalg.norm(error) >0.05 or countTimeStep < 1000:
                #print(env.robot.currentSampledAnglesStart)
                failedAngles['startAngle'].append(env.robot.currentSampledAnglesStart)
                failedAngles['targetAngle'].append(env.task.currentSampledAnglesReach)

            if np.linalg.norm(error) <=0.1:
                if abs(quaternionError.angle) <= 0.523:
                    successRate1030+=1 
                if abs(quaternionError.angle) <= 0.349:
                    successRate1020+=1 
                if abs(quaternionError.angle) <= 0.174:
                    successRate1010+=1  
            if np.linalg.norm(error) <=0.05:
                if abs(quaternionError.angle) <= 0.523:
                    successRate530+=1 
                if abs(quaternionError.angle) <= 0.349:
                    successRate520+=1 
                if abs(quaternionError.angle) <= 0.174:
                    successRate510+=1       

                #episode_reward+=reward
            #print("numberOfCollisionsbelow5cm:", env.sim.numberOfCollisionsBelow5cm)
            #print("numberOfCollisionsabove5cm:", env.sim.numberOfCollisionsAbove5cm)
            #print("error in traintest.py:", np.linalg.norm(error))
        
        #self.plotDesiredAndActualJntAngles(env)
        #self.plot2D3DCreatedDatasetPoints(env, "ur5_robot_W1_2D.png", "ur5_robot_W1_3D.png")
        #self.saveCreatedDataset(env, 'ur5_robot_W1.csv')
        
    #plt.show()
        rmse = np.sqrt((squaredError)/(numberOfSteps))
        mae = mae/numberOfSteps
        successRate1 = successRate1/numberOfSteps
        successRate5 = successRate5/numberOfSteps
        avgJntVel = avgJntVel/numberOfSteps
        avgQuaternionDistance=avgQuaternionDistance/numberOfSteps
        avgQuaternionAngle = avgQuaternionAngle/numberOfSteps
        successRate1030 = successRate1030/numberOfSteps
        successRate1020 = successRate1020/numberOfSteps
        successRate1010 = successRate1010/numberOfSteps
        successRate530 = successRate530/numberOfSteps
        successRate520 = successRate520/numberOfSteps
        successRate510 = successRate510/numberOfSteps
        if self.config['visualizeFailedSamples'] == False:
            self.saveMetrics(env, rmse, mae, successRate1, successRate5, avgJntVel, avgQuaternionDistance, avgQuaternionAngle,
                             successRate1030, successRate1020, successRate1010, successRate530, successRate520, successRate510)
            self.saveFailedSamples(failedAngles)
            self.plotFailedSamples(failedAngles, env)

    def saveCreatedDataset(self, env, csvFileName):

        with open(csvFileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            #print(env.sim.anglesForDatasetList)
            for angle in env.sim.anglesForDatasetList:
                writer.writerow(angle)
    def plotDesiredAndActualJntAngles(self, env):
        fig, axs = plt.subplots(env.robot.kinematic.numbOfJoints)
        for i, ax in enumerate(axs):
            ax.plot(env.robot.q_actual_list[:,i], linewidth = 3)
            ax.plot(env.robot.q_desired_list[:,i])
            ax.set_title('Joint {}'.format(i))
        plt.show()

    def plot2D3DCreatedDatasetPoints(self, env, plotFileName2d, plotFileName3d):
        plt.subplot(1,2,1)
        plt.plot(env.sim.xPointsForDataset, env.sim.yPointsForDataset, 'o')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.subplot(1,2,2)
        plt.plot(env.sim.xPointsForDataset, env.sim.zPointsForDataset, 'o')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.xlim([-1,1])
        plt.ylim([-1,1.5])
        plt.savefig(plotFileName2d)
        xPoints = env.sim.xPointsForDataset
        yPoints = env.sim.yPointsForDataset
        zPoints = env.sim.zPointsForDataset

        distances = np.sqrt(np.array(xPoints)**2 + np.array(yPoints)**2 + np.array(zPoints)**2)
        colormap = plt.cm.get_cmap('jet')
        normalize = plt.Normalize(vmin=min(distances), vmax=max(distances))
        print("minimum distance:", min(distances))
        print("max distance:", max(distances))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        zPoints_array = np.array(zPoints)
        mask = (-10<zPoints_array) & (zPoints_array< 10)
        #scatter = ax.scatter(xPoints, yPoints, zPoints, c=distances, cmap = colormap, norm = normalize)
        scatter = ax.scatter(np.array(xPoints)[mask], 
                            np.array(yPoints)[mask], 
                            np.array(zPoints)[mask], c=np.array(distances)[mask], cmap = colormap, norm = normalize)
        cbar = fig.colorbar(scatter)
        cbar.set_label('Distance')
        ax.set_xlabel('X [m]')
        #threshold = 0.2
        #ax.axes.set_xlim3d(left=-threshold, right=threshold)
        #ax.axes.set_ylim3d(bottom=-threshold, top=threshold)
        #ax.axes.set_zlim3d(bottom=-threshold, top=threshold)
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.savefig(plotFileName3d, dpi=300)

    def loadAndEvaluateModel(self, algorithm,env):
        
        model = algorithm.load(self.modelFileNameToSave)         

        env.task.np_random_reach, _ = gym.utils.seeding.np_random(200)
        env.robot.np_random_start, _ = gym.utils.seeding.np_random(100)

        if self.config['visualizeFailedSamples'] == True:
            env.robot.datasetFileName = self.config['failedSamplesSavePath']+"/failedStart"+str(self.config['expNumber'])+".csv"
            env.task.datasetFileName = self.config['failedSamplesSavePath']+"/failedTarget"+str(self.config['expNumber'])+".csv"
            env.robot.dataset = self.dataset = np.genfromtxt(env.robot.datasetFileName, delimiter=',')
            env.task.dataset = self.dataset = np.genfromtxt(env.task.datasetFileName, delimiter=',')
        else:
            env.robot.datasetFileName = self.datasetFileName
            env.task.datasetFileName = self.datasetFileName
            env.task.dataset = self.dataset
            env.robot.dataset = self.dataset
        self.evaluatePolicy(self.config['testSamples'], model, env)
        
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envName',type=str, help="Name of the environment")
    parser.add_argument('--expNumber',type=int, help="Experiement number")
    parser.add_argument('--curriNumber', type=int, help="If you want to use Curriculum learning,then determine the experiement number")
    parser.add_argument('--total_timesteps', type=float)
    parser.add_argument('--mode', type=bool, default=False,help="Traning:True, Testing:False")
    parser.add_argument('--render', type=bool, default=False,help="Rendering")
    parser.add_argument('--gamma', type=float, help=" Discount factor")
    parser.add_argument('--n_steps', type= int, help="The number of steps to run for each environment per update")
    parser.add_argument('--n_epochs', type= int, help="The number of epochs for training")
    parser.add_argument('--batch_size', type=int , help="Minibatch size")
    parser.add_argument('--learning_rate', type= float, help="Learning rate")
    parser.add_argument('--n_envs', type= int, help="Number of environment(s)")
    parser.add_argument('--testSamples', type= int, help="Number of samples for testing")
    parser.add_argument('--max_episode_steps', type= int, help="Episode time step")
    parser.add_argument('--verbose', type= int, help="0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages")
    parser.add_argument('--pseudoI', type= bool, help="Use pseudoinverse: True, do not use: False")
    parser.add_argument('--save_freq', type= float, help="checkpoint saving frequency")
    parser.add_argument('--sampleJointAnglesGoal', type= bool, help="Sample goal using random joint angles.If this is false, then you should determine goal_range value")
    parser.add_argument('--goal_range', type= float, help="The volume of the workspace")
    parser.add_argument('--randomStart', type= bool, help="The pose of the robot starts randomly.If this is false, then you should determine neutral joint angles.. These values are defined in myrobot.py")
    parser.add_argument('--lambdaErr', type= float, help="")
    parser.add_argument('--accelerationConstant', type= float, help="")
    parser.add_argument('--velocityConstant', type= float, help="")
    parser.add_argument('--velocityNormThreshold', type= float, help="")
    parser.add_argument('--thresholdConstant', type= float, help="")
    parser.add_argument('--alpha', type= float, help="")
    parser.add_argument('--activation_fn', type= int, help="ReLU:1, Tanh:0")
    parser.add_argument('--testSampleOnTraining', type=int, help="Number of samples to be testes while training.")
    parser.add_argument('--evalFreqOnTraining', type=int,help="Iteration freq for evaluating the metrics while training")
    parser.add_argument('--jointLimitLowStartID', type=str,help="")
    parser.add_argument('--jointLimitHighStartID', type=str,help="")
    parser.add_argument('--rmseThreshold', type=float,help="")
    parser.add_argument('--maeThreshold', type=float,help="")
    parser.add_argument('--avgJntVelThreshold', type=float,help="")
    parser.add_argument('--avgQuaternionAngleThreshold', type=float,help="")
    parser.add_argument('--orientationConstant', type=float, help="")
    parser.add_argument('--ee_link', type=int, help="")
    parser.add_argument('--body_name', type=str, help="")
    parser.add_argument('--configName', type=str, help="")
    parser.add_argument('--collisionConstant', type=float, help="")
    args = parser.parse_args()

    with open("currentConfigNumber"+".txt", 'w') as f:
            f.write('{}'.format(args.configName))
            
    with open("configFiles/"+args.configName) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    for arg in args._get_kwargs():
        print(arg)
        if not arg[1]==None:
            config[arg[0]] = arg[1]
    trainTest = TRAINTEST(config)
    if trainTest.config['algorithm']=="PPO":
        algorithm = PPO
        
    if trainTest.config['algorithm']=="TQC":
        algorithm = TQC
    
    if trainTest.config['mode'] == True:
        if trainTest.config['algorithm'] == "TQC":
            env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
            env.robot.config = config
            env.task.config = config
            #env._max_episode_steps = trainTest.config['max_episode_steps']
        else:
            env = make_vec_env(env_id=trainTest.config['envName'], n_envs=trainTest.config['n_envs'])
            for i in range(trainTest.config['n_envs']):
                env.envs[i].robot.config = config
                env.envs[i].task.config = config
                #env.envs[i]._max_episode_steps = trainTest.config['max_episode_steps']
                
        trainTest.train(algorithm, env)
        env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
        env.robot.config = config
        env.task.config = config
        #env._max_episode_steps = trainTest.config['max_episode_steps']
        time.sleep(5)
        trainTest.loadAndEvaluateModel(algorithm,env)
    else:
        print("------------------------------------------------------------")
        env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
        print("------------------------------------------------------------")
        env.robot.config = config
        env.task.config = config
        trainTest.loadAndEvaluateModel(algorithm,env)
    print("DONE!!!")
    
if __name__=='__main__':
    main()
    
