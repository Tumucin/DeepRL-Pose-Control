import sys
sys.path.append('/scratch/users/tbal21/.conda/envs/stableBaselines/lib/python3.8/site-packages')
sys.path.append('/scratch/users/tbal21/.conda/envs/stableBaselines/panda-gym')
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
class TRAINTEST():
    def __init__(self, config):
        self.config = config
        # The name of the checkpoint file for example PandaReach-v2PPO_3
        self.checkPointNamePrefix = self.config['envName']+self.config['algorithm']+"_"+str(self.config['expNumber'])
        self.tbParentFolderToSave = self.config['logSavePath']
        self.tbFileNameToSave = self.tbParentFolderToSave+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+str(self.config['expNumber'])
        self.modelParentFolderToSave = self.config['modelSavePath']
        self.modelFileNameToSave = self.modelParentFolderToSave+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['expNumber'])

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
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=[dict(pi=[self.config['hiddenUnits'], self.config['hiddenUnits']], 
                                                                            vf=[self.config['hiddenUnits'],self.config['hiddenUnits']])])
            
            model = algorithm(policy=self.config['policy'], env=env, n_steps=self.config['n_steps'],n_epochs=self.config['n_epochs'],verbose=self.config['verbose'], batch_size=self.config['batch_size'], learning_rate=self.config['learning_rate'],
                            tensorboard_log=self.tbFileNameToSave,policy_kwargs=policy_kwargs)
            #print(model._total_timesteps)
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
        #, tb_log_name = logDir+"/"+config['envName']+"_"+config['algorithm']
        model.learn(total_timesteps=self.config['total_timesteps'], callback=[checkpoint_callback,CustomCallBack])
        #print(model._total_timesteps)
        print("Total time:", time.time()-start_time)
        model.save(self.modelFileNameToSave)
        
        if self.config['algorithm']=='TQC':
            print("REPLAY BUFFER IS SAVED--NORMAL LEARNING")
            model.save_replay_buffer(self.config['bufferPath']+str(self.config['expNumber']))   

    def evaluatePolicy(self, numberOfSteps, model, env):
        mae = 0.0
        squaredError = 0.0
        successRate1 = 0.0
        successRate5 = 0.0
        avgJntVel = 0.0

        for step in range(numberOfSteps):
            obs = env.reset()
            print("step: ", step)
            done = False
            
            episode_reward = 0.0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            error = abs(obs['achieved_goal'] - obs['desired_goal'])
            mae = np.linalg.norm(error) + mae
            squaredError += np.sum(error**2)
            avgJntVel = np.linalg.norm(action) + avgJntVel
            if np.linalg.norm(error) <=0.01:
                successRate1+=1
            
            if np.linalg.norm(error) <=0.05:
                successRate5+=1

                #episode_reward+=reward
        rmse = np.sqrt((squaredError)/(numberOfSteps))
        mae = mae/numberOfSteps
        successRate1 = successRate1/numberOfSteps
        successRate5 = successRate5/numberOfSteps
        avgJntVel = avgJntVel/numberOfSteps

        return rmse, mae, successRate1, successRate5, avgJntVel

    def loadAndEvaluateModel(self, algorithm,env):
        modelDir = self.config['modelSavePath']
        
        model = algorithm.load(self.modelFileNameToSave) 
        #env = model.get_env()
                
        env.robot.jointLimitLow = env.robot.workspacesdict[str(list(env.robot.workspacesdict)[-2])]
        env.robot.jointLimitHigh = env.robot.workspacesdict[str(list(env.robot.workspacesdict)[-1])]
        rmse, mae, successRate1, successRate5, avgJntVel = self.evaluatePolicy(self.config['testSamples'],
                                                                               model, env)

        print("RMSE:", rmse)
        print("MAE:", mae)
        print("Success Rate 1 cm:", successRate1)
        print("Success Rate 5 cm:", successRate5)
        print("Average joint velocities:", avgJntVel)
        
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--envName',type=str, help="Name of the environment")
    parser.add_argument('--expNumber',type=int, help="Experiement number")
    parser.add_argument('--curriNumber', type=int, help="If you want to use Curriculum learning,then determine the experiement number")
    parser.add_argument('--total_timesteps', type=float)
    parser.add_argument('--mode', type=bool, help="Traning:True, Testing:False")
    parser.add_argument('--render', type=bool, help="Rendering")
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
    args = parser.parse_args()

    with open('configPPO.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    for arg in args._get_kwargs():
        if not arg[1]==None:
            print(arg[0])
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
            env._max_episode_steps = trainTest.config['max_episode_steps']
        else:
            env = make_vec_env(env_id=trainTest.config['envName'], n_envs=trainTest.config['n_envs'])
            for i in range(trainTest.config['n_envs']):
                env.envs[i].robot.config = config
                env.envs[i].task.config = config
                env.envs[i]._max_episode_steps = trainTest.config['max_episode_steps']

        trainTest.train(algorithm, env)
        env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
        env.robot.config = config
        env.task.config = config
        env._max_episode_steps = trainTest.config['max_episode_steps']
        time.sleep(5)
        trainTest.loadAndEvaluateModel(algorithm,env)
    else:
        env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
        env.robot.config = config
        env.task.config = config
        env._max_episode_steps = trainTest.config['max_episode_steps']
        trainTest.loadAndEvaluateModel(algorithm,env)
    print("DONE!!!")
    
if __name__=='__main__':
    main()
    
