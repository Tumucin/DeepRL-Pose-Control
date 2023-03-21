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

class TRAINTEST():
    def __init__(self, config):
        self.config = config

    def train(self,algorithm,env):
        checkpoint_callback = CheckpointCallback(save_freq=self.config['save_freq'], save_path=self.config['modelSavePath'],
                                            name_prefix=self.config['envName']+"_"+self.config['algorithm']+"_"+str(self.config['expNumber']))
        modelDir = self.config['modelSavePath']
        logDir = self.config['logSavePath']
        if self.config['algorithm']=="PPO":
            if self.config['curriLearning'] ==True:
                print("CURRICULUM LEARNING FOR PPO")
                model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['expNumber']), env=env,tensorboard_log= logDir+"/"+self.config['envName']+"_"+self.config['algorithm']+"/PPO_"+str(self.config['expNumber']))
            else:
                print("NORMAL LEARNING FOR PPO")
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128,128,])])

                model = algorithm(policy=self.config['policy'], env=env, n_steps=self.config['n_steps'],n_epochs=self.config['n_epochs'],verbose=self.config['verbose'], batch_size=self.config['batch_size'], learning_rate=self.config['learning_rate'],
                                tensorboard_log=logDir+"/"+self.config['envName']+"_"+self.config['algorithm'],policy_kwargs=policy_kwargs)
        
        if self.config['algorithm']=="DDPG":
            model = algorithm(policy=self.config['policy'], env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
                gamma=self.config['gamma'], batch_size=self.config['batch_size'], buffer_size=self.config['buffer_size'], learning_rate = 1e-3, replay_buffer_kwargs = self.config['rb_kwargs'],
                policy_kwargs = self.config['policy_kwargs'], tensorboard_log=logDir+"/"+self.config['envName']+""+self.config['algorithm'])
        
        if self.config['algorithm']=="TQC":
            
            if self.config['curriLearning'] ==True:
                print("CURRICULUM LEARNING FOR TQC")
                model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['expNumber']), env=env)
                print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")
                model.load_replay_buffer(self.config['bufferPath']+str(self.config['expNumber']))
                #model.set_env(env)
            else:
                print("NORMAL LEARNING FOR TQC")
                model = algorithm(policy=self.config['policy'], env=env, tensorboard_log=logDir+"/"+self.config['envName']+"_"+self.config['algorithm'],
                                verbose=self.config['verbose'], ent_coef=self.config['ent_coef'], batch_size=self.config['batch_size'], gamma=self.config['gamma'],
                                learning_rate=self.config['learning_rate'], learning_starts=self.config['learning_starts'],replay_buffer_class=HerReplayBuffer,
                                replay_buffer_kwargs=self.config['replay_buffer_kwargs'], policy_kwargs=self.config['policy_kwargs'])
        
        start_time = time.time()
        #, tb_log_name = logDir+"/"+config['envName']+"_"+config['algorithm']
        if self.config['curriLearning'] ==True:
            model.learn(total_timesteps=self.config['total_timesteps'], callback=checkpoint_callback, reset_num_timesteps=False, tb_log_name = logDir+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+self.config['algorithm'] +"_"+str(self.config['curriNumber']))
        else:
            model.learn(total_timesteps=self.config['total_timesteps'], callback=checkpoint_callback)
        print("Total time:", time.time()-start_time)
        if self.config['curriLearning'] ==True:
            model.save(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['curriNumber']))
        else:
            model.save(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['expNumber']))
        
        if self.config['algorithm']=='TQC':
            if self.config['curriLearning'] ==True:
                print("REPLAY BUFFER IS SAVED--CURRICULUM LEARNING")
                model.save_replay_buffer(self.config['bufferPath']+str(self.config['curriNumber']))   
            else:
                print("REPLAY BUFFER IS SAVED--NORMAL LEARNING")
                model.save_replay_buffer(self.config['bufferPath']+str(self.config['expNumber']))   

        #del model
        try:
            os.rename(self.config['logSavePath']+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+self.config['algorithm']+"_"+str(self.config['curriNumber'])+"_0", self.config['logSavePath']+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+self.config['algorithm']+"_"+str(self.config['curriNumber']))
        except FileNotFoundError:
            pass

    def load_model(self, algorithm,env):
        modelDir = self.config['modelSavePath']
        if self.config['curriLearning'] ==True:
            model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['curriNumber']), env=env) 
        else:
            model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+str(self.config['expNumber']), env=env) 
        env = model.get_env()
        mae = 0.0
        squaredError = 0.0
        successRate1 = 0.0
        successRate5 = 0.0
        avgJntVel = 0.0
        for step in range(self.config['testSamples']):
            print("step:", step)
            counter = 0        
            done = False
            obs = env.reset()
            episode_reward = 0.0
            while not done:
                counter+=1
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if counter==self.config['max_episode_steps']-2:
                    error = abs(obs['achieved_goal'] - obs['desired_goal'])
                    mae = np.linalg.norm(error) + mae
                    squaredError += np.sum(error**2)
                    avgJntVel = np.linalg.norm(action) + avgJntVel
                    if np.linalg.norm(error) <=0.01:
                        successRate1+=1
                    
                    if np.linalg.norm(error) <=0.05:
                        successRate5+=1
                if done:
                    #print("episode done.")
                    pass

                episode_reward+=reward
                #env.render()
            #print("episode reward is:", episode_reward)
        #print("Squared Error:",squaredError)
        print("RMSE:", np.sqrt((squaredError)/(self.config['testSamples'])))
        print("MAE:", mae/self.config['testSamples'])
        print("Success Rate 1 cm:", successRate1/self.config['testSamples'])
        print("Success Rate 5 cm:", successRate5/self.config['testSamples'])
        print("Average joint velocities:", avgJntVel/self.config['testSamples'])
        
def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
        
def main():
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--expNumber',type=int, help="Experiement number")
    parser.add_argument('--curriNumber', type=int, help="If you want to use Curriculum learning,then determine the experiement number")
    parser.add_argument('--total_timesteps', type=float)
    parser.add_argument('--mode', type=bool, help="Traning:True, Testing:False")
    parser.add_argument('--render', type=bool, help="Rendering")
    parser.add_argument('--gamma', type=float, help=" Discount factor")
    parser.add_argument('--n_steps', type= int, help="The number of steps to run for each environment per update")
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
    parser.add_argument('--curriLearning', type= bool, help="If this is true, than determine curriNumber")
    parser.add_argument('--lambdaErr', type= float, help="")
    parser.add_argument('--accelerationConstant', type= float, help="")
    parser.add_argument('--velocityConstant', type= float, help="")
    parser.add_argument('--velocityThreshold', type= float, help="")
    parser.add_argument('--thresholdConstant', type= float, help="")

    args = parser.parse_args()

    for arg in args._get_kwargs():
        if not arg[1]==None:
            print(arg[0])
            config[arg[0]] = arg[1]
    
    """

    with open('configPPO.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
        
    trainTest = TRAINTEST(config)
    print("number of env:",trainTest.config['n_envs'])
    if trainTest.config['algorithm']=="PPO":
        algorithm = PPO
        
    if trainTest.config['algorithm']=="DDPG":
        algorithm = DDPG
    if trainTest.config['algorithm']=="TQC":
        algorithm = TQC
    
    env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
    env._max_episode_steps = trainTest.config['max_episode_steps']
    
    if trainTest.config['mode'] == True:
        env = make_vec_env('PandaReach-v2', n_envs=trainTest.config['n_envs']) 
        trainTest.train(algorithm, env)
        env = gym.make(trainTest.config['envName'], render=trainTest.config['render'])
        env._max_episode_steps = trainTest.config['max_episode_steps']
        time.sleep(5)
        trainTest.load_model(algorithm,env)
    else:
        trainTest.load_model(algorithm,env)
    print("DONE!!!")
    
if __name__=='__main__':
    main()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        print("number of device",th.cuda.device_count())
        print("current device:",th.cuda.current_device())
        print("device name:",th.cuda.get_device_name())
        print('Allocated:', round(th.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(th.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
