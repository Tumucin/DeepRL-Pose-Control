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
                                            name_prefix=self.config['envName']+"_"+self.config['algorithm']+"_"+self.config['expNumber'])
        modelDir = self.config['modelSavePath']
        logDir = self.config['logSavePath']
        if self.config['algorithm']=="PPO":
            if self.config['curriLearning'] ==True:
                print("CURRICULUM LEARNING FOR PPO")
                model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+self.config['expNumber'], env=env,tensorboard_log= logDir+"/"+self.config['envName']+"_"+self.config['algorithm']+"/PPO_"+self.config['expNumber'])
            else:
                print("NORMAL LEARNING FOR PPO")
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128,128,])])

                model = algorithm(policy=self.config['policy'], env=env, n_steps=self.config['n_steps'],verbose=self.config['verbose'], batch_size=self.config['batch_size'], learning_rate=self.config['learning_rate'],
                                tensorboard_log=logDir+"/"+self.config['envName']+"_"+self.config['algorithm'],policy_kwargs=policy_kwargs)
        
        if self.config['algorithm']=="DDPG":
            model = algorithm(policy=self.config['policy'], env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
                gamma=self.config['gamma'], batch_size=self.config['batch_size'], buffer_size=self.config['buffer_size'], learning_rate = 1e-3, replay_buffer_kwargs = self.config['rb_kwargs'],
                policy_kwargs = self.config['policy_kwargs'], tensorboard_log=logDir+"/"+self.config['envName']+""+self.config['algorithm'])
        
        if self.config['algorithm']=="TQC":
            
            if self.config['curriLearning'] ==True:
                print("CURRICULUM LEARNING FOR TQC")
                model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+self.config['expNumber'], env=env)
                print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")
                model.load_replay_buffer(self.config['bufferPath']+self.config['expNumber'])
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
            model.learn(total_timesteps=self.config['total_timesteps'], callback=checkpoint_callback, reset_num_timesteps=False, tb_log_name = logDir+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+self.config['algorithm'] +"_"+self.config['curriNumber'])
        else:
            model.learn(total_timesteps=self.config['total_timesteps'], callback=checkpoint_callback)
        print("Total time:", time.time()-start_time)
        if self.config['curriLearning'] ==True:
            model.save(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+self.config['curriNumber'])
        else:
            model.save(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+self.config['expNumber'])
        
        if self.config['algorithm']=='TQC':
            if self.config['curriLearning'] ==True:
                print("REPLAY BUFFER IS SAVED--CURRICULUM LEARNING")
                model.save_replay_buffer(self.config['bufferPath']+self.config['curriNumber'])   
            else:
                print("REPLAY BUFFER IS SAVED--NORMAL LEARNING")
                model.save_replay_buffer(self.config['bufferPath']+self.config['expNumber'])   

        #del model
        try:
            os.rename(self.config['logSavePath']+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+self.config['algorithm']+"_"+self.config['curriNumber']+"_0", self.config['logSavePath']+"/"+self.config['envName']+"_"+self.config['algorithm']+"/"+self.config['algorithm']+"_"+self.config['curriNumber'])
        except FileNotFoundError:
            pass

    def load_model(self, algorithm,env):
        modelDir = self.config['modelSavePath']
        if self.config['curriLearning'] ==True:
            model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+self.config['curriNumber'], env=env) 
        else:
            model = algorithm.load(modelDir+"/"+self.config['envName']+""+self.config['algorithm']+"_"+self.config['expNumber'], env=env) 
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
        
    
def main():
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
    
