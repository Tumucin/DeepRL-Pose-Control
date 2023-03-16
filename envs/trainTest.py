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


def train(config, algorithm,env):
    checkpoint_callback = CheckpointCallback(save_freq=config['save_freq'], save_path=config['modelSavePath'],
                                         name_prefix='PandaReach-v2TQC_'+config['expNumber'])
    modelDir = config['modelSavePath']
    logDir = config['logSavePath']
    if config['algorithm']=="PPO":
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128,128,])])

        model = algorithm(policy=config['policy'], env=env, n_steps=config['n_steps'],verbose=1, batch_size=config['batch_size'], learning_rate=config['learning_rate'],
                          tensorboard_log=logDir+"/"+config['envName']+"_"+config['algorithm'],policy_kwargs=policy_kwargs)
    
    if config['algorithm']=="DDPG":
        model = algorithm(policy=config['policy'], env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
             gamma=config['gamma'], batch_size=config['batch_size'], buffer_size=config['buffer_size'], learning_rate = 1e-3, replay_buffer_kwargs = config['rb_kwargs'],
             policy_kwargs = config['policy_kwargs'], tensorboard_log=logDir+"/"+config['envName']+""+config['algorithm'])
    
    if config['algorithm']=="TQC":
        #c = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[128, 128,128], vf=[128,128, 128])])
        #policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,net_arch=dict(pi=[128, 128], qf=[128, 128]))
        if config['continueTraining'] ==True:
            #model = algorithm.load(modelDir+"/"+config['envName']+""+config['algorithm']+"_"+config['expNumber'], env=env)
            #model.set_env(env)
            pass
        else:
            #print("here")
            model = algorithm(policy=config['policy'], env=env, tensorboard_log=logDir+"/"+config['envName']+""+config['algorithm'],
                            verbose=1, ent_coef=config['ent_coef'], batch_size=config['batch_size'], gamma=config['gamma'],
                            learning_rate=config['learning_rate'], learning_starts=config['learning_starts'],replay_buffer_class=HerReplayBuffer,
                            replay_buffer_kwargs=config['replay_buffer_kwargs'], policy_kwargs=config['policy_kwargs'])
    model.learn(total_timesteps=config['total_timesteps'], callback=checkpoint_callback)
    model.save(modelDir+"/"+config['envName']+""+config['algorithm']+"_"+config['expNumber'])     

    del model

def load_model(parentDir,config,steps, algorithm,env):
    modelDir = config['modelSavePath']
    model = algorithm.load(modelDir+"/"+config['envName']+""+config['algorithm']+"_"+config['expNumber'], env=env) 
    env = model.get_env()
    mae = 0.0
    squaredError = 0.0
    successRate1 = 0.0
    successRate5 = 0.0
    avgJntVel = 0.0
    for step in range(steps):
        print("step:", step)
        counter = 0        
        done = False
        obs = env.reset()
        episode_reward = 0.0
        while not done:
            counter+=1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            #print("obs['achieved_goal']",obs['achieved_goal'])
            #print("obs['desired_goal']",obs['desired_goal'])
            if counter==499:
                error = abs(obs['achieved_goal'] - obs['desired_goal'])
                #print("error:", error)
                #print("errorMagnitude:", np.linalg.norm(error))
                #print("obs['achieved_goal']",obs['achieved_goal'])
                #print("obs['desired_goal']",obs['desired_goal'])
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
    print("RMSE:", np.sqrt((squaredError)/(steps)))
    print("MAE:", mae/steps)
    print("Success Rate 1 cm:", successRate1/steps)
    print("Success Rate 5 cm:", successRate5/steps)
    print("Average joint velocities:", avgJntVel/steps)
        
        
def main():

    with open('configPath.yaml') as cfgPath:
        configPath = yaml.load(cfgPath, Loader=SafeLoader)
        with open(configPath['configPath']) as cfg:
            config = yaml.load(cfg, Loader=SafeLoader)

    currentDir = os.getcwd()
    if config['algorithm']=="PPO":
        algorithm = PPO
        
    if config['algorithm']=="DDPG":
        algorithm = DDPG

    if config['algorithm']=="TQC":
        algorithm = TQC
    
    env = gym.make(config['envName'], render=config['render'])
    
    env._max_episode_steps = 700
    
    if config['mode'] == True:
        env = make_vec_env('PandaReach-v2', n_envs=1)
        start_time = time.time()
        train(config,algorithm, env)
        print("Total time:", time.time()-start_time)
    else:
        load_model(currentDir,config,5, algorithm,env)
    
if __name__=='__main__':
    main()
