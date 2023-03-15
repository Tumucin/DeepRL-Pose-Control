import gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import os 
###########################################################################
### This script trains the CartPole environment with PPO RL algorithm
### and saves the model and model history. It has 2 functionality, Training
### and inference...
### USAGE: 
# For training python3 cartPolePPOTrainTest.py -envName "CartPole-v1" -outputName "cartPolePPO" -mode "1"
# For testing python3 cartPolePPOTrainTest.py -envName "CartPole-v1" -outputName "cartPolePPO" -mode "0"
# To see the tensorboard , go to tensorboard dir and run: tensorboard --logdir .
###########################################################################

def train_save_PPO_CartPole(env, modelName):
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpole_tensorboard/")
    model.learn(total_timesteps=1)
    model.save(modelName)

    del model

def load_model(env, modelName):

    model = PPO.load(modelName, env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    env.close()

def load_model2(env, modelName, steps):

    model = PPO.load(modelName, env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    env = model.get_env()
    for step in range(steps):
        done = False
        obs = env.reset()
        episode_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward+=reward
            env.render()
        print("episode reward is:", episode_reward)


def main():
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    modelPath = os.path.join(parentDir, "model")
    print("model path:",modelPath)
    parser = argparse.ArgumentParser()
    parser.add_argument("-envName", type=str,help="(str)Environment name: ", required=True)
    parser.add_argument("-outputName", type=str,help="(str)The name of the file to be saved", required=True)
    parser.add_argument("-mode", type=str,help="(str)1 for training, 0 for inference", required=True)
    args = parser.parse_args()
    env = gym.make(args.envName)
    modelName = args.outputName
    #args.mode=False
    if args.mode=="1":
        #Training
        print("TRAINING...")
        train_save_PPO_CartPole(env, modelName)
    else:
        print("TESTING...")
        load_model2(env, modelName,5)
main()
