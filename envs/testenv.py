from panda_gym.envs.panda_tasks.panda_reach import PandaReachEnv
import sys
sys.path.append('/scratch/users/tbal21/.conda/envs/stableBaselines/lib/python3.8/site-packages')
sys.path.append('/scratch/users/tbal21/.conda/envs/stableBaselines/panda-gym')
import gym
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv


#env = gym.make('PandaReach-v2', render=True)
env = make_vec_env('PandaReach-v2', n_envs=2) 
env.reset()
#print(print(env.envs[0].robot.config))
#print(env.envs[0].action_space.sample())
for i in range(1,10):
    action = env.action_space.sample()
    acton=action*0
    obs, reward, done, info = env.step(action)
    
    print("i:",i)
    print("obs achieved_goal environment 1:",obs['achieved_goal'][0])
    print("obs achieved_goal environment 2:",obs['achieved_goal'][1])
    print("obs desired_goal environment 1:",obs['observation'][0])
    print("obs desired_goal environment 2:",obs['observation'][1])
    
    #if done[0]:
     #    env.reset()

