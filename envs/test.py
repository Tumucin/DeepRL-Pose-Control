from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(save_freq=1000,save_path="./a2c",
                                         name_prefix='test')
model = A2C("MlpPolicy","CartPole-v1",verbose=1, tensorboard_log="./a2c")
model.learn(total_timesteps=100, callback=checkpoint_callback)
