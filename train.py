import gym
import random
from stable_baselines3 import PPO
import random
import os



models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make("LunarLander-v2")
model = PPO("MlpPolicy", env, verbose=1)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

"""
def policy(observation): 
   return random.randint(0,3)
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
"""