import gym
from stable_baselines3 import PPO
import random

def policy(observation): 
   return random.randint(0,3)
env = gym.make("LunarLander-v2")
observation = env.reset()
for _ in range(10000):
   action = policy(observation)  # User-defined policy function
   observation, reward, done, info = env.step(action)
   env.render()
   if done:
      observation = env.reset()
env.close()
