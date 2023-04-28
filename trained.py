import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/150000.zip"
model = PPO.load(model_path, env=env)


vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000): 
   action, _ = model.predict(obs, deterministic=True)
   obs, reward, done, info = vec_env.step(action)
   vec_env.render()
