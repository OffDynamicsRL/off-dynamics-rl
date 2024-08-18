from sawyer.call_sawyer_env import call_sawyer_env

# create env

import gym
import numpy as np

env_config = {
    'env_name': 'sawyer-pick-place-broken',
    'shift_level': 'easy',
}

env = call_sawyer_env(env_config)

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)

print(obs.shape)
print(a.shape)

env_config = {
    'env_name': 'sawyer-pick-place-morph-gripper',
    'shift_level': 'easy',
}

env = call_sawyer_env(env_config)

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)

print(obs.shape)
print(a.shape)

env_config = {
    'env_name': 'sawyer-pick-place',
    'shift_level': None,
}

env = call_sawyer_env(env_config)

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)

print(obs.shape)
print(a.shape)