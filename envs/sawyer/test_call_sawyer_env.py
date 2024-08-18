from sawyer.call_sawyer_env import call_sawyer_env

# create env

import gym
import numpy as np

env_config = {
    'env_name': 'sawyer-pick-place-broken',
    'shift_level': 'easy',
}

env = call_sawyer_env(env_config)