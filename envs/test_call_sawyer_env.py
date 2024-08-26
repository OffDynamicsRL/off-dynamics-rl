from sawyer.call_sawyer_env import call_sawyer_env

# create env

import gym
import numpy as np
import time

env_config = {
    'env_name': 'sawyer-faucet-close-broken',
    'shift_level': 'medium',
}

env = call_sawyer_env(env_config)

# if this is okay, then test scripted policies

from sawyer.policies.sawyer_faucet_close_v2_policy import SawyerFaucetCloseV2Policy as policy

p = policy()

# interact with env
seed = 2024

env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
obs = env.reset()

count = 0
done = False

info = {}

while count < 500 and not done:
    action = p.get_action(obs)
    next_obs, _, _, info = env.step(action)
    # env.render()
    print(count, next_obs)
    if int(info["success"]) == 1:
        done = True
    obs = next_obs
    time.sleep(0.02)
    count += 1

print(info)

