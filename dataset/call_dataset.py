import gym
import d4rl
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
import gym
import d4rl

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def call_tar_dataset(tar_env_name, shift_scale, quality='random'):
    if '-' in tar_env_name:
        tar_env_name = tar_env_name.replace('-', '_')

    if any(name in tar_env_name for name in ['halfcheetah', 'hopper', 'walker2d']) or tar_env_name.split('_')[0] == 'ant':
        domain = 'mujoco'
        make_env_name = tar_env_name.split('_')[0]
        env = gym.make(make_env_name + '-medium-v2')
        _max_episode_steps = env._max_episode_steps
    elif any(name in tar_env_name for name in ['pen', 'door', 'relocate', 'hammer']):
        domain = 'adroit'
        make_env_name = tar_env_name.split('_')[0]
        env = gym.make(make_env_name + '-cloned-v0')
        _max_episode_steps = env._max_episode_steps
    elif 'antmaze' in tar_env_name:
        domain = 'antmaze'
        make_env_name = tar_env_name.split('_')[0]
        if 'small' in tar_env_name:
            env = gym.make(make_env_name + '-umaze-diverse-v0')
        else:
            env = gym.make(make_env_name + '-medium-diverse-v0')
        _max_episode_steps = env._max_episode_steps
    else:
        raise NotImplementedError

    if domain == 'antmaze':
        tar_dataset_path = str(Path(__file__).parent.absolute()) + '/' + domain + '/' + tar_env_name + '_' + str(shift_scale) + '.hdf5'
    else:
        tar_dataset_path = str(Path(__file__).parent.absolute()) + '/' + domain + '/' + tar_env_name + '_' + str(shift_scale) + '_' + str(quality) + '.hdf5'

    data_dict = {}
    with h5py.File(tar_dataset_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
        
    dataset = data_dict
    
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    # count how many trajectories are included, ensure that the quantity of trajectories do not exceed number_of_trajectories
    counter = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        try:
            reward = dataset['rewards'][i].astype(np.float32)[0]
        except:
            reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == _max_episode_steps - 1)

        if done_bool or final_timestep:
            counter +=1
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }

