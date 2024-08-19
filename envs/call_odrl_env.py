from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))

from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.sawyer.call_sawyer_env          import call_sawyer_env


def call_odrl_env(env_type='mujoco',
                  env_name='halfcheetah-friction',
                  shift_level='0.5'):
                  
    # we support mujoco, antmaze, adroit and sawyer tasks now
    env_type = env_type.lower()
    assert env_type in ['mujoco', 'antmaze', 'adroit', 'sawyer']

    env_config = {
        'env_name': env_name,
        'shift_level': shift_level,
    }

    if env_type == 'mujoco':
        env = call_mujoco_env(env_config)
    elif env_type == 'antmaze':
        env = call_antmaze_env(env_config)
    elif env_type == 'adroit':
        env = call_adroit_env(env_config)
    else:
        env = call_sawyer_env(env_config)
    
    return env


def call_odrl_dataset(env_name='halfcheetah-friction',
                      shift_level='0.5',
                      dataset_type='random',
                      ):
    dataset = call_tar_dataset(env_name, shift_level, dataset_type)

    return dataset

