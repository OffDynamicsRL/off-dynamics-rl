from typing import Dict
import gym


def call_sawyer_env(env_config: Dict) -> gym.Env:
    env_name     =   env_config['env_name'].lower()     #   eg. "sawyer-pick-place"
    shift_level  =   env_config['shift_level']          #   level(easy/medium/hard)

    print(env_name)

    if '_' in env_name:
        env_name = env_name.replace('_', '-')
    # decide which task it is, support the following tasks
    # box/pick-place/button-press/push/hammer         -morph-gripper
    #                                  - broken
    assert any([env_name.startswith(f'{e}') for e in ['sawyer']])

    if shift_level is not None:
        assert any([env_name.endswith(f'{e}') for e in ['broken', 'morph-gripper']])

    if shift_level is not None:
        env_name = env_name + '-' + str(shift_level) + '-v2'
    else:
        env_name = env_name + '-v2'

    return gym.make(env_name)