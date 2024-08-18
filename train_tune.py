import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make source domain envs
import algo.utils as utils

from pathlib                              import Path
from algo.call_tune_algo                  import call_tune_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--src_policy", default="SAC", help='policy to use in the source domain')
    parser.add_argument("--tar_policy", default="SAC", help='policy to use in the target domain')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="expert", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--max_tar_step', default=int(1e5), type=int)  # the maximum gradient steo for off-dynamics rl learning in tar env
    parser.add_argument('--max_src_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning in src env
    parser.add_argument('--src_params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--tar_params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    
    args = parser.parse_args()

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env.split('-')[0]
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments
    if 'antmaze' in src_env_name:
        if 'small' in src_env_name:
            src_env = gym.make('antmaze-umaze-v0')
            src_eval_env = gym.make('antmaze-umaze-v0')
        elif 'medium' in src_env_name:
            src_env = gym.make('antmaze-medium-play-v0')
            src_eval_env = gym.make('antmaze-medium-play-v0')
        elif 'large' in src_env_name:
            src_env = gym.make('antmaze-large-play-v0')
            src_eval_env = gym.make('antmaze-large-play-v0')
        src_env.seed(args.seed)
        src_eval_env.seed(args.seed + 100)
    elif domain == 'adroit':
        src_env_name += '-expert-v0'
        src_env = gym.make(src_env_name)
        src_env.seed(args.seed)
        src_eval_env = gym.make(src_env_name)
        src_eval_env.seed(args.seed + 100)
    else:
        src_env_name += '-expert-v2'
        src_env = gym.make(src_env_name)
        src_env.seed(args.seed)
        src_eval_env = gym.make(src_env_name)
        src_eval_env.seed(args.seed + 100)

    tar_env_config = {
        'env_name': tar_env_name,
        'shift_level': args.shift_level,
    }
    tar_env = call_env[domain](tar_env_config)
    tar_env.seed(args.seed)
    tar_eval_env = call_env[domain](tar_env_config)
    tar_eval_env.seed(args.seed + 100)
    
    if args.mode != 0:
        raise NotImplementedError # cannot support other modes
    
    src_policy_config_name = args.src_policy.lower()
    tar_policy_config_name = args.tar_policy.lower()
    
    # we only support finetuning using the same algorithm currently
    assert src_policy_config_name == tar_policy_config_name

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{src_policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        src_config = yaml.safe_load(f)
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{tar_policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        tar_config = yaml.safe_load(f)
    
    if args.src_params is not None:
        override_params = json.loads(args.src_params)
        src_config.update(override_params)
        print('The following parameters are updated to source policy:', args.src_params)

    if args.tar_params is not None:
        override_params = json.loads(args.tar_params)
        tar_config.update(override_params)
        print('The following parameters are updated to target policy:', args.tar_params)

    print("------------------------------------------------------------")
    print("Source Policy: {}, Target Policy: {}, Env: {}, Seed: {}".format(args.src_policy, args.tar_policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    # log path, we use logging with tensorboard
    policy_path = args.src_policy + '_' + args.tar_policy
    if domain == 'adroit':
        if args.mode == 1:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + '/r' + str(args.seed)
        elif args.mode == 2:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-tardatatype-' + args.tartype + '-' + '/r' + str(args.seed)
        elif args.mode == 3:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-srcdatatype-' + args.srctype + 'tardatatype-' + args.tartype + '-' + '/r' + str(args.seed)
        else:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '/r' + str(args.seed)
    else:
        if args.mode == 1:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + str(args.shift_level) + '/r' + str(args.seed)
        elif args.mode == 2:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
        elif args.mode == 3:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
        else:
            outdir = args.dir + '/' + policy_path + '/' + args.env + '-' + str(args.shift_level) + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    src_env.action_space.seed(args.seed) if src_env is not None else None
    tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    src_config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'max_step': int(args.max_src_step),
        'shift_level': shift_level,
    })

    tar_config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'max_step': int(args.max_tar_step),
        'shift_level': shift_level,
    })

    src_policy = call_tune_algo(args.src_policy, src_config, args.mode, device)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Source Policy: {}, Target Policy {}; Env: {}, seed: {}'.format(args.src_policy, args.tar_policy, args.env, args.seed))
        for item in src_config.items():
            f.write('\n {}'.format(item))
        for item in tar_config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    eval_cnt = 0
    
    eval_src_return = eval_policy(src_policy, src_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    # training paradigm here is different from train.py, the policy is first trained in the source domain, and then transfer to the target domain
    # that being said, the policy from the source domain is previously trained

    # we first train the source domain policy
    # online-online learning
    src_state, src_done = src_env.reset(), False
    src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0

    for t in range(int(src_config['max_step'])):
        src_episode_timesteps += 1

        # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
        src_action = (
            src_policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
        ).clip(-max_action, max_action)

        src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
        src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

        src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

        src_state = src_next_state
        src_episode_reward += src_reward

        src_policy.train(src_replay_buffer, batch_size=src_config['batch_size'], writer=writer)
        
        if src_done: 
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
            writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

            src_state, src_done = src_env.reset(), False
            src_episode_reward = 0
            src_episode_timesteps = 0
            src_episode_num += 1

        if (t + 1) % src_config['eval_freq'] == 0:
            src_eval_return = eval_policy(src_policy, src_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
            eval_cnt += 1

            if args.save_model:
                src_policy.save('{}/models/model'.format(outdir))

    # finish source domain training, get source domain policy
    print('Finish source domain policy learning!')
    print('Start training in the target domain!')

    tar_policy = call_tune_algo(args.tar_policy, tar_config, args.mode, device)
    
    # target domain is online
    tar_state, tar_done = tar_env.reset(), False
    tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

    for t in range(int(tar_config['max_step'])):
        # interaction with tar env
        tar_episode_timesteps += 1
        tar_action = (src_policy.select_action(np.array(tar_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
        ).clip(-max_action, max_action)

        tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
        tar_done_bool = float(tar_done) if tar_episode_timesteps < src_eval_env._max_episode_steps else 0

        tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

        tar_state = tar_next_state
        tar_episode_reward += tar_reward

        # the same policy, then finetune directly on the policy from the source domain
        src_policy.train(tar_replay_buffer, batch_size=src_config['batch_size'], writer=writer)
        
        if tar_done:
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
            writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)

            tar_state, tar_done = tar_env.reset(), False
            tar_episode_reward = 0
            tar_episode_timesteps = 0
            tar_episode_num += 1

        if (t + 1) % tar_config['eval_freq'] == 0:
            tar_eval_return = eval_policy(src_policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)

            eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
            writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)
            
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))

    writer.close()