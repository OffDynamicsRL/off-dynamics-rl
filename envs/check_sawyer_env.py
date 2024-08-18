import numpy as np
import pickle
import gzip
import h5py
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))

from sawyer.call_sawyer_env import call_sawyer_env
import torch
from PIL import Image
import os
import yaml
import imageio


def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    video_writer = imageio.get_writer("episode_{}_video.mp4".format(episode_id), fps=30)
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))
    
    # save video
    png_files = os.listdir(filename)
    num = len(png_files)
    for i in range(num):
        image = imageio.imread(filename + '/frame_{}.png'.format(i))
        video_writer.append_data(image)
    video_writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='antmaze-small', help='Maze type. small or default')
    parser.add_argument('--shift_level', default=None, help='shift level')
    parser.add_argument('--filename', type=str, help='file_name')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--steps', default=10000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    env_config = {
        'env_name': args.env,
        'shift_level': args.shift_level,
    }

    env = call_sawyer_env(env_config)
    env.seed(args.seed)

    device = torch.device("cuda")
    s = env.reset()
    # print(s.shape)
    act = env.action_space.sample()
    done = False

    if args.video:
        frames = []
    
    ts = 0
    num_episodes = 0
    episode_step = 0
    episode_reward = 0
    for k in range(args.steps):
        episode_step += 1
        state = torch.FloatTensor(s).to(device)

        # use the controller to gather trajectories
        act = env.action_space.sample()

        ns, r, done, info = env.step(act)
        episode_reward += r

        if episode_step % 200 == 0:
            done=True

        ts += 1

        if done:
            done = False
            ts = 0
            print(episode_reward)
            episode_reward = 0
            s = env.reset()
            if args.video:
                frames = np.array(frames)
                save_video('./videos/', args.env + '_', frames, num_episodes)
            
            num_episodes += 1
            print('episode step gives', episode_step)
            episode_step = 0
            frames = []
        else:
            s = ns

        if args.video:
            curr_frame = env.sim.render(600,600,camera_name="corner", depth=False)
            frames.append(curr_frame)

if __name__ == '__main__':
    main()

