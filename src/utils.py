import typing

import gymnasium as gym
import torch

from .wrappers import *


def make_env(env_id, seed, idx, capture_video, run_name):
    """Helper function to create an environment with some standard wrappers.

    Taken from cleanRL's DQN Atari implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py.
    """

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk


def set_cuda_configuration(gpu: typing.Any) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""
    if gpu is None or gpu == -1 or gpu is False:
        device = torch.device("cpu")
    elif isinstance(gpu, int):
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda")

    return device
