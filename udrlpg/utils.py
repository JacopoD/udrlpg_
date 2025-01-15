import torch.nn as nn
import numpy as np
import random
from udrlpg.dynamic_buckets import Buckets
import bisect
import torch.nn.functional as F
import torch
import copy


def scale(old_value, old_bottom, old_top, new_top, new_bottom):
    return (old_value - old_bottom) / (old_top - old_bottom) * (
        new_top - new_bottom
    ) + new_bottom


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

    return nn.Sequential(*layers)


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )


class ReplayBuffer:
    def __init__(
        self,
        rng: random.Random,
        max_size,
    ) -> None:
        self.buf = []
        if max_size is None:
            self.max_size = np.inf
        else:
            self.max_size = max_size
        self.max_seen = 0
        self.random = rng

    def add(self, element):
        # bisect.insort(self.buf, element, key=lambda x: x[0])
        self.buf.append(element)

        if self.len() > self.max_size:
            self.buf.pop(0)

        if element[0] > self.max_seen:
            self.max_seen = element[0]

    def sample(self, size, weights="uniform", scaling=1.0, separate=True):
        match weights:
            case "uniform":
                w = None
            case "reciprocal":
                w = (
                    np.reciprocal(np.arange(1, self.len() + 1, dtype=float))[::-1]
                    ** scaling
                )
            case _:
                raise NotImplementedError

        sample = self.random.choices(self.buf, weights=w, k=min(size, self.len()))

        if separate:
            return zip(*sample)
        return sample

    def len(self):
        return len(self.buf)


def parse_config(config, is_sweep):
    if (
        config.survival_bonus == "auto"
        or config.max_reward == "auto"
        or config.min_reward == "auto"
        or config.eval_frequency == "auto"
        or config.extensive_eval_frequency == "auto"
        or config.max_steps == "auto"
    ):
        assert (
            config.survival_bonus
            == config.max_reward
            == config.min_reward
            == config.eval_frequency
            == config.extensive_eval_frequency
            == config.max_steps
            == "auto"
        ), "All parameters should be either set manually or all set to auto"
        if is_sweep:
            config = dotdict(config.as_dict())
        config.survival_bonus = True
        match config.gym_env_name:
            case "CartPole-v1":
                config.max_reward = 500
                config.min_reward = 0
                config.eval_frequency = 1_000
                config.extensive_eval_frequency = 10_000
                config.max_steps = 100_000
            case "InvertedPendulum-v4":
                config.max_reward = 1000
                config.min_reward = 0
                config.eval_frequency = 1_000
                config.extensive_eval_frequency = 10_000
                config.max_steps = 100_000
            case "HalfCheetah-v4":
                config.max_reward = 4000
                config.min_reward = -100
                config.eval_frequency = 30_000
                config.extensive_eval_frequency = 300_000
                config.max_steps = 3_000_000
            case "Hopper-v4":
                config.max_reward = 3000
                config.min_reward = -100
                config.survival_bonus = False
                config.eval_frequency = 30_000
                config.extensive_eval_frequency = 300_000
                config.max_steps = 3_000_000
            case "Swimmer-v4":
                config.max_reward = 365
                config.min_reward = -100
                config.eval_frequency = 30_000
                config.extensive_eval_frequency = 300_000
                config.max_steps = 3_000_000
            case _:
                raise NotImplementedError
    return config


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
