import abc
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import distributions, nn
import torch.nn.functional as F


class Actor(abc.ABC, nn.Module):
    @abc.abstractmethod
    def distribution(self, obs) -> distributions.Distribution:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob_from_distribution(
        self, pi: distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distribution(pi, act)
        return pi, logp_a


def _mlp(dims: Sequence[int], activation: nn.Module):
    layers: List[nn.Module] = []
    dims = list(dims)
    for i in range(len(dims) - 1):
        if i != 0:
            layers.append(activation)
        layers.append(nn.Linear(dims[i], dims[i + 1]))
    return nn.Sequential(*layers)


class MLPCategoricalActor(Actor):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh()
    ) -> None:
        super().__init__()
        self.logits_net = _mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def distribution(self, obs) -> distributions.Distribution:
        logits = self.logits_net(obs)
        return distributions.Categorical(logits=logits)

    def log_prob_from_distribution(
        self, pi: distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(
        self, obs_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh()
    ) -> None:
        super().__init__()
        self.value_function = _mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return self.value_function(obs)


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64), activation=nn.Tanh()
    ) -> None:
        super().__init__()
        self.actor = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
        self.critic = MLPCritic(obs_dim, hidden_sizes, activation)

    @torch.no_grad()
    def step(self, obs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step

        Args:
            obs (Observation): [description]

        Returns:
            actions (np.ndarray): selected actions.
            values (np.ndarray): estimated values of the observation.
            logp_a (np.ndarray): log probability of the selected actions.
        """
        pi = self.actor.distribution(obs)
        a = pi.sample()
        logp_a = self.actor.log_prob_from_distribution(pi, a)
        v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs) -> np.ndarray:
        return self.step(obs)[0]
