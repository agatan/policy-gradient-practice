import abc
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from gym import spaces


Observation = torch.Tensor


class Actor(abc.ABC, nn.Module):
    @abc.abstractmethod
    def _distribution(self, obs: Observation) -> distributions.Distribution:
        ...

    @abc.abstractmethod
    def _log_prob_from_distribution(
        self, pi: distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        ...

    def forward(
        self, obs: Observation, act: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


def _mlp(
    sizes: Sequence[int],
    activation: Callable[[torch.Tensor], torch.Tensor],
    output_activation=nn.Identity,
):
    layers: List[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPCategoricalActor(Actor):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], activation=F.relu
    ) -> None:
        super().__init__()
        self.logits_net = _mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs: Observation) -> distributions.Distribution:
        logits = self.logits_net(obs)
        return distributions.Categorical(logits=logits)

    def _log_prob_from_distribution(
        self, pi: distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(
        self, obs_dim: int, hidden_sizes: Sequence[int], activation=F.relu
    ) -> None:
        super().__init__()
        self.v_net = _mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: Observation) -> torch.Tensor:
        """Returns estimated values of observations.

        Args:
            obs (Observation): Observed state. (batch_size, obs_dim)

        Returns:
            torch.Tensor: estimated values. (batch_size,)
        """
        return self.v_net(obs).squeeze(1)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space: spaces.Space,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)

        self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    @torch.no_grad
    def step(self, obs: Observation) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step

        Args:
            obs (Observation): [description]

        Returns:
            actions (np.ndarray): selected actions.
            values (np.ndarray): estimated values of the observation.
            logp_a (np.ndarray): log probability of the selected actions.
        """
        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs: Observation) -> np.ndarray:
        return self.step(obs)[0]
