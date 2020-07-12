import abc
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import dist, distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


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
