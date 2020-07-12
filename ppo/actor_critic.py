import abc
from typing import Optional, Tuple

import numpy as np
import torch
from torch import dist, distributions
import torch.nn as nn
import torch.distributions as distributions


Observation = torch.Tensor


class Actor(abc.ABC, nn.Module):
    @abc.abstractmethod
    def _distribution(self, obs: Observation) -> distributions.Distribution:
        ...

    @abc.abstractmethod
    def _log_prob_from_distribution(
        self, pi: distributions.Distribution, act: Optional[torch.Tensor]
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
