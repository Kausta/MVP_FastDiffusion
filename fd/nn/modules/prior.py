import abc

import torch
import torch.nn as nn

from ..functional import log_prob_standard_normal

__all__ = ["PriorBase", "StandardNormalPrior"]


class PriorBase(nn.Module, abc.ABC):
    def __init__(self, L: int):
        super().__init__()

        assert L > 0, "Latent size of the prior should be positive"
        self.L = L

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return

    @abc.abstractmethod
    def sample(self, N: int, dtype=None, device=None) -> torch.Tensor:
        return


class StandardNormalPrior(PriorBase):
    def __init__(self, L: int):
        super().__init__(L)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return log_prob_standard_normal(x)

    def sample(self, N: int, dtype=None, device=None) -> torch.Tensor:
        return torch.randn((N, self.L), dtype=dtype, device=device)
