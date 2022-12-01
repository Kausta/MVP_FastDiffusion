import math
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

__all__ = ["log_prob_standard_normal", "log_prob_mvdiag_normal", "soft_clamp5"]

@torch.jit.script
def log_prob_standard_normal(z: Tensor) -> Tensor:
    return -0.5 * (torch.log(2. * np.pi) + (z ** 2))

@torch.jit.script
def log_prob_mvdiag_normal(z: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
    return -0.5 * (torch.log(2. * np.pi) + log_var + torch.exp(-log_var) * ((z - mean) ** 2))

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)    #  5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]