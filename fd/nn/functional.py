import math
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

__all__ = ["log_prob_standard_normal", "log_prob_mvdiag_normal"]

@torch.jit.script
def log_prob_standard_normal(z: Tensor) -> Tensor:
    return -0.5 * (torch.log(2. * np.pi) + (z ** 2))

@torch.jit.script
def log_prob_mvdiag_normal(z: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
    return -0.5 * (torch.log(2. * np.pi) + log_var + torch.exp(-log_var) * ((z - mean) ** 2))
