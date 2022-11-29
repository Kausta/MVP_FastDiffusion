from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["EMA"]

class EMA(nn.Module):
    module: nn.Module

    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, beta=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.beta = beta

    def _update(self, model: nn.Module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module):
        self._update(model, update_fn=lambda e, m: self.beta * e + (1. - self.beta) * m)

    def set(self, model: nn.Module):
        self._update(model, update_fn=lambda e, m: m)