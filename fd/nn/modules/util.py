import torch
import torch.nn as nn

__all__ = ["BatchReshape"]

class BatchReshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape((x.shape[0], *self.shape))