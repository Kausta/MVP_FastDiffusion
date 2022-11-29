import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import fd.nn as lnn
import fd.nn.functional as LF

from .simple_ae import Encoder, Decoder

__all__ = ["SimpleVAE"]

class SimpleVAE(pl.LightningModule):
    def __init__(self,
                 in_ch=3,
                 latent_dim=256,
                 out_ch=3,
                 encoder_layers=[16, 32, 64, 128, 256],
                 decoder_layers=[256, 128, 64, 32, 16], 
                 prior: lnn.PriorBase = lnn.StandardNormalPrior(),
                 act_fn=nn.SiLU,
                 final_act_fn=nn.Tanh):
        super().__init__()

        self.encoder = Encoder(in_ch, latent_dim, encoder_layers, act_fn)
        self.decoder = Decoder(
            latent_dim, out_ch, decoder_layers, act_fn, final_act_fn=final_act_fn)
        self.prior = prior

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, stage="train"):
        mean, log_var = self.encoder(x)
        if stage == "train":
            z = self.reparametrize(mean, log_var)
        else:
            z = mean
        out = self.decoder(z)
        return out, {
            "mean": mean,
            "log_var": log_var,
            "z": z
        }

    def reparametrize(self, mean, log_var):
        return mean + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    def kl_divergence(self, z, mean, log_var, reduction="sum"):
        B = z.shape[0]
        z, mean, log_var = z.reshape(B, -1), mean.reshape(B, -1), log_var.reshape(B, -1)
        
        posterior_log_prob = LF.log_prob_mvdiag_normal(z, mean, log_var)
        prior_log_prob = self.prior.log_prob(z)

        kl_div = posterior_log_prob - prior_log_prob
        if reduction == "sum":
            return kl_div.sum(dim=-1)
        elif reduction == "mean":
            return kl_div.mean(dim=-1)
        elif reduction is None:
            return kl_div
        else:
            raise ValueError(f"Unknown reduction {reduction}")