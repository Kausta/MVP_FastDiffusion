import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import _weight_norm

import pytorch_lightning as pl

import fd.nn as lnn
import fd.nn.functional as LF

from .simple_ae import Encoder, Decoder

__all__ = ["TranslationVAE"]

class TranslationVAE(pl.LightningModule):
    wn_initialized: torch.BoolTensor

    def __init__(self,
                 in_ch=3,
                 cond_ch=1,
                 latent_dim=256,
                 out_ch=3,
                 encoder_layers=[16, 32, 64, 128, 256],
                 decoder_layers=[256, 128, 64, 32, 16], 
                 act_fn=nn.SiLU,
                 final_act_fn=nn.Tanh):
        super().__init__()

        self.encoder = Encoder(in_ch, 2 * latent_dim, encoder_layers, act_fn)
        self.cond_encoder = Encoder(cond_ch, 2 * latent_dim, encoder_layers, act_fn)
        self.decoder = Decoder(
            latent_dim, out_ch, decoder_layers, act_fn, final_act_fn=final_act_fn)

        self.all_in_layers = []
        self.all_conv_layers = []
        self.register_buffer("wn_initialized", torch.scalar_tensor(False))

        self.num_power_iter = 2

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.affine:
                    self.all_in_layers.append(m)
            elif isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m = weight_norm(m)
                self.all_conv_layers.append(m)            

    def _data_dependent_init(self):
        def init_hook_(module, input, output):
            with torch.no_grad():
                std, mean = torch.std_mean(output, dim=[0, 2, 3])
                g = getattr(module, 'weight_g')
                g.copy_(g / (std.reshape((len(std), 1, 1, 1)) + 1e-8))
                b = getattr(module, 'bias')
                if b is not None:
                    b.copy_((b - mean) / (std + 1e-8))
                setattr(module, 'weight', _weight_norm(getattr(module, 'weight_v'), g, dim=0))
            return module._conv_forward(input[0], module.weight, module.bias)

        handles = []
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                handles.append(m.register_forward_hook(init_hook_))
        return handles

    def _data_dependent_post_init(self, handles):
        for h in handles:
            h.remove()

    def forward(self, x, x_cond, stage="train"):
        if not self.wn_initialized.item():
            handles = self._data_dependent_init()
        z_recon = LF.soft_clamp5(self.encoder(x))
        mean_recon, log_var_recon = torch.chunk(z_recon, 2, dim=1)
        if stage == "train":
            z_recon = self.reparametrize(mean_recon, log_var_recon)
        else:
            z_recon = mean_recon

        z_cond = LF.soft_clamp5(self.cond_encoder(x_cond))
        mean_cond, log_var_cond = torch.chunk(z_cond, 2, dim=1)
        if stage == "train":
            z_cond = self.reparametrize(mean_cond, log_var_cond)
        else:
            z_cond = mean_cond
        
        out = self.decoder(z_recon)
        out_cond = self.decoder(z_cond)

        if not self.wn_initialized.item():
            self._data_dependent_post_init(handles)
            self.wn_initialized.copy_(torch.scalar_tensor(True))

        return out, out_cond, {
            "mean_recon": mean_recon,
            "log_var_recon": log_var_recon,
            "mean_cond": mean_cond,
            "log_var_cond": log_var_cond,
            "z_recon": z_recon,
            "z_cond": z_cond
        }

    def reparametrize(self, mean, log_var):
        return mean + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    def kl_divergence(self, z, mean, log_var, reduction="sum"):
        B = z.shape[0]
        z, mean, log_var = z.reshape(B, -1), mean.reshape(B, -1), log_var.reshape(B, -1)

        posterior_log_prob = LF.log_prob_mvdiag_normal(z, mean, log_var)
        prior_log_prob = LF.log_prob_standard_normal(z)

        kl_div = posterior_log_prob - prior_log_prob
        if reduction == "sum":
            return kl_div.sum(dim=-1)
        elif reduction == "mean":
            return kl_div.mean(dim=-1)
        elif reduction is None:
            return kl_div
        else:
            raise ValueError(f"Unknown reduction {reduction}")

    def spectral_norm_loss(self):
        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        N = 0
        for i in weights:
            N += len(weights[i])
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                u_name, v_name = f"sr_u_{i[0]}_{i[1]}", f"sr_v_{i[0]}_{i[1]}"
                num_iter = self.num_power_iter
                if not hasattr(self, u_name):
                    print(f"Initializing {u_name} and {v_name}")
                    num_w, row, col = weights[i].shape
                    self.register_buffer(u_name, F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3))
                    self.register_buffer(v_name, F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3))
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    setattr(self, v_name, F.normalize(torch.matmul(getattr(self, u_name).unsqueeze(1), weights[i]).squeeze(1), dim=1, eps=1e-3))  # bx1xr * bxrxc --> bx1xc --> bxc
                    setattr(self, u_name, F.normalize(torch.matmul(weights[i], getattr(self, v_name).unsqueeze(2)).squeeze(2), dim=1, eps=1e-3))  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(getattr(self, u_name).unsqueeze(1), torch.matmul(weights[i], getattr(self, v_name).unsqueeze(2)))
            loss += torch.sum(sigma)

        return loss / N

    def affine_loss(self):
        loss = 0
        N = 0
        for l in self.all_in_layers:
            loss += torch.max(torch.abs(l.weight))
            N += 1

        return loss / N

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in state_dict.keys():
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            if local_key.startswith("sr_"):
                self.register_buffer(local_key, torch.empty_like(state_dict[key]))
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
