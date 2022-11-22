import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import fd.nn as lnn
import fd.nn.functional as LF

__all__ = ["SimpleVAE", "DownBlock", "UpBlock", "Encoder", "Decoder"]


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act_fn=nn.SiLU):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3,
                      padding=1, stride=1, bias=True),
            nn.AvgPool2d(2),
            nn.InstanceNorm2d(in_ch, affine=True),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=1, bias=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,
                      padding=0, stride=1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        out = self.res_block(x)
        out += self.shortcut(x)
        return out / np.sqrt(2)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act_fn=nn.SiLU):
        super().__init__()

        self.pre_res_block = nn.Sequential(
            nn.InstanceNorm2d(in_ch, affine=True),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=1, bias=True),
        )
        self.post_res_block = nn.Sequential(
            nn.InstanceNorm2d(out_ch, affine=True),
            act_fn(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      padding=1, stride=1, bias=True),
        )
        self.shortcut = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        shortcut = F.interpolate(self.shortcut(
            x), scale_factor=2, mode="nearest")
        out = self.pre_res_block(x)
        out = self.post_res_block(F.interpolate(
            out, scale_factor=2, mode="nearest"))
        out += shortcut
        return out / np.sqrt(2)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent_dim=256, channels=[16, 32, 64, 128, 256], act_fn=nn.SiLU):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, channels[0], kernel_size=3,
                      padding=1, stride=1, bias=True)
        ]
        for i in range(1, len(channels)):
            layers.extend([
                DownBlock(channels[i-1], channels[i], act_fn=act_fn),
            ])
        layers.extend([
            nn.InstanceNorm2d(channels[-1], affine=True),
            act_fn(),
            nn.AdaptiveAvgPool2d(1),
            lnn.BatchReshape(-1),
            nn.Linear(channels[-1], 2 * latent_dim)
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        z = self.model(x)
        mean, log_var = torch.chunk(z, 2, dim=-1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_ch=3, channels=[256, 128, 64, 32, 16], init_h=2, init_w=2, act_fn=nn.SiLU, final_act_fn=nn.Tanh):
        super().__init__()

        layers = [
            nn.Linear(latent_dim, channels[0] * init_h * init_w),
            lnn.BatchReshape(channels[0], init_h, init_w),
        ]
        for i in range(1, len(channels)):
            layers.extend([
                UpBlock(channels[i-1], channels[i], act_fn=act_fn),
            ])
        layers.extend([
            nn.InstanceNorm2d(channels[-1], affine=True),
            act_fn(),
            nn.Conv2d(channels[-1], out_ch, kernel_size=1,
                      padding=0, stride=1, bias=True),
            final_act_fn()
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class SimpleVAE(pl.LightningModule):
    def __init__(self,
                 in_ch=3,
                 latent_dim=256,
                 out_ch=3,
                 encoder_layers=[16, 32, 64, 128, 256],
                 decoder_layers=[256, 128, 64, 32, 16],
                 image_h=32,
                 image_w=32,
                 prior: lnn.PriorBase = lnn.StandardNormalPrior(256),
                 act_fn=nn.SiLU,
                 final_act_fn=nn.Tanh):
        super().__init__()

        decoder_init_h = image_h // (2 ** (len(decoder_layers)-1))
        decoder_init_w = image_w // (2 ** (len(decoder_layers)-1))

        self.encoder = Encoder(in_ch, latent_dim, encoder_layers, act_fn)
        self.decoder = Decoder(
            latent_dim, out_ch, decoder_layers, decoder_init_h, decoder_init_w, act_fn, final_act_fn=final_act_fn)
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

    def sample(self, num_samples):
        z = self.prior.sample(
            num_samples, dtype=self.dtype, device=self.device)
        return self.decoder(z), {
            "z": z
        }
