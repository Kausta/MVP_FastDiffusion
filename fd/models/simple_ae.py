import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import fd.nn as lnn
import fd.nn.functional as LF

__all__ = ["SimpleAE", "DownBlock", "UpBlock", "Encoder", "Decoder"]

class ResBlock(nn.Module):
    def __init__(self, ch, act_fn=nn.SiLU):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.InstanceNorm2d(ch, affine=True),
            act_fn(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3,
                      padding=1, stride=1, bias=True),
            nn.InstanceNorm2d(ch, affine=True),
            act_fn(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3,
                      padding=1, stride=1, bias=True),
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.res_block(x)
        out += self.shortcut(x)
        return out / np.sqrt(2)

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
                      padding=1, stride=1, bias=True),
            ResBlock(channels[0], act_fn=act_fn)
        ]
        for i in range(1, len(channels)):
            layers.extend([
                DownBlock(channels[i-1], channels[i], act_fn=act_fn),
                ResBlock(channels[i], act_fn=act_fn),
                ResBlock(channels[i], act_fn=act_fn),
            ])
        layers.extend([
            nn.InstanceNorm2d(channels[-1], affine=True),
            act_fn(),
            nn.Conv2d(channels[-1], latent_dim, kernel_size=1, stride=1, padding=0)
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        z = self.model(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_ch=3, channels=[256, 128, 64, 32, 16], act_fn=nn.SiLU, final_act_fn=nn.Tanh):
        super().__init__()

        layers = [
            nn.Conv2d(latent_dim, channels[0], kernel_size=1, stride=1, padding=0),
            ResBlock(channels[0], act_fn=act_fn)
        ]
        for i in range(1, len(channels)):
            layers.extend([
                UpBlock(channels[i-1], channels[i], act_fn=act_fn),
                ResBlock(channels[i], act_fn=act_fn),
                ResBlock(channels[i], act_fn=act_fn),
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


class SimpleAE(pl.LightningModule):
    def __init__(self,
                 in_ch=3,
                 latent_dim=256,
                 out_ch=3,
                 encoder_layers=[16, 32, 64, 128, 256],
                 decoder_layers=[256, 128, 64, 32, 16],
                 act_fn=nn.SiLU,
                 final_act_fn=nn.Tanh):
        super().__init__()

        self.encoder = Encoder(in_ch, latent_dim, encoder_layers, act_fn)
        self.decoder = Decoder(
            latent_dim, out_ch, decoder_layers, act_fn, final_act_fn=final_act_fn)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, {
            "z": z
        }
