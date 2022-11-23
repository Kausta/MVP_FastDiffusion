from functools import partial
from inspect import isfunction
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import fd.nn as lnn
import fd.nn.functional as LF
import fd.nn.modules.guided_diffusion as gd

from .simple_ae import DownBlock, UpBlock

__all__ = ["AEDDPM", "Encoder", "Decoder"]


class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=16, channels=[16, 32, 64, 128], act_fn=nn.SiLU):
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
            nn.Conv2d(channels[-1], out_ch, kernel_size=1, stride=1, padding=0)
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_ch=32, out_ch=3, channels=[128, 64, 32, 16], act_fn=nn.SiLU, final_act_fn=nn.Tanh):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, channels[-1], kernel_size=1, stride=1, padding=0)
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


class AEDDPM(pl.LightningModule):
    gammas: torch.Tensor
    sqrt_recip_gammas: torch.Tensor
    sqrt_recipm1_gammas: torch.Tensor
    
    def __init__(self,
                 beta_schedule,
                 target_ch=3,
                 cond_ch=1,
                 latent_dim=16,
                 encoder_layers=[16, 32, 64, 128],
                 decoder_layers=[128, 64, 32, 16],
                 image_h=32,
                 image_w=32,
                 act_fn=nn.SiLU,
                 final_act_fn=nn.Tanh):
        super().__init__()

        self.beta_schedule = beta_schedule

        ddpm_h = image_h // (2 ** (len(decoder_layers)-1))
        ddpm_w = image_w // (2 ** (len(decoder_layers)-1))

        self.target_encoder = Encoder(target_ch, latent_dim, encoder_layers, act_fn)
        self.cond_encoder = Encoder(cond_ch, latent_dim, encoder_layers, act_fn)
        self.decoder = Decoder(latent_dim, target_ch, decoder_layers, act_fn, final_act_fn=final_act_fn)

        self.ddpm = gd.UNet(
            image_size=(2*latent_dim, ddpm_h, ddpm_w),
            in_channel=2*latent_dim,
            inner_channel=64,
            out_channel=latent_dim,
            res_blocks=2,
            channel_mults=[1, 2, 4],
            attn_res=[4],
            dropout=0.2,
        )

        self.weight_init()

    def weight_init(self):
        for m in [*self.target_encoder.modules(), *self.cond_encoder.modules(), *self.decoder.modules()]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target, cond, stage="train"):
        target_down = self.target_encoder(target)
        cond_down = self.cond_encoder(cond)

        out = self.decoder(z)
        return out, {
            "mean": mean,
            "log_var": log_var,
            "z": z
        }

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.ddpm(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward_ddpm(self, y_0, y_cond=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)


        noise_hat = self.ddpm(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


