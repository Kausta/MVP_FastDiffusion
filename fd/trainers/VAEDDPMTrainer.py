import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
import lpips

import fd.nn as lnn
import fd.nn.functional as LF

from fd.util.config import ConfigType

from fd.models import AEDDPM
from fd.trainers import TranslationVAETrainer
from fd.util import load_config

__all__ = ["VAEDDPMTrainer"]


class VAEDDPMTrainer(pl.LightningModule):
    hparams: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__()

        self.save_hyperparameters(config)

        tvae_trainer = TranslationVAETrainer.load_from_checkpoint(config.diffusion.ae_ckpt)
        tvae_trainer.freeze()

        self.tvae = tvae_trainer.model

        model_params = self.hparams.model
        ddpm_h = model_params.image_h // (2 ** (len(model_params.encoder_channels)-1))
        ddpm_w = model_params.image_w // (2 ** (len(model_params.encoder_channels)-1))
        self.model = AEDDPM(
            beta_schedule = config.diffusion.beta_schedule,
            latent_dim = model_params.latent_dim,
            ddpm_h = ddpm_h,
            ddpm_w = ddpm_w,
            clip_bounds=5,
            concat=self.hparams.diffusion.concat
        )
        self.model_ema = lnn.EMA(self.model, self.hparams.diffusion.ema_decay)

        self.current_noise_schedule = "train"
        self.model.set_new_noise_schedule(device=self.device, phase="train")

        self.ema_noise_schedule = "test"
        self.model_ema.module.set_new_noise_schedule(device=self.device, phase="test")
    
    def _update_noise_schedule(self, phase) -> None:
        if self.current_noise_schedule != phase:
            self.model.set_new_noise_schedule(device=self.device, phase=phase)
            self.current_noise_schedule = phase

    def encode(self, encoder, x, reparametrize=True):
        z_recon = LF.soft_clamp5(encoder(x))
        mean_recon, log_var_recon = torch.chunk(z_recon, 2, dim=1)
        if reparametrize:
            z_recon = self.tvae.reparametrize(mean_recon, log_var_recon)
        else:
            z_recon = mean_recon
        return z_recon

    def inference(self, input_dict):
        self._update_noise_schedule("test")

        cond = input_dict["input"].to(self.device)
        z_cond = self.encode(self.tvae.cond_encoder, cond)
        pred, visuals = self.model.restoration(
            z_cond, sample_num=self.hparams.diffusion.sample_num)
        pred = self.tvae.decoder(pred)
        pred.clamp_(min=-1, max=1)
        # Not used right now, ignore
        # visuals = self.ae.decoder(visuals)
        return pred, visuals
    
    def inference_ema(self, input_dict):
        if self.ema_noise_schedule != "test":
            self.model.set_new_noise_schedule(device=self.device, phase="test")
            self.ema_noise_schedule = "test"
        
        cond = input_dict["input"].to(self.device)
        z_cond = self.encode(self.tvae.cond_encoder, cond)
        pred, visuals = self.model_ema.module.restoration(
            z_cond, sample_num=self.hparams.diffusion.sample_num)
        pred = self.tvae.decoder(pred)
        pred.clamp_(min=-1, max=1)
        # Not used right now, ignore
        # visuals = self.ae.decoder(visuals)
        return pred, visuals

    def loss(self, input_dict):
        self._update_noise_schedule("train")

        target = input_dict["target"].to(self.device)
        cond = input_dict["input"].to(self.device)

        with torch.no_grad():
            z_target = self.encode(self.tvae.encoder, target)
            z_cond = self.encode(self.tvae.cond_encoder, cond)
        
        loss_noise = self.model(z_target, z_cond) 
        loss = loss_noise

        out = {
            "loss_noise": loss_noise
        }

        out["loss"] = loss

        return out

    def log_all(self, out, step="train"):
        on_step = (step == "train")
        for k, v in out.items():
            self.log(f'{step}/{k}', v, on_step=on_step, on_epoch=True)

    def training_step(self, batch, batch_idx):
        out = self.loss(batch)
        self.log_all(out, "train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.loss(batch)
        self.log_all(out, "val")
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.loss(batch)
        self.log_all(out, "test")
        return out["loss"]

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)
        self.model_ema.update(self.model)

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer
        opt_name = opt_params.optimizer

        parameters = self.model.parameters()
        if opt_name == "adam":
            opt = optim.Adam(parameters, lr=opt_params.lr,
                             weight_decay=opt_params.wd)
        elif opt_name == "adamw":
            opt = optim.AdamW(parameters,
                              lr=opt_params.lr, weight_decay=opt_params.wd)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        if opt_params.scheduler is None:
            return opt

        estimated_stepping_batches = self.trainer.estimated_stepping_batches
        if opt_params.scheduler == "cosine_warmup":
            sched = lnn.get_cosine_schedule_with_warmup(
                opt, opt_params.warmup * self.trainer.num_training_batches, estimated_stepping_batches)
        else:
            raise ValueError(f"Unknown scheduler {opt_params.scheduler}")

        return [opt], [{
            "scheduler": sched,
            "interval": opt_params.sched_interval
        }]

