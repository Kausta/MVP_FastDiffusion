import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
import lpips

import fd.nn as lnn

from fd.util.config import ConfigType

from fd.models import AEDDPM
from fd.models.simple_ae import Encoder
from fd.trainers import SimpleAETrainer
from fd.util import load_config

__all__ = ["AEDDPMTrainer"]


class AEDDPMTrainer(pl.LightningModule):
    hparams: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__()

        self.save_hyperparameters(config)

        simple_ae_trainer = SimpleAETrainer.load_from_checkpoint(config.diffusion.ae_ckpt)
        simple_ae_trainer.freeze()

        self.ae = simple_ae_trainer.model
        self.cond_encoder = Encoder(
            in_ch = config.model.in_ch,
            latent_dim = config.model.latent_dim,
            channels = config.model.encoder_channels,
            act_fn = nn.SiLU
        )

        model_params = self.hparams.model
        ddpm_h = model_params.image_h // (2 ** (len(model_params.encoder_channels)-1))
        ddpm_w = model_params.image_w // (2 ** (len(model_params.encoder_channels)-1))
        self.model = AEDDPM(
            beta_schedule = config.diffusion.beta_schedule,
            latent_dim = model_params.latent_dim,
            ddpm_h = ddpm_h,
            ddpm_w = ddpm_w
        )
        self.model_ema = lnn.EMA(self.model, self.hparams.diffusion.ema_decay)

        self.current_noise_schedule = "train"
        self.model.set_new_noise_schedule(device=self.device, phase="train")

        self.ema_noise_schedule = "test"
        self.model_ema.module.set_new_noise_schedule(device=self.device, phase="test")

        if self.hparams.loss.cond_weight is not None:
            self.loss_fn_vgg = lpips.LPIPS(net='vgg') 

        self.weight_init()

    def weight_init(self):
        for m in self.cond_encoder.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _update_noise_schedule(self, phase) -> None:
        if self.current_noise_schedule != phase:
            self.model.set_new_noise_schedule(device=self.device, phase=phase)
            self.current_noise_schedule = phase

    def inference(self, input_dict):
        self._update_noise_schedule("test")

        cond = input_dict["input"].to(self.device)
        z_cond = self.cond_encoder(cond)
        pred, visuals = self.model.restoration(
            z_cond, sample_num=self.hparams.diffusion.sample_num)
        pred = self.ae.decoder(pred)
        # Not used right now, ignore
        # visuals = self.ae.decoder(visuals)
        return pred, visuals
    
    def inference_ema(self, input_dict):
        if self.ema_noise_schedule != "test":
            self.model.set_new_noise_schedule(device=self.device, phase="test")
            self.ema_noise_schedule = "test"
        
        cond = input_dict["input"].to(self.device)
        z_cond = self.cond_encoder(cond)
        pred, visuals = self.model_ema.module.restoration(
            z_cond, sample_num=self.hparams.diffusion.sample_num)
        pred = self.ae.decoder(pred)
        # Not used right now, ignore
        # visuals = self.ae.decoder(visuals)
        return pred, visuals

    def loss(self, input_dict):
        self._update_noise_schedule("train")

        target = input_dict["target"].to(self.device)
        cond = input_dict["input"].to(self.device)

        with torch.no_grad():
            z_target = self.ae.encoder(target)
        z_cond = self.cond_encoder(cond)
        
        loss_noise = self.model(z_target, z_cond) 
        loss = loss_noise
        out = {
            "loss_noise": loss_noise
        }

        if self.hparams.loss.cond_weight is not None:
            pred = self.ae.decoder(z_cond)
            recon_loss = F.mse_loss(pred, target, reduction="mean")
            lpips_loss = self.loss_fn_vgg(pred, target).mean()

            loss = loss + self.hparams.loss.mse_weight * recon_loss + self.hparams.loss.lpips_weight * lpips_loss
            out["loss_recon"] = recon_loss
            out["loss_lpips"] = lpips_loss 

        if self.hparams.loss.cond_latent_weight is not None:
            cond_latent_loss = F.mse_loss(z_cond, z_target, reduction="mean")
            loss = loss + cond_latent_loss
            out["loss_cond_latent"] = cond_latent_loss

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

        parameters = [*self.model.parameters(), *self.cond_encoder.parameters()]
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

