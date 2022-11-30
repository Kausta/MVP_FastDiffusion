import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pytorch_lightning as pl
import lpips

import fd.nn as lnn

from fd.util.config import ConfigType

from fd.models import TranslationVAE

__all__ = ["TranslationVAETrainer"]


class TranslationVAETrainer(pl.LightningModule):
    hparams: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__()

        self.save_hyperparameters(config)

        model_params = self.hparams.model
        self.model = TranslationVAE(
            in_ch=model_params.out_ch,
            cond_ch=model_params.in_ch,
            latent_dim=model_params.latent_dim,
            out_ch=model_params.out_ch,
            encoder_layers=model_params.encoder_channels,
            decoder_layers=model_params.decoder_channels,
            act_fn=nn.SiLU,
            final_act_fn=nn.Identity
        )
        print(self.model)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg') 

    def forward(self, input_dict):
        pred, pred_cond, outs = self.model(input_dict["target"].to(self.device), input_dict["input"].to(self.device))
        return pred, pred_cond, outs

    def inference(self, input_dict, reparametrize=True):
        pred, pred_cond, outs = self.model(
            input_dict["target"].to(self.device), 
            input_dict["input"].to(self.device), 
            stage='train' if reparametrize else 'eval')
        pred.clamp_(min=-1, max=1)
        pred_cond.clamp_(min=-1, max=1)
        return pred, pred_cond, outs

    def loss(self, input_dict, phase="train", kl_mult=1.0, wdn_coeff=1.0):
        input_dict["input"] = input_dict["input"].to(self.device)
        input_dict["target"] = input_dict["target"].to(self.device)

        pred, pred_cond, out = self.forward(input_dict)

        input = input_dict["input"]
        target = input_dict["target"]

        loss_weights = self.hparams.loss
        recon_loss = F.l1_loss(pred, target, reduction="mean") + loss_weights.trans_vae_cond_weight * F.l1_loss(pred_cond, target, reduction="mean")
        
        KL = self.model.kl_divergence(out["z_recon"], out["mean_recon"], out["log_var_recon"], reduction="mean").mean() \
            + loss_weights.trans_vae_cond_weight * self.model.kl_divergence(out["z_cond"], out["mean_cond"], out["log_var_cond"], reduction="mean").mean()
        latent_l1 = F.l1_loss(out["z_cond"], out["z_recon"].detach(), reduction="mean")

        loss = self.hparams.loss.l1_weight * recon_loss + kl_mult * self.hparams.loss.kl_weight * KL + latent_l1

        outs = {
            "loss_recon": recon_loss,
            "loss_KL": KL,
            "loss_latent_l1": latent_l1
        }

        # if self.current_epoch >= self.hparams.optimizer.warmup:
        lpips_loss = self.loss_fn_vgg(pred, target).mean() + loss_weights.trans_vae_cond_weight * self.loss_fn_vgg(pred_cond, target).mean() 
        loss += self.hparams.loss.lpips_weight * lpips_loss
        outs["loss_lpips"] = lpips_loss

        if phase == "train":
            if loss_weights.affine_weight is not None:
                affine_loss = self.model.affine_loss()
                loss += wdn_coeff * loss_weights.affine_weight * affine_loss 
                outs["loss_affine"] = affine_loss

            if loss_weights.spectral_weight is not None:
                spectral_norm_loss = self.model.spectral_norm_loss()
                loss += wdn_coeff * loss_weights.spectral_weight * spectral_norm_loss
                outs["loss_spectral_norm"] = spectral_norm_loss

        outs["loss"] = loss
        return outs

    def log_all(self, out, step="train"):
        on_step = (step == "train")
        for k, v in out.items():
            self.log(f'{step}/{k}', v, on_step=on_step, on_epoch=True)

    def training_step(self, batch, batch_idx):
        kl_mult_epoch = ((self.current_epoch) %
                         self.hparams.loss.kl_cycle) / self.hparams.loss.kl_cycle
        kl_mult_step = ((batch_idx) % self.trainer.num_training_batches) / \
            self.trainer.num_training_batches
        kl_mult = min(1., 2 * (kl_mult_epoch + kl_mult_step /
                      self.hparams.loss.kl_cycle))

        wdn_coeff = (1. - kl_mult) * np.log(1.) + kl_mult * np.log(1e-2)
        wdn_coeff = np.exp(wdn_coeff)

        out = self.loss(batch, phase="train", kl_mult=kl_mult, wdn_coeff=wdn_coeff)
        self.log_all(out, "train")
        self.log("trainer/kl_mult", kl_mult, on_step=True, on_epoch=False)
        self.log("trainer/wdn_coeff", wdn_coeff, on_step=True, on_epoch=False)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.loss(batch, phase="val")
        self.log_all(out, "val")
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.loss(batch, phase="test")
        self.log_all(out, "test")
        return out["loss"]

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer
        opt_name = opt_params.optimizer

        if opt_name == "adam":
            opt = optim.Adam(self.model.parameters(), lr=opt_params.lr,
                             weight_decay=opt_params.wd)
        elif opt_name == "adamw":
            opt = optim.AdamW(self.model.parameters(),
                              lr=opt_params.lr, weight_decay=opt_params.wd)
        elif opt_name == "adamax":
            opt = optim.Adamax(self.model.parameters(),
                              lr=opt_params.lr, weight_decay=opt_params.wd, eps=1e-3)
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

