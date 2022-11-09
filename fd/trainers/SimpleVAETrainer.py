import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

import fd.nn as lnn

from fd.util.config import ConfigType

from fd.models import SimpleVAE

__all__ = ["SimpleVAETrainer"]


class SimpleVAETrainer(pl.LightningModule):
    hparams: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__()

        self.save_hyperparameters(config)

        model_params = self.hparams.model
        self.model = SimpleVAE(
            in_ch=model_params.in_ch,
            latent_dim=model_params.latent_dim,
            out_ch=model_params.out_ch,
            image_h=model_params.image_h,
            image_w=model_params.image_w,
            encoder_layers=model_params.encoder_channels,
            decoder_layers=model_params.decoder_channels,
            prior=lnn.StandardNormalPrior(model_params.latent_dim),
            act_fn=nn.SiLU,
            final_act_fn=nn.Tanh
        )

    def forward(self, input_dict):
        pred, outs = self.model(input_dict["input"].to(self.device))
        return pred, outs

    def inference(self, input_dict):
        pred, outs = self.model(
            input_dict["input"].to(self.device), stage='eval')
        return pred, outs

    def sample(self, num_samples):
        pred, outs = self.model.sample(num_samples)
        return pred, outs

    def loss(self, input_dict, kl_mult=1.0):
        pred, out = self.forward(input_dict)

        recon_loss = F.l1_loss(pred, out["target"].to(
            self.device), reduction="sum")
        KL = self.model.kl_divergence(
            out["z"], out["mean"], out["log_var"], reduction="sum")

        loss = self.hparams.loss.l1_weight * recon_loss \
            + kl_mult * self.hparams.loss.kl_weight * KL

        return {
            "loss": loss,
            "loss_recon": recon_loss,
            "loss_KL": KL
        }

    def log_all(self, out, step="train"):
        on_step = (step == "train")
        for k, v in out.items():
            self.log(f'{step}/{k}', v, on_step=on_step, on_epoch=True)

    def training_step(self, batch, batch_idx):
        kl_mult_epoch = ((self.current_epoch + 1) %
                         self.hparams.loss.kl_cycle) / self.hparams.loss.kl_cycle
        kl_mult_step = ((batch_idx + 1) % self.trainer.num_training_batches) / \
            self.trainer.num_training_batches
        kl_mult = max(1., 2 * (kl_mult_epoch + kl_mult_step /
                      self.hparams.loss.kl_cycle))

        out = self.loss(batch, kl_mult=kl_mult)
        self.log_all(out, "train")
        self.log("trainer/kl_mult", kl_mult, on_step=True, on_epoch=False)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.loss(batch)
        self.log_all(out, "val")
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.loss(batch)
        self.log_all(out, "test")
        return out["loss"]

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer
        opt_name = opt_params.optimizer

        if opt_name == "adam":
            opt = optim.Adam(self.parameters(), lr=opt_params.lr,
                             weight_decay=opt_params.wd)
        elif opt_name == "adamw":
            opt = optim.AdamW(self.parameters(),
                              lr=opt_params.lr, weight_decay=opt_params.wd)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        if opt_params.scheduler is None:
            return opt

        if opt_params.scheduler == "cosine_warmup":
            sched = lnn.get_cosine_schedule_with_warmup(
                opt, opt_params.warmup * self.trainer.num_training_batches, self.trainer.estimated_stepping_batches)
        else:
            raise ValueError(f"Unknown scheduler {opt_params.scheduler}")

        return [opt], [{
            "scheduler": sched,
            "interval": opt_params.sched_interval
        }]
