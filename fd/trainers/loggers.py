import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToPILImage

import pytorch_lightning as pl

import wandb

__all__ = ["ImageSampleLogger", "ImageReconLogger"]

class ImageSampleLogger(pl.Callback):
    def __init__(self, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.last_run = None
        self.transform = ToPILImage("RGB")

    def _log_samples(self, trainer, pl_module):
        samples, _ = pl_module.sample(self.num_samples)
        samples = (samples.cpu() + 1.) / 2.
        images = []
        for sample in samples:
            images.append(wandb.Image(self.transform(sample)))

        trainer.logger.experiment.log({
            "samples": images,
            "global_step": trainer.global_step
            })
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._log_samples(trainer, pl_module)

class ImageReconLogger(pl.Callback):
    def __init__(self, input_dict):
        super().__init__()
        self.input_dict = input_dict
        self.last_run = None
        self.transform = ToPILImage("RGB")
        self.gray_transform = ToPILImage("L")

    def _log_samples(self, trainer, pl_module):
        recons, _ = pl_module.forward(self.input_dict)
        recons = (recons.cpu() + 1.) / 2.
        grays = (self.input_dict["input"].cpu() + 1.) / 2.
        originals = (self.input_dict["target"].cpu() + 1.) / 2.

        images, inps, gts = [], [], []
        for recon, gray, original in zip(recons, grays, originals):
            images.append(wandb.Image(self.transform(recon)))
            inps.append(wandb.Image(self.gray_transform(gray)))
            gts.append(wandb.Image(self.transform(original)))

        trainer.logger.experiment.log({
            "recons": images,
            "inps": inps,
            "gts": gts,
            "global_step": trainer.global_step
            })
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._log_samples(trainer, pl_module)

                                                       