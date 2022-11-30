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
        out = pl_module.inference(self.input_dict)
        recons = out[0]
        recons = (recons.cpu() + 1.) / 2.
        grays = (self.input_dict["input"].cpu() + 1.) / 2.
        originals = (self.input_dict["target"].cpu() + 1.) / 2.

        images, inps, gts = [], [], []
        for recon, gray, original in zip(recons, grays, originals):
            images.append(wandb.Image(self._to_image(recon)))
            inps.append(wandb.Image(self._to_image(gray)))
            gts.append(wandb.Image(self._to_image(original)))

        results = {
            "recons": images,
            "inps": inps,
            "gts": gts,
            "global_step": trainer.global_step
        }
        
        if len(out) > 2:
            conds = out[1]
            conds = (conds.cpu() + 1.) / 2.
            conds = [wandb.Image(self._to_image(img)) for img in conds]
            results["conds"] = conds

        if hasattr(pl_module, "inference_ema"):
            ema_recons, _ = pl_module.inference_ema(self.input_dict)
            ema_recons = (ema_recons.cpu() + 1.) / 2.
            emas = [wandb.Image(self._to_image(img)) for img in ema_recons]
            results["emas"] = emas

        trainer.logger.experiment.log(results)

    def _to_image(self, tensor):
        if tensor.shape[0] == 3:
            return self.transform(tensor)
        elif tensor.shape[0] == 1:
            return self.gray_transform(tensor)
        else:
            raise ValueError("Unknown channel count")
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        with torch.no_grad():
            self._log_samples(trainer, pl_module)

                                                       