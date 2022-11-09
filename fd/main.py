import os
import argparse
import json

import torch
import torch.backends.cuda

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import fd.trainers as trainers
from fd.util import load_config, print_config

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/debug.yaml", help="Main Config File")
    args = parser.parse_args()

    config = load_config(args.config)
    print(">>> Config:")
    print_config(config)
    print("-----------")

    pl.seed_everything(config.misc.seed)

    wandb_logger = WandbLogger(project=config.project, group=config.group)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.pl.accerelator,
        devices=config.trainer.pl.devices,
        benchmark=config.trainer.pl.cudnn_benchmark,
        log_every_n_steps=config.trainer.pl.log_freq,
        callbacks=[
            LearningRateMonitor("step")
        ]
    )

    Model = getattr(trainers, config.trainer.trainer)
    model = Model(config)

if __name__ == "__main__":
    main()