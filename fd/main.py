import os
import argparse
import json

import torch
import torch.backends.cuda

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import fd.trainers as trainers
from fd.util import load_config, print_config
import fd.data as data

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

    datamodule = data.DataModule(config, getattr(data, config.data.dataset_cls))
    datamodule.prepare_data()
    datamodule.setup()
    val_loader = datamodule.get_dataloader(datamodule.val_set, 8, False, 0)
    recon_dict = next(iter(val_loader))

    wandb_logger = WandbLogger(project=config.project, group=config.group, log_model=False)
    checkpoint_callback = ModelCheckpoint(os.path.join(config.trainer.out_dir, wandb_logger.experiment.name), monitor="val/loss_recon", save_last=True, save_top_k=1)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.pl.accerelator,
        devices=config.trainer.pl.devices,
        benchmark=config.trainer.pl.cudnn_benchmark,
        log_every_n_steps=config.trainer.pl.log_freq,
        callbacks=[
            LearningRateMonitor("step"),
            trainers.ImageSampleLogger(num_samples=8),
            trainers.ImageReconLogger(recon_dict),
            checkpoint_callback
        ]
    )

    Model = getattr(trainers, config.trainer.trainer)
    model = Model(config)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()