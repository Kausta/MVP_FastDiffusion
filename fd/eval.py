import os
from pathlib import Path
import argparse
import json

import torch
import torch.backends.cuda

import torch.nn.functional as F

from torchvision.transforms import functional as TF

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import fd.trainers as trainers
from fd.util import load_config, print_config
import fd.data as data

import time

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainer", type=str, default="VAEDDPMTrainer", help="Trainer class")
    parser.add_argument("-c", "--checkpoint", type=str, default=str(Path("~/ckpt/vaeddpm/vaeddpm.ckpt").expanduser()), help="Checkpoint file")
    parser.add_argument("-o", "--output_dir", type=str, default=str(Path("~/outputs/eval/vaeddpm").expanduser()), help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for k in ["recons", "inps", "gts"]:
        os.makedirs(os.path.join(output_dir, k), exist_ok=True)

    Model = getattr(trainers, args.trainer)
    model = Model.load_from_checkpoint(args.checkpoint).to(device)
    model.freeze()
    model.eval()

    if "DDPM" in args.trainer:
        model.current_noise_schedule = "test"
        model.model.set_new_noise_schedule(device=device, phase="test")

        model.ema_noise_schedule = "test"
        model.model_ema.module.set_new_noise_schedule(device=device, phase="test")
    
    config = model.hparams
    print(">>> Config:")
    print_config(config)
    print("-----------")

    pl.seed_everything(config.misc.seed)

    datamodule = data.DataModule(config, getattr(data, config.data.dataset_cls))
    datamodule.prepare_data()
    datamodule.setup()

    BATCH_SIZE = 64
    test_loader = datamodule.get_dataloader(datamodule.test_set, BATCH_SIZE, False, 4)

    N = 0
    total_ms = 0
    total_mae = 0
    if "DDPM" in args.trainer:
        total_mae_ema = 0
        os.makedirs(os.path.join(output_dir, "emas"), exist_ok=True)
    if "Translation" in args.trainer:
        total_mae_cond = 0
        os.makedirs(os.path.join(output_dir, "conds"), exist_ok=True)

    with torch.inference_mode():
        for i, input_dict in enumerate(test_loader):
            input_dict["input"] = input_dict["input"].to(device)
            input_dict["target"] = input_dict["target"].to(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            out = model.inference(input_dict)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            recons = out[0]
            B = recons.shape[0]
            N += B

            mae = F.l1_loss(recons, input_dict["target"], reduction="mean").item()
            total_mae += mae * B

            elapsed = start.elapsed_time(end)
            total_ms += elapsed
            print(f"Took {elapsed} ms for {B} elems, mae: {mae}, average mae: {total_mae / N}")

            recons = (recons.cpu() + 1.) / 2.
            grays = (input_dict["input"].cpu() + 1.) / 2.
            originals = (input_dict["target"].cpu() + 1.) / 2.

            images, inps, gts = [], [], []
            for recon, gray, original in zip(recons, grays, originals):
                images.append(TF.to_pil_image(recon))
                inps.append(TF.to_pil_image(gray))
                gts.append(TF.to_pil_image(original))

            results = {
                "recons": images,
                "inps": inps,
                "gts": gts,
            }

            if "Translation" in args.trainer:
                conds = out[1]
                cond_mae = F.l1_loss(conds, input_dict["target"], reduction="mean").item()
                total_mae_cond += cond_mae * B
                print(f"Conditional output mae: {cond_mae}, average mae: {total_mae_cond / N}")

                conds = (conds.cpu() + 1.) / 2.
                conds = [TF.to_pil_image(img) for img in conds]
                results["conds"] = conds
            if "DDPM" in args.trainer:
                ema_recons, _ = model.inference_ema(input_dict)
                ema_mae = F.l1_loss(ema_recons, input_dict["target"], reduction="mean").item()
                total_mae_ema += ema_mae * B
                print(f"EMA output mae: {ema_mae}, average mae: {total_mae_ema / N}")
                ema_recons = (ema_recons.cpu() + 1.) / 2.
                emas = [TF.to_pil_image(img) for img in ema_recons]
                results["emas"] = emas

            for k, v in results.items():
                path_base = os.path.join(output_dir, k)
                for j, img in enumerate(v):
                    path = os.path.join(path_base, f"img_{i}_{j}.jpg")
                    img.save(path)

    print(f"{N} outputs")
    print(f"Total/average time in ms: {total_ms}, {total_ms / N}")
    print(f"Average MAE: {total_mae / N}")
    if "DDPM" in args.trainer:
        print(f"Average EMA MAE: {total_mae_ema / N}")
    if "Translation" in args.trainer:
        print(f"Average Cond MAE: {total_mae_cond / N}")


if __name__ == "__main__":
    main()