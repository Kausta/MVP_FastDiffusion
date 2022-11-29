import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field

import yaml
from omegaconf import OmegaConf, DictConfig

__all__ = ["FDConfig", "ConfigType", "load_config", "print_config",
           "ModelConfig", "OptimizerConfig", "DataConfig", "TrainerConfig",
           "MiscConfig", "PLTrainerConfig", "LossConfig", "DiffusionConfig", 
           "PhaseBetaScheduleConfig", "BetaScheduleConfig"]


@dataclass
class ModelConfig:
    in_ch: int = 3
    latent_dim: int = 256
    out_ch: int = 3
    image_h: int = 256
    image_w: int = 256
    encoder_channels: List[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256])
    decoder_channels: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32, 16])


@dataclass
class OptimizerConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    wd: float = 1e-2
    scheduler: Union[str, None] = "cosine_warmup"
    sched_interval: str = "step"
    warmup: int = 5


@dataclass
class DataConfig:
    data_root: str = str(Path("~/data").expanduser())
    dataset_cls: str = "CelebaHQColorizationDataset"
    dataset: str = "celeba_hq_256"
    split: str = "celebahq"
    train_batch_size: int = 64
    val_batch_size: int = 16
    test_batch_size: int = 16
    train_workers: int = 4
    val_workers: int = 4
    test_workers: int = 4


@dataclass
class LossConfig:
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    kl_weight: float = 1.0
    kl_cycle: int = 25
    lpips_weight: float = 1.0
    cond_weight: Union[float, None] = None
    cond_latent_weight: Union[float, None] = None


@dataclass
class PLTrainerConfig:
    accerelator: str = "gpu"
    devices: int = 1
    cudnn_benchmark: bool = True
    log_freq: int = 50
    monitor: str = "val/loss_recon"


@dataclass
class TrainerConfig:
    trainer: str = "SimpleAETrainer"
    pl: PLTrainerConfig = PLTrainerConfig()
    max_epochs: int = 100
    out_dir: str = str(Path("~/outputs/simpleae").expanduser())


@dataclass
class MiscConfig:
    seed: int = 42

@dataclass
class PhaseBetaScheduleConfig:
    schedule: str = "linear"
    n_timestep: int = 2000
    linear_start: float = 1e-6
    linear_end: float = 0.01

@dataclass 
class BetaScheduleConfig:
    train: PhaseBetaScheduleConfig = PhaseBetaScheduleConfig("linear", 2000, 1e-6, 0.01)
    test: PhaseBetaScheduleConfig = PhaseBetaScheduleConfig("linear", 1000, 1e-4, 0.09)

@dataclass 
class DiffusionConfig:
    ae_ckpt: str = str(Path("~/ckpt/simpleae/simpleae_recon.ckpt").expanduser())
    # ae_config: str = "./config/simpleae_recon.yaml"
    beta_schedule: BetaScheduleConfig = BetaScheduleConfig()
    sample_num: int = 8
    ema_decay: float = 0.9999

@dataclass
class FDConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossConfig = LossConfig()
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
    misc: MiscConfig = MiscConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    project: str = "fast_diffusion"
    group: str = "default"


ConfigType = Union[FDConfig, DictConfig]


def load_config(yaml_path: Union[str, bytes, os.PathLike]) -> ConfigType:
    schema = OmegaConf.structured(FDConfig)
    conf = OmegaConf.load(yaml_path)
    return OmegaConf.merge(schema, conf)


def print_config(config: ConfigType):
    cont = OmegaConf.to_container(config)
    print(yaml.dump(cont, allow_unicode=True, default_flow_style=False), end='')
