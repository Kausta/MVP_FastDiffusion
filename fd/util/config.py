import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field

import yaml
from omegaconf import OmegaConf, DictConfig

__all__ = ["FDConfig", "ConfigType", "load_config", "print_config",
           "ModelConfig", "OptimizerConfig", "DataConfig", "TrainerConfig",
           "MiscConfig", "PLTrainerConfig", "LossConfig"]


@dataclass
class ModelConfig:
    in_ch: int = 3
    latent_dim: int = 256
    out_ch: int = 3
    image_h: int = 32
    image_w: int = 32
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
    batch_size: int = 64


@dataclass
class LossConfig:
    l1_weight: float = 1.0
    kl_weight: float = 1.0
    kl_cycle: int = 25


@dataclass
class PLTrainerConfig:
    accerelator: str = "gpu"
    devices: int = 1
    cudnn_benchmark: bool = True
    log_freq: int = 50


@dataclass
class TrainerConfig:
    trainer: str = "SimpleVAETrainer"
    pl: PLTrainerConfig = PLTrainerConfig()
    max_epochs: int = 100


@dataclass
class MiscConfig:
    seed: int = 42


@dataclass
class FDConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossConfig = LossConfig()
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
    misc: MiscConfig = MiscConfig()
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