import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os
import torch
import numpy as np

from fd.util.config import ConfigType

__all__ = ["CelebaHQColorizationDataset", "CelebaHQReconDataset"]

class CelebaHQColorizationDataset(data.Dataset):
    def __init__(self, phase: str, config: ConfigType):
        super().__init__()
        assert phase in ["train", "val", "test"]

        self.data_root = Path(config.data.data_root) / config.data.dataset
        self.phase = phase

        split_dir = Path(__file__).parent.resolve() / "splits" / config.data.split
        with open(split_dir / f"{phase}.txt", "r") as f:
            flist = [x.strip() for x in f.readlines() if x.strip() != '']
        self.flist = flist

        self.image_size = [config.model.image_h, config.model.image_w]
        self.transform = transforms.Compose([
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
        ])
        self.gray_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, index):
        fname = self.flist[index]

        gt_image = Image.open(self.data_root / fname).convert('RGB')
        gt_image = self.transform(gt_image)
        gray_image = self.gray_transform(gt_image)
        gt_image = self.gt_transform(gt_image)

        return {
            "target": gt_image,
            "input": gray_image
        }

class CelebaHQReconDataset(data.Dataset):
    def __init__(self, phase: str, config: ConfigType):
        super().__init__()
        assert phase in ["train", "val", "test"]

        self.data_root = Path(config.data.data_root) / config.data.dataset
        self.phase = phase

        split_dir = Path(__file__).parent.resolve() / "splits" / config.data.split
        with open(split_dir / f"{phase}.txt", "r") as f:
            flist = [x.strip() for x in f.readlines() if x.strip() != '']
        self.flist = flist

        self.image_size = [config.model.image_h, config.model.image_w]
        self.transform = transforms.Compose([
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, index):
        fname = self.flist[index]

        gt_image = Image.open(self.data_root / fname).convert('RGB')
        gt_image = self.transform(gt_image)

        return {
            "target": gt_image,
            "input": gt_image
        }

   
