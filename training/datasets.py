import glob
from random import shuffle
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

import config
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Resize, Normalize
from albumentations.pytorch import ToTensorV2


class DFDCDatasetNPZ(Dataset):
    """
    Loads data from numpy npz files
    """

    def __init__(self, npz_path, mode='train'):
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=True)
        self.data = self.npz['data']
        self.label_names = self.npz['labels']
        self.labels = np.where(self.label_names == 'REAL', 1, 0)

        self.transform_mode = mode
        if mode is not None:
            self.transformer = self.get_transformer(mode=mode)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()
        image = np.asarray(self.data[idx])
        if self.transform_mode:
            augmented = self.transformer(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_transformer(mode, size=224):
        if mode == 'train':
            return Compose([
                Resize(size, size),
                ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                # TODO: IsotropicResize
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), HueSaturationValue()], p=0.7),  # FancyPCA() is missing
                ToGray(p=0.2),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT,
                                 p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        elif mode == 'valid':
            return Compose([
                Resize(size, size),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])


class DFDCLightningDataset(pl.LightningDataModule):

    def __init__(self):
        super(DFDCLightningDataset, self).__init__()
        self.train_paths = None
        self.valid_path = None
        self.current_chunk_idx = 0
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        self.train_paths = glob.glob(config.CHUNK_PATH)
        self.valid_path = self.train_paths.pop(-1)

        shuffle(self.train_paths)
        print(f'Validation: {self.valid_path}')

    def train_dataloader(self):
        self.dataset = DFDCDatasetNPZ(self.train_paths[self.current_chunk_idx], mode='train')
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            num_workers=11,
            pin_memory=True
        )

        self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.train_paths)
        return loader

    def val_dataloader(self):
        self.dataset = DFDCDatasetNPZ(self.valid_path, mode='valid')
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=False,
            num_workers=11,
            pin_memory=True
        )
        return loader
