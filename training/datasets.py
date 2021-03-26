import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

import config
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Resize, Normalize
from albumentations.pytorch import ToTensorV2


class DFDCDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.names = df['name'].values
        self.label_names = df['label'].values
        self.labels = np.where(self.label_names == 'REAL', 1, 0)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()

        file_name = self.names[idx]
        file_path = f'{config.root}/train_faces/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


class DFDCDatasetNPZ(pl.LightningDataModule):
    """
    Loads data from numpy npz files
    """

    def __init__(self, npz_path):
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=True)
        self.data = self.npz['data']
        self.label_names = self.npz['labels']
        self.labels = np.where(self.label_names == 'REAL', 1, 0)
        self.transform = self.transform('train')

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()
        image = np.asarray(self.data[idx])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.data)

    def transform(self, mode, size=224):
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