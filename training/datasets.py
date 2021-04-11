import glob
import os
from random import shuffle
from typing import Optional

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
import config as path_config

from local_properties import NUM_WORKERS
from training.transformers import get_transformer


class DFDCDatasetImages(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.path = path_config.TRAIN_IMAGES
            self.df = pd.read_csv(path_config.TRAIN_LABELS)
        elif mode == 'valid':
            self.path = path_config.VAL_IMAGES
            self.df = pd.read_csv(path_config.VAL_LABELS)
        elif mode == 'test':
            self.path = path_config.TEST_IMAGES
            self.df = pd.read_csv(path_config.TEST_LABELS)

        self.names = self.df['names'].values
        self.label_names = self.df['labels'].values
        self.labels = np.where(self.label_names == 'REAL', 1, 0)
        self.transform = get_transformer(mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()

        file_name = self.names[idx]
        file_path = os.path.join(self.path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


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
            self.transformer = get_transformer(mode=mode)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()
        image = np.asarray(self.data[idx])
        if self.transform_mode:
            augmented = self.transformer(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.data)


class DFDCLightningDataset(pl.LightningDataModule):

    def __init__(self, config):
        super(DFDCLightningDataset, self).__init__()
        self.config = config
        self.train_paths = None
        self.valid_path = None
        self.current_chunk_idx = 0
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        self.train_paths = glob.glob(path_config.CHUNK_PATH)
        self.valid_path = self.train_paths.pop(-1)

        shuffle(self.train_paths)
        print(f'Validation: {self.valid_path}')

    def train_dataloader(self):
        if self.config.use_chunks:
            self.dataset = DFDCDatasetNPZ(self.train_paths[self.current_chunk_idx], mode='train')
        else:
            self.dataset = DFDCDatasetImages(mode='train')

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        if self.config.use_chunks:
            self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.train_paths)
        return loader

    def val_dataloader(self, mode='valid'):
        if self.config.use_chunks:
            self.dataset = DFDCDatasetNPZ(self.valid_path, mode=mode)
        else:
            self.dataset = DFDCDatasetImages(mode=mode)

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        return self.val_dataloader(mode='test')
