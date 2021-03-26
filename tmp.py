import glob
import os
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchvision import datasets

import config
from train import DFDCDatasetNPZ


"""
Sources to check:

https://beginers.tech/linux/ml/pytorchlightning
https://github.com/PyTorchLightning/pytorch-lightning/issues/2138
https://pytorch.org/vision/stable/datasets.html
https://pytorch-lightning.readthedocs.io/en/stable/advanced/multiple_loaders.html
"""

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class DFDCDataset(pl.LightningDataModule):

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(self, path):
        super(DFDCDataset, self).__init__()
        self.path = path

    def train_dataloader(self):
        chunks = [DFDCDatasetNPZ(p) for p in glob.glob(self.path + "/*")[:10]]
        concat_dataset = ConcatDataset(*chunks)

        loader = torch.utils.data.DataLoader(
            concat_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=12,
            pin_memory=True
        )
        return loader