import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

import train_config


class LITEfficientNet(pl.LightningModule):
    def __init__(self, version='b0'):
        super().__init__()
        if version in ['b0', 'b1', 'b2', 'b3']:
            model_name = f'efficientnet_{version}'
        else:
            model_name = f'tf_efficientnet_{version}'

        self.model = timm.create_model(model_name, pretrained=train_config.LOAD_PRETRAINED)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, train_config.TARGET_SIZE)

        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([2 / 10, 8 / 10]).cuda())

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=train_config.LR)
        # TODO: Add scheduler here
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        images = images.to(train_config.DEVICE)
        labels = labels.to(train_config.DEVICE)
        y_preds = self.model(images)
        loss = self.criterion(y_preds, labels)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        images = images.to(train_config.DEVICE)
        labels = labels.to(train_config.DEVICE)
        y_preds = self.model(images)
        loss = self.criterion(y_preds, labels)

        return loss

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward()


model = LITEfficientNet(version='b0')
trainer = pl.Trainer()
trainer.fit(model, )