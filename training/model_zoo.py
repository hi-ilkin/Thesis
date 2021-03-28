import timm
import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl

import train_config


def get_criterion():
    return nn.CrossEntropyLoss(weight=torch.FloatTensor([2 / 10, 8 / 10]).cuda())


class Models(pl.LightningModule):
    def __int__(self):
        super(Models, self).__int__()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=train_config.LR)
        # TODO: Add scheduler here
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        y_preds = self.model(images)
        loss = self.criterion(y_preds, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        y_preds = self.model(images)
        loss = self.criterion(y_preds, labels)
        self.log('val_loss', loss)
        return loss

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward()


class EfficientNet(Models):
    def __init__(self, version='b0'):
        super().__init__()
        if version in ['b0', 'b1', 'b2', 'b3']:
            model_name = f'efficientnet_{version}'
        else:
            model_name = f'tf_efficientnet_{version}'

        self.model = timm.create_model(model_name, pretrained=train_config.LOAD_PRETRAINED)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, train_config.TARGET_SIZE)
        self.criterion = get_criterion()


class DeiT(Models):
    def __init__(self):
        super().__init__()

        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224',
                                    pretrained=train_config.LOAD_PRETRAINED)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, train_config.TARGET_SIZE)
        self.criterion = get_criterion()
        self.batch_size = None
        self.lr = 0.001
