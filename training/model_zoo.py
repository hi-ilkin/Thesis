import timm
import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_criterion(weights):
    if weights is None or type(weights) != list:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())


class Models(pl.LightningModule):
    def __int__(self):
        super(Models, self).__int__()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.lr_min)
        print(f"{self.config.lr_t0} x {self.config.lr_tmult}")
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=self.config.lr_t0,
                                                T_mult=self.config.lr_tmult)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

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
        print(f"LR: {optimizer.param_groups[0]['lr']}")
        loss.backward()


class EfficientNet(Models):
    def __init__(self, config, version='b0'):
        super().__init__()
        self.config = config
        if version in ['b0', 'b1', 'b2', 'b3']:
            model_name = f'efficientnet_{version}'
        else:
            model_name = f'tf_efficientnet_{version}'

        self.model = timm.create_model(model_name, pretrained=self.config.load_pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, self.config.target_size)
        self.criterion = get_criterion(self.config.output_weights)


class DeiT(Models):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224',
                                    pretrained=self.config.load_pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, self.config.traget_size)
        self.criterion = get_criterion(self.config.output_weights)
