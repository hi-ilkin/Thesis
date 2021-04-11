import timm
import torch
import wandb
import numpy as np
from torch import nn
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, StepLR
from sklearn.metrics import f1_score, recall_score, precision_score
from pytorch_lightning.metrics import Accuracy, Precision, Recall, MetricCollection, F1


def get_criterion(weights):
    if weights is None or type(weights) != list:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())


class DFDCModels(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = self.config.model_name

        if 'efficient' in self.model_name or self.model_name.startswith('densenet'):
            self.model = timm.create_model(self.model_name, pretrained=self.config.load_pretrained)
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, self.config.target_size)
        elif self.model_name == 'deit':
            self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224',
                                        pretrained=self.config.load_pretrained)

            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, self.config.target_size)
        else:
            RuntimeError(f"Unknown model: {self.model_name}")

        self.criterion = get_criterion(self.config.output_weights)
        self.val_metric = MetricCollection(
            [Accuracy(), Precision(is_multiclass=False), Recall(is_multiclass=False), F1(num_classes=2)])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config.opt_name == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.config.lr_max)
        elif self.config.opt_name == 'SGD':
            optimizer = SGD(self.parameters(), self.config.lr_max)
        else:
            raise NameError("Unsupported optimizer is chosen")

        if self.config.lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                    T_0=self.config.lr_t0,
                                                    T_mult=self.config.lr_tmult)
        elif self.config.lr_scheduler == 'cyclic2':
            scheduler = CyclicLR(optimizer,
                                 base_lr=self.config.lr_min,
                                 max_lr=self.config.lr_max,
                                 step_size_up=self.config.lr_step_size,
                                 mode=self.config.lr_mode,
                                 cycle_momentum=False)
        elif self.config.lr_scheduler == 'step':
            scheduler = StepLR(optimizer,
                               step_size=self.config.lr_step_size,
                               gamma=self.config.lr_gamma
                               )
        else:
            raise NameError(
                'Wrong scheduler name. Currently supported: ["cyclic", "cyclic2", "CosineAnnealingWarmRestarts"]')

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        y_preds = self.model(images)
        loss = self.criterion(y_preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx, prefix='val'):
        images, labels = val_batch
        y_preds = self.model(images)
        logits = torch.argmax(torch.sigmoid(y_preds.squeeze()), dim=1)

        loss = self.criterion(y_preds, labels)
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        return {f'{prefix}_loss': loss, 'preds': logits.cpu().numpy(), 'target': labels.cpu().numpy()}

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx, prefix='test')

    def validation_epoch_end(self, outputs, prefix='val'):
        preds = np.concatenate([tmp['preds'] for tmp in outputs])
        targets = np.concatenate([tmp['target'] for tmp in outputs])

        metrics = self.get_metrics_with_sklearn(preds, targets, prefix=prefix)
        self.log_dict(metrics)

        if prefix == 'val':
            conf_mat_name = 'conf_mat'  # for backward compatibility
        elif prefix == 'test':
            conf_mat_name = 'test_conf_mat'
            wandb.log({"Test results": wandb.Table(data=[list(metrics.values())], columns=list(metrics.keys()))})
        else:
            conf_mat_name = 'conf_other_mat'

        wandb.log(
            {conf_mat_name: wandb.plot.confusion_matrix(y_true=targets, preds=preds, class_names=['fake', 'real'])})

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, prefix='test')

    @staticmethod
    def get_metrics_with_sklearn(y_preds, targets, prefix='val'):
        """
        Sklearn implementation of metrics.
        Returns : F1_score, recall and precision for each class
        """

        res = {}
        for c in range(2):
            for f in [f1_score, recall_score, precision_score]:
                res[f'{prefix}_{f.__name__}_{c}'] = round(f(targets, y_preds, pos_label=c), 3)
        return res

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward()
