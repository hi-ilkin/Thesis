import timm
import pytorch_lightning as pl
from torchvision import models
from torch import nn


class DFDCSmallModels(pl.LightningModule):
    def __init__(self, model_name, target_size=2):
        super().__init__()
        self.model_name = model_name
        self.target_size = target_size

        if 'efficient' in self.model_name or self.model_name.startswith('densenet') \
                or self.model_name.startswith('mobilenet'):
            self.model = timm.create_model(self.model_name)
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, self.target_size))

        elif self.model_name in ['xception', 'resnet50']:
            self.model = timm.create_model(self.model_name)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, self.target_size))

        elif self.model_name == 'inception_resnet_v2':
            self.model = timm.create_model(self.model_name)
            n_features = self.model.classif.in_features
            self.model.classif = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, self.target_size))

        elif self.model_name == 'inception_v4':
            self.model = timm.create_model(self.model_name)
            n_features = self.model.last_linear.in_features
            self.model.last_linear = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, self.target_size))

        elif self.model_name.startswith('densenet161'):
            self.model = models.densenet161(pretrained=self.load_pretrained)
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, self.target_size))

        else:
            RuntimeError(f"Unknown model: {self.model_name}")

    def forward(self, x):
        return self.model(x)
