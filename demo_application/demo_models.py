import pytorch_lightning as pl
import timm
from torch import nn


class DFDCSmallModels(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(self.model_name)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, 2))

    def forward(self, x):
        return self.model(x)

