import os
import wandb
import yaml

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import config
from training.datasets import DFDCLightningDataset
from training.model_zoo import EfficientNet, DeiT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ['REQUESTS_CA_BUNDLE'] = '/home/ilkin/Downloads/piktiv.crt'


def headlines():
    with open('config-defaults.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        return data['project']['value'], data['model_name']['value']


def train_fn():
    project, name = headlines()
    wandb_logger = WandbLogger(project=project, name=name)
    params = wandb_logger.experiment.config

    model = EfficientNet(params, version='b0')
    # model = DeiT()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{config.CHECKPOINT_PATH}',
        filename=f'{params.model_name}' + '-{epoch:02d}-{val_loss:.4f}',
        save_last=True,
        mode='min',
    )

    dataset = DFDCLightningDataset(params)
    trainer = pl.Trainer(gpus=params.gpus, precision=params.precision,
                         logger=wandb_logger,
                         accumulate_grad_batches=params.accumulate_grad_batches,
                         reload_dataloaders_every_epoch=True,
                         check_val_every_n_epoch=5,
                         resume_from_checkpoint=None,
                         callbacks=[checkpoint_callback],
                         max_epochs=params.epochs,
                         default_root_dir=config.CHECKPOINT_PATH,
                         )
    trainer.fit(model, dataset)


def tune_hyper_params():
    sweep_config = {

        "method": "random",  # Random search
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "batch_size": {
                "values": [64, 128]
            },
            "precision": {
                "values": [16, 32]
            },
            # "lr": {
            #     "values":[0.001, 0.005, 0.0001, 0.0005]
            # }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='sweep-test')
    wandb.agent(sweep_id, function=train_fn)


if __name__ == '__main__':
    seed_everything(99)

    wandb.login(key='c4e2ebb0c3dbe0876676fb7d125f81c39bbc7367')
    # tune_hyper_params()
    train_fn()
