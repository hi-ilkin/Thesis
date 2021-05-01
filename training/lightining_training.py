import os
import time
import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import config
import local_properties
from training.datasets import DFDCLightningDataset
from training.model_zoo import DFDCModels

os.environ['REQUESTS_CA_BUNDLE'] = local_properties.SSL_CERTIFICATE_PATH


def headlines():
    with open('config-defaults.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        return data['project']['value'], data['run_name']['value']


def train_fn():
    project, name = headlines()
    wandb_logger = WandbLogger(project=project, name=name)
    params = wandb_logger.experiment.config
    run_id = wandb_logger.experiment._run_id

    params.update({'run_id':run_id})
    model = DFDCModels(params)

    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{config.CHECKPOINT_PATH}',
        filename=f'model={params.model_name}-run_id={run_id}' + '-{epoch:02d}-{val_loss:.4f}',
        save_last=True,
        mode='min',
    )

    dataset = DFDCLightningDataset(params)
    trainer = pl.Trainer(gpus=params.gpus, precision=params.precision,
                         logger=wandb_logger,
                         accumulate_grad_batches=params.accumulate_grad_batches,
                         reload_dataloaders_every_epoch=params.use_chunks,
                         check_val_every_n_epoch=params.val_freq,
                         log_every_n_steps=params.log_freq,
                         resume_from_checkpoint=None,
                         callbacks=[checkpoint_callback, lr_monitor_callback],
                         max_epochs=params.epochs,
                         default_root_dir=config.CHECKPOINT_PATH,
                         limit_train_batches=params.limit_train_batches,
                         stochastic_weight_avg=params.swa,
                         deterministic=True
                         )
    trainer.fit(model, dataset)

    print(f'{config.BEST_MODEL_PATH} : {os.path.exists(config.BEST_MODEL_PATH)}')
    trainer.test()


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
    delay = input("Delay: ")
    time.sleep(eval(delay))

    wandb.login(key=local_properties.WANDB_KEY)
    # tune_hyper_params()
    train_fn()
