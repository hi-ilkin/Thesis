import os
import wandb

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import config
from training.datasets import DFDCLightningDataset
from training.model_zoo import EfficientNet, DeiT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    wandb.login()
    wandb_logger = WandbLogger(project='DFDC-test', name='test-run')

    seed_everything(99)
    # model = EfficientNet(version='b0')
    model = DeiT()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{config.CHECKPOINT_PATH}',
        filename='sample-deit-{epoch:02d}-{val_loss:.4f}',
        save_last=True,
        mode='min',
    )
    dataset = DFDCLightningDataset()
    trainer = pl.Trainer(gpus=1, precision=16,
                         logger=wandb_logger,
                         accumulate_grad_batches=4,
                         reload_dataloaders_every_epoch=True,
                         check_val_every_n_epoch=20,
                         resume_from_checkpoint=os.path.join(config.CHECKPOINT_PATH, 'sample-deit-epoch=19-val_loss=0.6676.ckpt'),
                         callbacks=[checkpoint_callback],
                         max_epochs=40*20,
                         default_root_dir=config.CHECKPOINT_PATH,
                         )
    trainer.fit(model, dataset)
