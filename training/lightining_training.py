from pytorch_lightning import seed_everything

import config
from training.datasets import DFDCLightningDataset
from training.model_zoo import EfficientNet, DeiT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    seed_everything(99)
    model = EfficientNet(version='b0')
    # model = DeiT()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='D:/DFDC/checkpoints',
        filename='sample-efnb0-{epoch:02d}-{val_loss:.2f}',
        mode='min',
    )
    dataset = DFDCLightningDataset()
    trainer = pl.Trainer(gpus=1, precision=16,
                         accumulate_grad_batches=4,
                         progress_bar_refresh_rate=20,
                         reload_dataloaders_every_epoch=True,
                         check_val_every_n_epoch=20,
                         resume_from_checkpoint=r'D:\DFDC\checkpoints\epoch=79-step=1039.ckpt',
                         callbacks=[checkpoint_callback],
                         max_epochs=40 * 3
                         )

    trainer.fit(model, dataset)
