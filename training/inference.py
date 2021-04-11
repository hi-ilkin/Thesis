import os

import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import config
import local_properties
from training.datasets import DFDCLightningDataset
from training.model_zoo import DFDCModels

os.environ['REQUESTS_CA_BUNDLE'] = local_properties.SSL_CERTIFICATE_PATH


# model.load_state_dict(checkpoint['state_dict'])

def headlines():
    with open('config-defaults.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        return data['project']['value'], data['run_name']['value']


def main():
    project, name = headlines()

    runs = {
        "1llb9qpj": ("tf_efficientnet_b0_ns-1llb9qpj-epoch=03-val_loss=0.0168.ckpt", "tf_efficientnet_b0"),
        "5qis6ixz": ("efb0-ns-epoch=02-val_loss=0.0237.ckpt", "tf_efficientnet_b0_ns"),
        "b9nqa6xq": ("efb0-v1-epoch=03-val_loss=0.0115.ckpt", "tf_efficientnet_b0_ns")
        # "aa8dkthj": "model=densenet121-run_id=aa8dkthj-epoch=00-val_loss=0.0891.ckpt"
    }

    fitted = False
    for run_id, (ckpt_name, model_name) in runs.items():
        wandb_logger = WandbLogger(project=project, id=run_id, reinit=True)

        params = wandb_logger.experiment.config
        trainer = Trainer(
            gpus=params.gpus,
            precision=params.precision,
            max_epochs=0
        )
        p = os.path.join(config.CHECKPOINT_PATH, ckpt_name)

        params.update({"model_name": model_name}, allow_val_change=True)
        dataset = DFDCLightningDataset(params)
        model = DFDCModels(params)
        trainer.fit(model, dataset)

        print(f"{p} : {os.path.exists(p)}")
        trainer.test(ckpt_path=p)
        wandb.finish()


if __name__ == '__main__':
    main()
