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

runs = {
    'oq5sbbte': ('model=inception_v4-run_id=oq5sbbte-epoch=04-val_loss=0.4404.ckpt', 'inception_v4')
}


def main():
    project, name = headlines()

    fitted = False
    for run_id, (ckpt_name, model_name) in runs.items():
        print(f'{run_id}: {ckpt_name}')
        wandb_logger = WandbLogger(project=project, id=run_id, name=f'{model_name}_{run_id}', reinit=True,
                                   mode='offline')

        params = wandb_logger.experiment.config
        trainer = Trainer(
            gpus=params.gpus,
            precision=params.precision,
            max_epochs=0,
            deterministic=True
        )
        p = os.path.join(config.CHECKPOINT_PATH, ckpt_name)

        params.update({"model_name": model_name, 'run_id': run_id}, allow_val_change=True)
        dataset = DFDCLightningDataset(params)
        model = DFDCModels(params)
        trainer.fit(model, dataset)

        print(f"{p} : {os.path.exists(p)}")
        trainer.test(ckpt_path=p)
        wandb.finish()
        print('=-' * 50, end='\n\n')


if __name__ == '__main__':
    main()
