#!/usr/bin/env python
# coding: utf-8

# # Training sample model with DeiT: Data-efficient Image Transformer

# ## Useful soruces:
# 
# * [DeiT classifier for Cassava dataset](https://www.kaggle.com/huseynlilkin/cnn-or-transformer-pytorch-xla-tpu-for-cassava/edit)
# * [Kaggle Utilty Script](https://www.kaggle.com/huseynlilkin/kaggle-pytorch-utility-script/edit?rvi=1)
# * [Face-net Pytorch for face detection](https://www.kaggle.com/huseynlilkin/guide-to-mtcnn-in-facenet-pytorch/edit)
# * [DeepFake starter kit](https://www.kaggle.com/huseynlilkin/deepfake-starter-kit/edit)
# * [Has nice helper functions](https://www.kaggle.com/huseynlilkin/my-deep-fake-solution/edit)
# * [LR reduce - good webiste](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/#reduce-on-loss-plateau-decay-patience0-factor01)
# * [Selim's code](https://github.com/selimsef/dfdc_deepfake_challenge)
# * [DeiT github page](https://github.com/facebookresearch/deit)
# * [AffinityPropagation for clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation)

# In[1]:


import os
import cv2
import glob
import time


import numpy as np


import timm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Resize, Normalize
from albumentations.pytorch import ToTensorV2

import config


# CONFIG
from utils import timeit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'deit_base_patch16_224'  # other model names ['deit_base_patch16_224', 'vit_base_patch16_384', 'resnext50_32x4d', 'tf_efficientnet_b3_ns']
LOAD_PRETRAINED = True
TARGET_SIZE = 2
LOAD_CHECKPOINT = False

LR = 1e-3
# lr scheduler
MODE = 'min'
FACTOR = 0.1
PATIENCE = 1

EPOCHS = 5
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() - 1


# ## Dataset

class DFDCDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.names = df['name'].values
        self.label_names = df['label'].values
        self.labels = np.where(self.label_names == 'REAL', 1, 0)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()

        file_name = self.names[idx]
        file_path = f'{config.root}/train_faces/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


class DFDCDatasetNPZ(Dataset):
    """
    Loads data from numpy npz files
    """

    def __init__(self, npz_path, transform=None):
        self.npz = np.load(npz_path, allow_pickle=True)
        self.data = self.npz['data']
        self.label_names = self.npz['labels']
        self.labels = np.where(self.label_names == 'REAL', 1, 0)
        self.transform = transform

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).long()
        image = np.asarray(self.data[idx])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.data)


# Original transform functions from Selim's code
def create_train_transforms_by_selim(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )


def create_val_transforms_by_selim(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


# My transforms
def create_train_transforms(size=224):
    return Compose([
        Resize(224, 224),
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        # TODO: IsotropicResize
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), HueSaturationValue()], p=0.7),  # FancyPCA() is missing
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_val_transforms(size=224):
    return Compose([
        Resize(224, 224),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# ## Models

class DeiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', MODEL_NAME, pretrained=LOAD_PRETRAINED)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, TARGET_SIZE)

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    def __init__(self, version='b0'):
        super().__init__()
        if version in ['b0', 'b1', 'b2', 'b3']:
            model_name = f'efficientnet_{version}'
        else:
            model_name = f'tf_efficientnet_{version}'
        self.model = timm.create_model(model_name, pretrained=LOAD_PRETRAINED)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, TARGET_SIZE)

    def forward(self, x):
        return self.model(x)


# - **TODO: CHECK HOW THEY TRAINED DEIT ORIGINALLY**
# - **TODO: NEXT TIME, DOWNLOAD OTHER NOTEBOOK AND OPEN LOCALLY**


class AverageMeter(object):
    """Computers and stores the average oand current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, scheduler, scaler, epoch, loss


def get_validation_loader(path_to_chunk):
    validation_dataset = DFDCDatasetNPZ(path_to_chunk, create_val_transforms())
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=True,
                                   drop_last=True)

    return validation_loader


def get_train_loader(path_to_chunk):
    train_dataset = DFDCDatasetNPZ(path_to_chunk, create_train_transforms())
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True)

    return train_loader


# train and validation loop
if __name__ == '__main__':

    train_loss = []
    validation_loss = []
    loss = 0
    model = EfficientNet('b4')
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode=MODE, factor=FACTOR, patience=PATIENCE, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([2 / 10, 8 / 10]).cuda())
    scaler = GradScaler()
    chunks = glob.glob(os.path.dirname(config.CHUNK_PATH) + '/*')

    # using last chunk as validation
    validation_chunk_path = chunks.pop()
    print(f"Chunk {os.path.basename(validation_chunk_path)} was used as validation")
    validation_loader = get_validation_loader(validation_chunk_path)

    if LOAD_CHECKPOINT:
        model, optimizer, scheduler, scaler, epoch, loss = load_checkpoint(config.BEST_MODEL_PATH, model,
                                                                           optimizer, scheduler, scaler)

    for epoch in range(EPOCHS):
        t = time.time()
        model.train()
        for chunk in chunks:
            train_loader = get_train_loader(chunk)
            print(
                f"Train starts for : EPOCH: {epoch} CHUNK: {os.path.basename(chunk)} Number of iterations: {len(train_loader)}")
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                batch_size = labels.size(0)

                with autocast():
                    y_preds = model(images)
                    loss = criterion(y_preds, labels)
                    train_loss.append(loss.item())
                    scaler.scale(loss).backward()
                    # TODO: should we do gradient clipping here?
                    # grad_norm = clip_grad_norm_(model.parameters, 1e-7)
                    # TODO: what is a gradient accumulation?
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path=config.CHECKPOINT_PATH)
        print(f'Train : Epoch {epoch} completed in {time.time() - t:.2f} secs')
        print(f'Last train loss: {train_loss[-1]} AVG train loss: {np.mean(train_loss)}')

        # validation
        model.eval()
        tmp_loss = []
        for step, (images, labels) in enumerate(validation_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            with torch.no_grad():
                y_preds = model(images)
            v_loss = criterion(y_preds, labels)
            tmp_loss.append(v_loss.item())

        cur_val_loss = np.mean(tmp_loss)
        validation_loss.append(cur_val_loss)
        prev_loss = 99 if len(validation_loss) < 2 else validation_loss[-2]
        print(f"Validation loss: {cur_val_loss} previous validation loss: {prev_loss}")

        if cur_val_loss < prev_loss:
            print(f"Average validation loss improved from {prev_loss} to {cur_val_loss}. Saving checkpoint")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss,
                            path = config.BEST_MODEL_PATH)

        scheduler.step(np.mean(validation_loss))

    # 20 epochs 2h 40m

    # plt.figure(figsize=(24, 12))
    print(
        f"Training completed: average training loss: {np.mean(train_loss)}, average validation loss: {np.mean(validation_loss)}")
    # plt.plot(train_loss)
    # plt.plot(validation_loss)

    np.savez('efb0-losses.npz', train=train_loss, validation=validation_loss)
