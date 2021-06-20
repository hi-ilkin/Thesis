import cv2

from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Normalize, LongestMaxSize, CoarseDropout, \
    GridDropout, CenterCrop, IAASharpen
from albumentations.pytorch import ToTensorV2


def get_transformer(mode, size=240):
    if mode == 'train':
        return Compose([
            LongestMaxSize(size),
            ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            OneOf([GaussianBlur(), IAASharpen()], p=0.5),
            HorizontalFlip(),
            OneOf([RandomBrightnessContrast(), HueSaturationValue()], p=0.7),  # FancyPCA() is missing
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10,
                             border_mode=cv2.BORDER_CONSTANT, p=0.5),
            # CoarseDropout(min_holes=2, max_holes=2, max_width=64, max_height=64),
            # GridDropout(holes_number_x=5, holes_number_y=5, random_offset=True, p=0.5),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    elif mode in ['valid', 'test']:
        return Compose([
            LongestMaxSize(size),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            # CenterCrop(height=size, width=size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
