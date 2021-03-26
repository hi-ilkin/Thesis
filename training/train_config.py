import torch

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

EPOCHS = 20
BATCH_SIZE = 128
NUM_WORKERS = 0  # os.cpu_count() - 1
