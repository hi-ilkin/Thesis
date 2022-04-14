import os

import local_properties

root = local_properties.DATA_ROOT_DIR
part = 2

OVERWRITE_FACE_COORDINATES = True

# frame sampling settings
START = 0
STOP = None
STEP = 30

# face detector
FACE_DETECTOR_KWARGS = {
    'image_size': 224,
    'margin': 30,
    'keep_all': True,
    'min_face_size': 60,
    'thresholds': [0.7, 0.8, 0.8],
    'device': 'cuda:0'
}

# paths
VIDEO_PATH = f'{root}/videos/dfdc_train_part_{part}/*.mp4'
METADATA_PATH = f'{root}/videos/dfdc_train_part_{part}/metadata.json'

FACE_COORDINATES_PATH = f'{root}/coordinates/face_coordinates_step30_{part}.json'
CLEANED_FACE_COORDINATES_PATH = f'{root}/coordinates/cleaned_coordinates_{part}.json'
FACE_LABELS_PATH = f'{root}/labels_{part}.csv'
CHUNK_PATH = f'{root}/chunks/chunk_*'
RUN_ID='1mqztpfc'
BEST_MODEL_PATH = f'{root}/checkpoints/model=tf_efficientnet_b4_ns-run_id=1mqztpfc-epoch=00-val_loss=nan.ckpt'
CHECKPOINT_PATH = f'{root}/checkpoints'

TRAIN_IMAGES = f'{root}/train_images'
TRAIN_LABELS = f'{root}/train_sampled.csv'

VAL_IMAGES = f'{root}/val_images'
VAL_LABELS = f'{root}/val_labels.csv'

TEST_IMAGES = f'{root}/Celeb-DF-v2/test'
TEST_LABELS = f'{root}/celeb-df-v2_test.csv'

# TEST_IMAGES = f'{root}/test_images'
# TEST_LABELS = f'{root}/test_labels.csv'

TEST_IMG_OUTPUT = f'{root}/test_results'

zpart = part
if part < 10:
    zpart = f'0{part}'

ZIP_FILE_SRC_DIRECTORY = f'{root}/zipped_dfdc_train_parts/dfdc_train_part_{zpart}.zip'
ZIP_FILE_DST_DIRECTORY = f'{root}/videos'

DIR_CLUSTERS = f'{root}/outputs/clusters_{part}'
DIR_COMPARED_FACES = f'{root}/outputs/compare_{part}'
DIR_FACE_IMAGES = f'{root}/faces'

dirs_to_create = [DIR_CLUSTERS, DIR_COMPARED_FACES, f'{root}/chunks', ZIP_FILE_DST_DIRECTORY, TEST_IMG_OUTPUT]

for d in dirs_to_create:
    if not os.path.exists(d):
        os.makedirs(d)
