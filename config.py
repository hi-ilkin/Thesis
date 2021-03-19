import os

root = 'D:/DFDC'
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
VIDEO_PATH = f'{root}/dfdc_train_part_{part}/*.mp4'
METADATA_PATH = f'{root}/dfdc_train_part_{part}/metadata.json'

FACE_COORDINATES_PATH = f'{root}/coordinates/face_coordinates_step30_{part}.json'
CLEANED_FACE_COORDINATES_PATH = f'{root}/coordinates/cleaned_coordinates_{part}.json'
FACE_LABELS_PATH = f'{root}/labels_{part}.csv'
DIR_FACE_IMAGES = f'{root}/train_faces_{part}'
DIR_CLUSTERS = f'{root}/outputs/clusters_{part}'

DIR_COMPARED_FACES = f'{root}/outputs/compare_{part}'

dirs_to_create = [DIR_CLUSTERS, DIR_COMPARED_FACES, DIR_FACE_IMAGES]

for d in dirs_to_create:
    if not os.path.exists(d):
        os.makedirs(d)
