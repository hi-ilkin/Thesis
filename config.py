import os

root = 'D:/DFDC'
part = 1

VIDEO_PATH = f'{root}/dfdc_train_part_{part}/*.mp4'
METADATA_PATH = f'{root}/dfdc_train_part_{part}/metadata.json'

FACE_COORDINATES_PATH = f'{root}/coordinates/face_coordinates_{part}.json'
CLEANED_FACE_COORDINATES_PATH = f'{root}/coordinates/cleaned_coordinates_{part}.json'
FACE_LABELS_PATH = f'{root}/labels_{part}.csv'
DIR_FACE_IMAGES = f'{root}/train_faces_{part}'
DIR_CLUSTERS = f'{root}/outputs/clusters_{part}'

DIR_COMPARED_FACES = f'{root}/outputs/compare_{part}'

dirs_to_create = [DIR_CLUSTERS, DIR_COMPARED_FACES, DIR_FACE_IMAGES]

for d in dirs_to_create:
    if not os.path.exists(d):
        os.makedirs(d)