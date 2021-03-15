import os

root = 'D:/DFDC'
part = 48

VIDEO_PATH = f'{root}/dfdc_train_part_{part}/*.mp4'
METADATA_PATH = f'{root}/dfdc_train_part_{part}/metadata.json'

FACE_COORDINATES_PATH = f'{root}/coordinates/cleaned_faces_{part}_higher_th.json'
FACE_IMAGES = f'{root}/outputs/faces_{part}_train'
DIR_CLUSTERS = f'{root}/outputs/clusters_{part}'

DIR_COMPARED_FACES = f'{root}/outputs/compare_{part}'

dirs_to_create = [DIR_CLUSTERS, DIR_COMPARED_FACES, FACE_IMAGES]

for d in dirs_to_create:
    if not os.path.exists(d):
        os.makedirs(d)