import os

VIDEO_PATH = r'D:\DFDC\dfdc_train_part_48\*.mp4'
METADATA_PATH = r'D:\DFDC\dfdc_train_part_48\metadata.json'

FACE_COORDINATES_PATH = 'data/face_coordinates_48.json'
FACE_IMAGES = 'outputs/faces_48'
DIR_CLUSTERS = 'outputs/clusters_48'

DIR_COMPARED_FACES = 'outputs/compare_48'

if not os.path.exists(DIR_CLUSTERS):
    os.makedirs(DIR_CLUSTERS)


if not os.path.exists(DIR_COMPARED_FACES):
    os.makedirs(DIR_COMPARED_FACES)