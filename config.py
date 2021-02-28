import os

VIDEO_PATH = r'E:\Thesis\Thesis\data\dfdc_train_part_48\*.mp4'
METADATA_PATH = r'E:\Thesis\Thesis\data\dfdc_train_part_48\metadata.json'

FACE_COORDINATES_PATH = 'data/face_coordinates_48.json'
FACE_IMAGES = 'faces_48'
DIR_CLUSTERS = 'clusters_48'

DIR_COMPARED_FACES = 'compare_48'

if not os.path.exists(DIR_CLUSTERS):
    os.makedirs(DIR_CLUSTERS)


if not os.path.exists(DIR_COMPARED_FACES):
    os.makedirs(DIR_COMPARED_FACES)