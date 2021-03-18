import shutil

import config
from cleaning_with_coordinate_clustering import run_face_extraction, clean_face_coordinates, prepare_labels, \
    read_face_coordinates
from face_extractor_v2 import extract_face_coordinates_from_original_videos

if __name__ == '__main__':
    # extract_face_coordinates_from_original_videos()
    clean_face_coordinates()
    run_face_extraction()
    prepare_labels()
    shutil.make_archive('train_faces', 'zip', config.DIR_FACE_IMAGES)
