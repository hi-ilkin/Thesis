import glob
import json
import multiprocessing
import os
from time import time
from face_extractor import FaceExtractor
from multiprocessing import  Pool, cpu_count

from utils import get_frames, save_video

video_paths = glob.glob('data/dfdc_train_part_48/*.mp4')


metadata = json.load(open('data/dfdc_train_part_48/metadata.json', 'r'))


def detect_faces_and_extract():
    face_extractor = FaceExtractor()
    try:
        with open('data/face_coordinates_48.json', 'r') as f:
            face_coordinates = json.load(f)
    except:
        face_coordinates = {}
    real_video_counter = 0

    for i, p in enumerate(video_paths):
        name = os.path.basename(p)

        if metadata[name]['label'] == 'REAL' and name not in list(face_coordinates.keys()):
            coord = face_extractor.extract_faces(p)
            face_coordinates[name] = coord['face_coordinates']

            with open('data/face_coordinates_48.json', 'w') as f:
                json.dump(face_coordinates, f)

            real_video_counter += 1

        if (i + 1) % 100 == 0:
            print(f"Videos checked: {i + 1}/{len(video_paths)}")

    print(f"{real_video_counter} videos are real from {len(video_paths)}")


def crop_faces_using_original_boxes(path_to_video):
    """
    Uses pre-detected bounding boxes to crop faces. Reads bounding box information from json file
    :param path_to_video:
    :return:
    """
    face_extractor = FaceExtractor(use_mtcnn=False)
    with open('data/face_coordinates_48.json', 'r') as f:
        face_coordinates = json.load(f)

    name = os.path.basename(path_to_video)
    if metadata[name]['label'] != 'FAKE':
        return

    original = metadata[name]['original']
    faces = face_coordinates[original]
    frames = get_frames(path_to_video)

    cropped_faces = []
    for i, frame in enumerate(frames):
        if faces.get(str(i), None) is None:
            continue
        cropped_faces.append(frame.crop(faces[str(i)][0]))
    save_video(cropped_faces, path_to_video, face_extractor.fourcc, face_extractor.FPS, (240, 240))


if __name__ == '__main__':
    # detect_faces_and_extract()
    with Pool(4) as pool:
        pool.map(crop_faces_using_original_boxes, video_paths)
