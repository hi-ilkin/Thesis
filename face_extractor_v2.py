import threading

from facenet_pytorch.models.mtcnn import MTCNN

import config
import glob
import json
import multiprocessing
import os
import pandas as pd
import time
from face_extractor import FaceExtractor
from multiprocessing import Pool, cpu_count
from multiprocessing import Process

from utils import get_frames, save_video

video_paths = {os.path.basename(f): f for f in glob.glob(config.VIDEO_PATH)}
metadata = pd.read_json(config.METADATA_PATH).T

mtcnn = MTCNN(image_size=300, margin=20, min_face_size=60, keep_all=True, post_process=False, device='cuda:0')
frames_pool = {}


def extract_frames(video_names):
    global frames_pool
    for n in video_names:
        video_path = video_paths[n]
        frames_pool[n] = get_frames(video_path)


def extract_face_coordinates(frames):
    face_coordinates = {}
    for i, frame in enumerate(frames):
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        face_coordinates[i] = None

        if boxes is not None:
            face_coordinates[i] = boxes.tolist()

    return {'face_coordinates': face_coordinates}


def save_face_coordinates(total_video_count):
    global frames_pool
    try:
        with open(config.FACE_COORDINATES_PATH, 'r') as f:
            face_coordinates = json.load(f)
    except:
        face_coordinates = {}

    counter = 0
    while True:
        total_waiting_time = 0
        pool_size = len(frames_pool)

        while pool_size == 0 and total_waiting_time < 10:
            time.sleep(1)
            total_waiting_time += 1
            pool_size = len(frames_pool)
            print(f"Waiting for frames... {total_waiting_time}")

        if total_waiting_time >= 10:
            print(f"Total waiting time exceeded. Finishing execution!")
            break

        name = list(frames_pool.keys())[0]
        frames = frames_pool.pop(name)

        if name not in list(face_coordinates.keys()):
            coord = extract_face_coordinates(frames)
            face_coordinates[name] = coord['face_coordinates']

        if counter % 5 == 0:
            with open(config.FACE_COORDINATES_PATH, 'w') as f:
                json.dump(face_coordinates, f)

        if (counter + 1) % 50 == 0:
            print(f"Videos checked: {counter + 1}/{total_video_count}")

        counter += 1

    with open(config.FACE_COORDINATES_PATH, 'w') as f:
        json.dump(face_coordinates, f)


def extract_faces_using_coordinates(path_to_video):
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


def extract_face_coordinates_from_original_videos():
    originals = metadata[metadata['label'] == 'REAL'].index.values
    print(f"{len(originals)} of {len(metadata)} are original videos.")

    threading.Thread(target=extract_frames, args=(originals,)).start()

    save_face_coordinates(len(originals))


if __name__ == '__main__':
    extract_face_coordinates_from_original_videos()

    # with Pool(4) as pool:
    #     pool.map(crop_faces_using_original_boxes, video_paths)
