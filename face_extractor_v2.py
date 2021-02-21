import threading

from facenet_pytorch.models.mtcnn import MTCNN

import cv2
import numpy as np
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


def get_item_from_pool():
    global frames_pool
    total_waiting_time = 0
    pool_size = len(frames_pool)

    while pool_size == 0 and total_waiting_time < 10:
        time.sleep(1)
        total_waiting_time += 1
        pool_size = len(frames_pool)
        print(f"Waiting for frames... {total_waiting_time}")

    if total_waiting_time >= 10:
        print(f"Total waiting time exceeded. Finishing execution!")
        return None, None

    name = list(frames_pool.keys())[0]
    frames = frames_pool.pop(name)

    return name, frames


def get_face_coordinates():
    try:
        with open(config.FACE_COORDINATES_PATH, 'r') as f:
            face_coordinates = json.load(f)
    except:
        face_coordinates = {}

    return face_coordinates


def save_faces_to_json(coordinates):
    with open(config.FACE_COORDINATES_PATH, 'w') as f:
        json.dump(coordinates, f)


def save_face_coordinates(total_video_count):
    face_coordinates = get_face_coordinates()
    counter = 0

    while True:
        name, frames = get_item_from_pool()
        if name is None:
            break

        if name not in list(face_coordinates.keys()):
            coord = extract_face_coordinates(frames)
            face_coordinates[name] = coord['face_coordinates']

        if counter % 5 == 0:
            save_faces_to_json(face_coordinates)

        if (counter + 1) % 50 == 0:
            print(f"Videos checked: {counter + 1}/{total_video_count}")

        counter += 1
    save_faces_to_json(face_coordinates)


def extract_faces_using_coordinates(name):
    """
    Uses pre-detected bounding boxes to crop faces. Reads bounding box information from json file
    :return:
    """
    face_coordinates = get_face_coordinates()
    path_to_video = video_paths[name]

    name_original = name
    if metadata.loc[name]['label'] == 'FAKE':
        name_original = metadata.loc[name]['original']

    if name_original not in face_coordinates.keys():
        return

    faces = face_coordinates[name_original]
    frames = get_frames(path_to_video)

    for i, frame in enumerate(frames):
        if faces.get(str(i), None) is None:
            continue

        face_output = f'{config.FACE_IMAGES}/{name}'

        if not os.path.exists(face_output):
            os.mkdir(face_output)

        for frame_id, frame in enumerate(frames):
            if faces.get(str(frame_id), None) is None:
                continue

            coordinates = faces[str(frame_id)]
            for face_id, c in enumerate(coordinates):
                face = cv2.cvtColor(np.array(frame.crop(c)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{face_output}/{name}_{frame_id}_{face_id}.jpg', face)


def extract_face_coordinates_from_original_videos():
    originals = metadata[metadata['label'] == 'REAL'].index.values
    print(f"{len(originals)} of {len(metadata)} are original videos.")
    threading.Thread(target=extract_frames, args=(originals,)).start()
    save_face_coordinates(len(originals))


if __name__ == '__main__':
    # extract_face_coordinates_from_original_videos()

    with Pool(4) as pool:
        pool.map(extract_faces_using_coordinates, video_paths.keys())
