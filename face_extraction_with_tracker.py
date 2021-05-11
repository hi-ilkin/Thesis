import glob
import os
import pickle

import numpy as np

import iou
import config

from utils import timeit
from utils import get_frames
from facenet_pytorch.models.mtcnn import MTCNN

mtcnn = MTCNN(**config.FACE_DETECTOR_KWARGS)
iou_threshold = 0.75
output_path = f'{config.root}/tracked_videos'


@timeit
def get_resized_frames(p):
    frames = get_frames(p, start=0, step=1, output_type='PIL')
    return frames
    # tmp_frame = frames[0]
    # new_width, new_height = tmp_frame.width // 2, tmp_frame.height // 2

    # return [frame.resize((new_width, new_height)) for frame in frames]


def get_last_faces(face_id, frame_id, faces, loss_tolerance=2):
    """
    @param face_id: id of latest detected face
    @param frame_id: id of currently processing frame
    @param faces: dictionary of detected faces
    @param loss_tolerance: acceptable missed frame count
    """

    last_faces = []
    last_face_ids = []

    for fid in range(face_id):
        last_face, prev_frame_id = faces[fid][-1]

        # loosing same face for 2 frames is ok
        if frame_id - prev_frame_id <= loss_tolerance:
            last_faces.append(last_face)
            last_face_ids.append(fid)

    return last_faces, last_face_ids


@timeit
def detect_and_track(frames):
    coordinates, probabilities = mtcnn.detect(frames)
    face_id = 0
    faces = {}

    for frame_id, frame in enumerate(coordinates):

        last_faces, last_face_ids = get_last_faces(face_id, frame_id, faces)
        for c in frame:
            if len(last_faces) == 0:
                faces[face_id] = [(c, frame_id)]
                face_id += 1

            else:
                all_iou, iou_max, nmax = iou.get_max_iou(np.array(last_faces), c)
                if iou_max > iou_threshold:
                    last_face_id = last_face_ids[nmax]
                    faces[last_face_id].append((c, frame_id))
                else:
                    faces[face_id] = [(c, frame_id)]
                    face_id += 1
    return faces


@timeit
def main():
    video_paths = glob.glob(f'{config.root}/videos_test/*.mp4')
    os.makedirs(output_path, exist_ok=True)

    for i, vp in enumerate(video_paths[:2]):
        print(f'{i}/{len(video_paths)} {vp}')
        frames = get_resized_frames(vp)
        tracked_faces = detect_and_track(frames)
        with open(f'{output_path}/{os.path.basename(vp)}.pkl', 'wb') as f:
            pickle.dump(tracked_faces, f)


if __name__ == '__main__':
    main()
