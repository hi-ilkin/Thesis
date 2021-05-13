import glob
import os
import pickle
import traceback

import numpy as np

import iou
import config

from utils import timeit, run_in_parallel, extract_box
from utils import get_frames
from facenet_pytorch.models.mtcnn import MTCNN

mtcnn = MTCNN(**config.FACE_DETECTOR_KWARGS)
iou_threshold = 0.75
output_path = f'{config.root}/tracked_videos'


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


def detect_and_track(frames):
    coordinates, probabilities = [], []
    batch_size = 200
    for i in range(0, len(frames), batch_size):
        cds, pbs = mtcnn.detect(frames[i:i + batch_size])
        coordinates.extend(cds)
        probabilities.extend(pbs)

    face_id = 0
    faces = {}

    for frame_id, frame in enumerate(coordinates):

        last_faces, last_face_ids = get_last_faces(face_id, frame_id, faces)
        if frame is None:
            continue

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
def extract_faces(threshold=50, sample_count=50):
    all_data = glob.glob(f'{config.root}/tracked_videos/*.pkl')
    for p_coord in all_data:
        video_name = os.path.basename(p_coord).split('.pkl')[0]
        video_path = f'{config.root}/videos_test/{video_name}'

        with open(p_coord, 'rb') as fp:
            tracking_data = pickle.load(fp)

        if len(tracking_data) == 0:
            print(f'No face detected at video {video_name}')

        freq_coordinates = []
        freq_counts = []
        # we want to sample more frames from most tracked faces
        for k, coordinates in tracking_data.items():
            len_v = len(coordinates)
            if len_v > threshold:
                freq_counts.append(len_v)
                freq_coordinates.append(coordinates)

        if len(freq_coordinates) == 0:
            print(f'No face tracked enough at video {video_name}')

        frames = get_frames(video_path, start=0, step=1, output_type='PIL')
        step = round(sum(freq_counts) / sample_count)
        counter = 0

        for coords in freq_coordinates:
            for i in range(0, len(coords), step):
                coord, frame_id = coords[i]
                extract_box(frames[frame_id], coord, f'{config.TEST_IMAGES}/{video_name}_{counter}.png')
                counter += 1
        print(f'{counter} faces were extracted from video {video_name} from {len(freq_counts)} tracked faces.')

@timeit
def main():
    video_paths = glob.glob(f'{config.root}/videos_test/*.mp4')
    os.makedirs(output_path, exist_ok=True)

    def _loop():
        frames = get_resized_frames(vp)
        tracked_faces = detect_and_track(frames)
        with open(f'{output_path}/{os.path.basename(vp)}.pkl', 'wb') as fp:
            pickle.dump(tracked_faces, fp)

    for i, vp in enumerate(video_paths):
        print(f'{i}/{len(video_paths)} {vp}')
        try:
            _loop()
        except Exception as e:
            print(f' >>> Problem with {vp} : {e}')
            with open('log.txt', 'a') as f:
                f.write(f'>>>> {vp} : {e}\n')
                f.write(traceback.format_exc())
                f.write('\n==============\n')


if __name__ == '__main__':
    # main()
    extract_faces()