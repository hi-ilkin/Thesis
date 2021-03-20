import glob
import json
import math
import os
import shutil

import numpy as np
import pandas as pd

from datetime import datetime
from IPython.display import Video
from sklearn.cluster import AgglomerativeClustering

import config
from utils import get_frames, timeit, run_in_parallel, extract_box


class Metadata:
    def __init__(self):
        self.metadata = pd.read_json(config.METADATA_PATH).T

    def __getitem__(self, name):
        if name not in self.metadata.index:
            return None, None
        return self.metadata.loc[name][['label', 'original']].values

    def fakes(self, original=None):
        fakes = self.metadata[self.metadata['label'] == 'FAKE']
        if original is not None:
            return fakes[fakes['original'] == original]
        return fakes

    def reals(self):
        return self.metadata[self.metadata['label'] == 'REAL']


metadata = Metadata()


class VideoReader:
    def __init__(self, name, face_coordinates=None):

        self.frame_ids = []
        self.name = name
        self.path = os.path.join(os.path.dirname(config.VIDEO_PATH), name)
        self.label, self.original = metadata[name]
        self.frames = get_frames(self.path, output_type='PIL')
        self.face_centers = []
        self.cluster_labels = []

        if self.label == 'REAL':
            self.coordinates = face_coordinates.loc[name].values
        else:
            self.coordinates = face_coordinates.loc[self.original].values

    def __get__(self, frame_id):
        return self.coordinates[frame_id]

    def get_face(self, frame_id, face_id=None):
        faces = []
        for i, c in enumerate(self.coordinates[frame_id]):
            if face_id is not None and i != face_id:
                continue
            faces.append(self.frames[frame_id].crop(c))
        return faces

    def get_all_faces(self):
        faces = []
        for i, frame in enumerate(self.coordinates):
            for face in frame:
                faces.append(self.frames[i].crop(face))
        return faces

    def flatten_face_coordinates(self):
        faces = []
        for i, frame in enumerate(self.coordinates):
            if frame is None or type(frame) != list:
                continue
            for face in frame:
                faces.append(face)

        return faces

    def calculate_face_centers(self):
        for i, frame in enumerate(self.coordinates):
            if frame is None or type(frame) != list:
                print(f'[WARNING] No faces detected in frame {i} of {self.name}')
                continue

            for face in frame:
                self.face_centers.append([(face[0] + face[2]) / 2,
                                          (face[1] + face[3]) / 2])
                self.frame_ids.append(i)

    def _cluster(self, threshold=100, linkage='average'):
        if len(self.face_centers) == 0:
            return

        if len(self.face_centers) == 1:
            self.cluster_labels = [0]
            return
        try:
            self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                          distance_threshold=threshold,
                                                          linkage=linkage).fit(self.face_centers).labels_
        except Exception as e:
            print(self.name, e)
            exit(-1)

    def cluster_faces(self, use_images=False, labels=None):

        clusters = {}

        if labels is not None:
            self.cluster_labels = labels
        else:
            self.calculate_face_centers()
            self._cluster()

        if use_images:
            faces = self.get_all_faces()
        else:
            faces = self.flatten_face_coordinates()

        for frame_id, label, face in zip(self.frame_ids, self.cluster_labels, faces):
            label = int(label)
            if clusters.get(label, None) is None:
                clusters[label] = []
            clusters[label].append((frame_id, face))
        return clusters

    def extract_faces(self):
        """
        Works with clustered faces coordinates
        :return:
        """
        for coordinates in self.coordinates:
            try:
                for i, (frame_id, face) in enumerate(coordinates):
                    extract_box(self.frames[frame_id], face, f'{config.DIR_FACE_IMAGES}/{self.name}_{frame_id}_{i}.jpg')
            except Exception as e:
                if math.isnan(coordinates):
                    continue
                print(f'Problem with {self.name} : {e}')

    def extract_faces_v2(self):
        """
        works with non-clustered face coordinates
        :return:
        """
        faces, labels = [], []

        for coordinate, frame in zip(self.coordinates, self.frames):
            if coordinate is not None and type(coordinate) == list:
                for face in coordinate:
                    faces.append(extract_box(frame, face))
                    labels.append(self.label)
            else:
                print(f'[WARNING] No face detected on {self.path}')
        return faces, labels

    def play(self):
        """
        Only available with jupyter notebook
        :return: Video instance
        """
        return Video(self.path, embed=True, width=640)


def process_clusters(clusters):
    # if single cluster, return
    if len(clusters.keys()) == 1:
        return clusters

    processed_clusters = clusters.copy()
    for k, v in clusters.items():
        if len(v) < 10:
            del processed_clusters[k]

    return processed_clusters


def run(name):
    video = VideoReader(name)
    clusters = process_clusters(video.cluster_faces())
    return {name: clusters}


@timeit
def clean_face_coordinates(face_coordinates):
    """
    - Cluster faces using face centers
    - Keep single clusters
    - Remove clusters with less than 10 items
    - Save new face coordinates
    """
    print(f'[{datetime.now()}] Removing redundant face clusters from Part-{config.part}')
    results = run_in_parallel(run, face_coordinates.index)

    cleaned_faces = {}
    for res in results:
        cleaned_faces.update(res)
    with open(config.CLEANED_FACE_COORDINATES_PATH, 'w') as f:
        json.dump(cleaned_faces, f)


def _run(name):
    face_coordinates = pd.read_json(config.FACE_COORDINATES_PATH).T
    video = VideoReader(name, face_coordinates)
    return video.extract_faces_v2()


@timeit
def run_face_extraction():
    print(f'[{datetime.now()}] Extracting detected face images from Part-{config.part}')
    videos = [os.path.basename(f) for f in glob.glob(config.VIDEO_PATH)]
    results = run_in_parallel(_run, videos)

    all_faces_in_chunk = []
    all_labels_in_chunk = []

    for res in results:
        faces, labels = res
        all_faces_in_chunk.extend(faces)
        all_labels_in_chunk.extend(labels)

    np.savez(config.CHUNK_PATH, data=all_faces_in_chunk, labels=all_labels_in_chunk)


@timeit
def prepare_labels():
    print(f'[{datetime.now()}] Preparing labels for Part-{config.part}')
    face_images = [os.path.basename(f) for f in glob.glob(config.DIR_FACE_IMAGES + '/*.jpg')]
    labels = {}

    for i, f in enumerate(face_images):
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(face_images)}")
        name = f.split('_')[0]
        label = metadata[name][0]
        if label is None:
            continue
        labels[f] = label

    pd.DataFrame(labels.items(), columns=['name', 'label']).to_csv(config.FACE_LABELS_PATH, index=False)


if __name__ == '__main__':
    run_face_extraction()
    # clean_face_coordinates()
    # prepare_labels()
    # shutil.make_archive('train_faces', 'zip', config.DIR_FACE_IMAGES)
    # videos = glob.glob(config.VIDEO_PATH)
    # n = os.path.basename(videos[0])
    # print(n, len(videos))
    # _run(n)
