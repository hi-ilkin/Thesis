import json
import math
import os
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool
from ipywidgets import Video
from sklearn.cluster import AgglomerativeClustering

import config
from utils import get_frames, timeit, run_in_parallel

face_coordinates = pd.read_json(config.FACE_COORDINATES_PATH).T


class Metadata:
    def __init__(self):
        self.metadata = pd.read_json(config.METADATA_PATH).T

    def __getitem__(self, name):
        if name not in self.metadata.index:
            return None, None
        return self.metadata.loc[name][['label', 'original']].values

    def fakes(self):
        return self.metadata[self.metadata['label'] == 'FAKE']

    def reals(self):
        return self.metadata[self.metadata['label'] == 'REAL']


class VideoReader:
    def __init__(self, name, face_coordinates):
        metadata = Metadata()
        self.name = name
        self.path = os.path.join(os.path.dirname(config.VIDEO_PATH), name)
        self.label = metadata[name][0]
        self.original = metadata[name][1]
        self.frames = get_frames(self.path)
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

    def _cluster(self, threshold=100, linkage='average'):
        if len(self.face_centers) == 0:
            return

        self.cluster_labels = AgglomerativeClustering(n_clusters=None,
                                                      distance_threshold=threshold,
                                                      linkage=linkage).fit(self.face_centers).labels_

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

        for label, face in zip(self.cluster_labels, faces):
            label = int(label)
            if clusters.get(label, None) is None:
                clusters[label] = []
            clusters[label].append(face)
        return clusters

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
    video = VideoReader(name, face_coordinates)
    clusters = process_clusters(video.cluster_faces())
    return {name: clusters}


if __name__ == '__main__':

    results = run_in_parallel(run, face_coordinates.index)
    cleaned_faces_path = os.path.dirname(config.FACE_COORDINATES_PATH) + '/cleaned_faces_48_higher_th.json'

    cleaned_faces = {}
    for res in results:
        cleaned_faces.update(res)
    with open(cleaned_faces_path, 'w') as f:
        json.dump(cleaned_faces, f)
