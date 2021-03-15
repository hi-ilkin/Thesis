import glob
import json
import math
import os
import pandas as pd

from ipywidgets import Video
from sklearn.cluster import AgglomerativeClustering

import config
from utils import get_frames, timeit, run_in_parallel, extract_box

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


metadata = Metadata()


class VideoReader:
    def __init__(self, name):

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

        for frame_id, label, face in zip(self.frame_ids, self.cluster_labels, faces):
            label = int(label)
            if clusters.get(label, None) is None:
                clusters[label] = []
            clusters[label].append((frame_id, face))
        return clusters

    def extract_faces(self):
        for coordinates in self.coordinates:
            try:
                for i, (frame_id, face) in enumerate(coordinates):
                    extract_box(self.frames[frame_id], face, f'{config.FACE_IMAGES}/{self.name}_{frame_id}_{i}.jpg')
            except Exception as e:
                if math.isnan(coordinates):
                    continue
                print(f'Problem with {self.name} : {e}')

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


def clean_face_coordinates():
    """
    - Cluster faces using face centers
    - Keep single clusters
    - Remove clusters with less than 10 items
    - Save new face coordinates
    """
    results = run_in_parallel(run, face_coordinates.index)
    cleaned_faces_path = os.path.dirname(config.FACE_COORDINATES_PATH) + '/cleaned_faces_48_higher_th.json'

    cleaned_faces = {}
    for res in results:
        cleaned_faces.update(res)
    with open(cleaned_faces_path, 'w') as f:
        json.dump(cleaned_faces, f)


def _run(name):
    video = VideoReader(name)
    video.extract_faces()
    return None


def run_face_extraction():
    videos = [os.path.basename(f) for f in glob.glob(config.VIDEO_PATH)]
    run_in_parallel(_run, videos)


@timeit
def prepare_labels():
    face_images = [os.path.basename(f) for f in glob.glob(config.FACE_IMAGES + '/*.jpg')]
    labels = {}

    for i, f in enumerate(face_images):
        if (i+1) % 1000 == 0:
            print(f"{i + 1}/{len(face_images)}")
        name = f.split('_')[0]
        labels[f] = metadata[name][0]

    pd.DataFrame(labels.items(), columns=['name', 'label']).to_csv('labels.csv', index=False)


if __name__ == '__main__':
    # clean_face_coordinates()
    # run_face_extraction()
    prepare_labels()
