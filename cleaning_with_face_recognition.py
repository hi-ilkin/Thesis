import os
import glob
import random
import threading
import time

import cv2
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import face_recognition
from collections import Counter
from PIL import Image

from utils import display

random.seed(1)
"""
bvzjkezkms.mp4 - is bad: Counter({1: 9, 0: 7})
bsqgziaylx.mp4 - two faces, 3 clusters: Counter({0: 16, 1: 15, 2: 1})

"""


def get_encoding(p_img):
    img = face_recognition.load_image_file(p_img)
    h1, w1, _ = img.shape
    encoding = face_recognition.face_encodings(img, known_face_locations=[[0, w1, h1, 0]], model='large')[0]
    return encoding


def tmp_show_distance():
    for i in range(1, len(paths)):
        for j in range(0, len(paths)):
            t = time.time()
            img1 = face_recognition.load_image_file(paths[i])
            img2 = face_recognition.load_image_file(paths[j])
            print(f"Face load time: {time.time() - t}")
            h1, w1, _ = img1.shape
            h2, w2, _ = img2.shape

            t = time.time()
            encoding1 = face_recognition.face_encodings(img1, known_face_locations=[[0, w1, h1, 0]], model='large')[0]
            encoding2 = face_recognition.face_encodings(img2, known_face_locations=[[0, w1, h1, 0]], model='large')[0]
            print(f'Encoding time: {time.time() - t}')

            t = time.time()
            distance = face_recognition.face_distance([encoding1], encoding2)
            print(f'Distance{time.time() - t}')
            print(f"Distance between : {os.path.basename(paths[i])} and {os.path.basename(paths[j])} is {distance}")
            display([img1, img2])


def cluster(paths_to_images):
    encodings = [get_encoding(p) for p in paths_to_images]
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.55, linkage='average').fit(encodings)

    threading.Thread(target=save_clusters, args=(clustering.labels_, paths_to_images)).start()


def save_clusters(cluster_labels, paths_to_images):
    if not os.path.exists('clusters'):
        os.mkdir('clusters')

    clusters = {}
    name = f'{os.path.basename(paths_to_images[0])}'

    max_w, max_h = 0, 0

    images = []
    for p in paths_to_images:
        img = cv2.imread(p)
        h, w, c = img.shape
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
        images.append(img)

    empty_image = np.ones((max_h+1, max_w+1, 3), dtype='uint8') * 127
    for label, img in zip(cluster_labels, images):
        if clusters.get(label, None) is None:
            clusters[label] = []

        template = empty_image.copy()
        h, w, _ = img.shape
        template[:h, :w] = img
        clusters[label].append(template)

    max_size = max([len(l) for l in clusters.values()])

    rows = []
    for k, v in clusters.items():
        if len(v) != max_size:
            v.extend([empty_image for _ in range(max_size - len(v))])
        rows.append(np.hstack(v))

    cv2.imwrite(f'clusters/{name}', np.vstack(rows))


if __name__ == '__main__':
    # folders = ['bvzjkezkms.mp4', 'bsqgziaylx.mp4'] + [random.choice(os.listdir('faces')) for _ in range(5)]
    folders = os.listdir('faces')
    for folder in folders:
        if folder.endswith('mp4'):
            print(folder)
            paths = [p for p in glob.glob(f'faces/{folder}/*.jpg') if not p.endswith('mean_face.jpg')]
            cluster(paths)
