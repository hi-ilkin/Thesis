import os
import cv2
import glob
import numpy as np
import pandas as pd
import config
import logging
from skimage.metrics import structural_similarity

from utils import display

metadata = pd.read_json(config.METADATA_PATH).T
video_names = os.listdir(config.FACE_IMAGES)


def save_mean_face(images_path):
    images = []
    shapes = []

    for p in images_path:
        im = cv2.imread(p)
        images.append(im)
        shapes.append(im.shape)

    mean_h, mean_w, _ = np.mean(shapes, axis=0, dtype='int')
    images = [cv2.resize(img, (mean_w, mean_h)).astype('int') for img in images]
    basedir = os.path.dirname(images_path[0])

    mean_face = np.mean(images, axis=0)
    cv2.imwrite(f'{basedir}/mean_face.jpg', mean_face)


if __name__ == '__main__':
    for p in glob.glob(f'{config.FACE_IMAGES}/*.mp4'):
        print(p)
        try:
            save_mean_face(glob.glob(f'{p}/*.jpg'))
        except Exception as e:
            print(e)
