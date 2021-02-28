import os
import cv2
import glob
import numpy as np
import pandas as pd
import config
import logging
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt

from utils import display

print(f'{config.METADATA_PATH} exists? {os.path.exists(config.METADATA_PATH)}')
metadata = pd.read_json(config.METADATA_PATH).T
video_names = os.listdir(config.FACE_IMAGES)


def hist(img, title=''):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.title(title)
    plt.show()


def should_process_video(video_name):
    if video_name not in metadata.index:
        logging.warning(f'{video_name} does not have metadata')
        return False, None
    if metadata.loc[video_name]['label'] == 'REAL':
        return False, None

    real_video_name = metadata.loc[video_name]['original']
    if real_video_name not in video_names:
        logging.warning(f'{video_name} does not have original video {real_video_name} in the coordinates')
        return False, None

    return True, real_video_name


def diff_images(p1, p2):
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)

    # score, ssim_diff = structural_similarity(img1, img2, multichannel=True, full=True)
    # print(score)

    diff = abs(img1.astype('int') - img2.astype('int'))
    print(np.count_nonzero(diff))

    # display(np.hstack((img1, img2, diff)).astype('uint8'))
    # hist(diff.astype('uint8'), title='Histogram of diference')
    # hist(img1, title='Histogram of original image')
    # hist(img2, title='Histogram of fake image')

    return np.hstack((img1, img2, diff)).astype('uint8')


def save_face_comparision():
    for fake_video_name in video_names:
        ret, real_video_name = should_process_video(fake_video_name)
        if not ret:
            continue

        fake_images = glob.glob(f'{config.FACE_IMAGES}/{fake_video_name}/*.jpg')
        real_images = glob.glob(f'{config.FACE_IMAGES}/{real_video_name}/*.jpg')

        assert len(fake_images) == len(
            real_images), f'Image counts does not match: fake: {fake_video_name}, original: {real_video_name}'

        p = f'{config.DIR_COMPARED_FACES}/{fake_video_name}'
        if not os.path.exists(p):
            os.mkdir(p)
        for r, f in zip(real_images, fake_images):
            combined = diff_images(r, f)
            cv2.imwrite(f'{p}/{os.path.basename(f)}', cv2.resize(combined, (0, 0), fx=2, fy=2))


def multiple_face_videos():
    for v in glob.glob(config.FACE_IMAGES + '/*'):
        fake_imgs = glob.glob(v + '/*')
        if len(fake_imgs) < 30:
            continue

        name = os.path.basename(v)
        if metadata.loc[name]['label'] == 'REAL':
            continue

        real_name = metadata.loc[name]['original']
        print(f'Fake: {os.path.basename(v)}, Real: {real_name}, count: {len(fake_imgs)}')

        real_imgs = glob.glob(f'{config.FACE_IMAGES}/{real_name}/*')
        for real, fake in zip(real_imgs, fake_imgs):
            display([diff_images(real, fake)])
