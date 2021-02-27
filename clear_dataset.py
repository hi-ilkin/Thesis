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


def diff_images(p1, p2):
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)

    score, ssim_diff = structural_similarity(img1, img2, multichannel=True, full=True)
    diff = abs(img1.astype('int') - img2.astype('int'))
    print(score)

    cv2.imshow('diff', np.hstack((img1, img2, diff)).astype('uint8'))
    q = cv2.waitKey(0)

    if q == ord('q'):
        exit(-1)

    cv2.destroyAllWindows()
    return np.hstack((img1, img2, diff, ssim_diff * 255)).astype('uint8')


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


def save_face_comparision():
    for fake_video_name in video_names:
        ret, real_video_name = should_process_video(fake_video_name)
        if not ret:
            continue

        fake_images = glob.glob(f'{config.FACE_IMAGES}/{fake_video_name}/*.jpg')
        real_images = glob.glob(f'{config.FACE_IMAGES}/{real_video_name}/*.jpg')

        assert len(fake_images) == len(
            real_images), f'Image counts does not match: fake: {fake_video_name}, original: {real_video_name}'

        p = f'compare/{fake_video_name}'
        if not os.path.exists(p):
            os.mkdir(p)
        for r, f in zip(real_images, fake_images):
            combined = diff_images(r, f)
            cv2.imwrite(f'{p}/{os.path.basename(f)}', cv2.resize(combined, (0, 0), fx=2, fy=2))

if __name__ == '__main__':
    # save_face_comparision()

    for p in glob.glob(f'{config.FACE_IMAGES}/*.mp4'):
        print(p)
        try:
            save_mean_face(glob.glob(f'{p}/*.jpg'))
        except Exception as e:
            print(e)
