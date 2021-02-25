import os
import cv2
import glob
import numpy as np
import pandas as pd
import config
import logging
from skimage.metrics import structural_similarity

metadata = pd.read_json(config.METADATA_PATH).T
video_names = os.listdir(config.FACE_IMAGES)


def compare_images(p1, p2):
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
    return np.hstack((img1, img2, diff, ssim_diff*255)).astype('uint8')


for fake_video_name in video_names:

    if fake_video_name not in metadata.index:
        logging.warning(f'{fake_video_name} does not have metadata')
        continue
    if metadata.loc[fake_video_name]['label'] == 'REAL': continue

    real_video_name = metadata.loc[fake_video_name]['original']
    if real_video_name not in video_names:
        logging.warning(f'{fake_video_name} does not have original video {real_video_name} in the coordinates')
        continue

    fake_images = glob.glob(f'{config.FACE_IMAGES}/{fake_video_name}/*.jpg')
    real_images = glob.glob(f'{config.FACE_IMAGES}/{real_video_name}/*.jpg')

    assert len(fake_images) == len(
        real_images), f'Image counts does not match: fake: {fake_video_name}, original: {real_video_name}'

    if len(fake_images) == 16:
        continue

    p = f'compare/{fake_video_name}'
    if not os.path.exists(p):
        os.mkdir(p)
    for r, f in zip(real_images, fake_images):
        combined = compare_images(r, f)
        cv2.imwrite(f'{p}/{os.path.basename(f)}', cv2.resize(combined, (0, 0), fx=2, fy=2))
