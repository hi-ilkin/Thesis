import os
import glob

import cv2
import numpy as np
from utils import get_frames, get_metadata, save_video, get_frames_with_cv

fake_paths = {os.path.basename(p): p for p in glob.glob('data/faces/*.mp4')}
original_paths = {os.path.basename(p): p for p in glob.glob('data/faces_48/*.mp4')}
metadata = get_metadata()


def compare(path1, path2):
    frames_original = get_frames_with_cv(path1, True)
    frames_fake = get_frames_with_cv(path2, True)
    # edge detection filter
    kernel = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 4.0, -1.0],
                       [0.0, -1.0, 0.0]])
    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

    diff = np.abs(np.subtract(frames_original, frames_fake))

    gray_original = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_original]
    gray_fake = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_fake]
    grayscale_diff = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in np.abs(np.subtract(gray_original, gray_fake))]

    diff_red = diff.copy()
    diff_green = diff.copy()
    diff_blue = diff.copy()

    # r g b
    diff_red[:, :, :, 1] = 0
    diff_red[:, :, :, 2] = 0

    diff_green[:, :, :, 0] = 0
    diff_green[:, :, :, 2] = 0

    diff_blue[:, :, :, 0] = 0
    diff_blue[:, :, :, 1] = 0

    np.mean(diff_red)

    combined = [np.hstack(frames) for frames in
                zip(frames_original, frames_fake, diff_red, diff_green, diff_blue, grayscale_diff)]
    save_video(combined, path1, output_path='combined')


if __name__ == '__main__':
    counter = 50
    for key, value in metadata.items():

        if value['label'] == 'FAKE':
            original = original_paths[value['original']]
            fake = fake_paths[key]
            print(f"Comparing {original} and {fake}")
            compare(original, fake)
            counter -= 1

        if not counter:
            break
