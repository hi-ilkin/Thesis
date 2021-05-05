import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from functools import wraps

from albumentations import Compose, LongestMaxSize, PadIfNeeded
from albumentations.pytorch import ToTensorV2

import config


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} completed in {time.perf_counter() - t:.3f} seconds.")
        return result

    return wrapper


def get_frames(video_path, start=config.START, stop=config.STOP, step=config.STEP, output_type='CV'):
    """
    samples given amount of frames with given frequency either in opencv of PIL Image format
    :param video_path: path to video
    :param start: start of frame sampling
    :param stop:  end of frame sampling if None checks all frames, default : None
    :param step: sampling frequency, default 1
    :param output_type: 'CV' or 'PIL' type of output image
    :return: list of images
    """

    capture = cv2.VideoCapture(video_path)
    frames = []

    total_frames_in_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if stop is None:
        stop = total_frames_in_video

    stop = min(total_frames_in_video, stop)

    for i in range(stop):
        ret = capture.grab()
        if i < start:
            continue

        if i % step == 0:
            ret, frame = capture.retrieve()
            if output_type == 'PIL':
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            elif output_type == 'CV':
                frames.append(frame)
    return frames


def get_frames_with_cv(video_path, convert_bgr=False):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if convert_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return frames


def save_video(frames, input_path, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), fps=30, output_size=None,
               output_path='data/faces'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_name = os.path.basename(input_path)
    output_path = os.path.join(output_path, video_name)

    if output_size is not None:
        w, h = output_size
    else:
        h, w, c = frames[0].shape

    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Total detected frames: {len(frames)}")
    for i, frame in enumerate(frames):
        video_writer.write(cv2.resize(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR), (w, h)))

    video_writer.release()

    print(f"{video_name} saved to {output_path}")


def display(images, names=None):
    if names is None:
        names = [f'image-{i}' for i in range(len(images))]
    for image, name in zip(images, names):
        cv2.imshow(name, image)

    q = cv2.waitKey(0)
    if q == ord('q'):
        exit(-1)
    cv2.destroyAllWindows()


def extract_box(image, coordinates, output_path=None):
    """
    Extracts given coordinates from an image and saves
    :param image: input image
    :param coordinates: coordinates of box.
    :param output_path: (Optional) path to save image
    :return: extracted image, if output_path is not provided
    """
    crop = image.crop(coordinates)
    if output_path is not None:
        crop.save(output_path)
    return crop


@timeit
def run_in_parallel(func, args) -> list:
    """
    Runs given function for each of the items in the argument list in parallel.
    Gathers all results and returns
    :param func: Function to run in parallel. Should return a value
    :param args: List of items to pass function
    :return: List, Results of all runs
    """
    results = []
    with Pool(processes=11) as pool:
        for item in args:
            result = pool.apply_async(func, item)
            results.append(result)

        return [result.get() for result in results]


def transform(size):
    return Compose([
        LongestMaxSize(size),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ])


def get_unified_img(p, size=224):
    transforms = transform(size)
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transforms(image=img)['image']