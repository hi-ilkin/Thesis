import os
import json
import time
from multiprocessing import Pool

import cv2
import mmcv
import numpy as np
from PIL import Image
from functools import wraps


def get_frames(video_path, frame_limit=16, step=1):
    video = mmcv.VideoReader(video_path)
    frames = []
    counter = 0
    for i, frame in enumerate(video):
        if counter == frame_limit:
            break
        if i % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        counter += 1
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


def get_metadata():
    return json.load(open('data/dfdc_train_part_48/metadata.json', 'r'))


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


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} completed in {time.perf_counter() - t:.3f} seconds.")
        return result

    return wrapper

@timeit
def run_in_parallel(func, args):
    results = []
    with Pool(processes=os.cpu_count() - 1) as pool:
        for item in args:
            result = pool.apply_async(func, (item,))
            results.append(result)

        return [result.get() for result in results]
