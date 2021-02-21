import glob
import json
import os
import time
import numpy as np
from facenet_pytorch import MTCNN

from utils import get_frames

mtcnn = MTCNN(image_size=224, margin=20, min_face_size=60, keep_all=True, post_process=False, device='cuda')

video_paths = glob.glob(r'train_sample_videos\*.mp4')
metadata = json.load(open(r'train_sample_videos\metadata.json', 'r'))
batch_size = 128

if not os.path.exists('faces'):
    os.mkdir('faces')

for p in video_paths[:1]:
    name = os.path.basename(p).split('.')[0]
    if not os.path.exists(name):
        os.mkdir(name)

    t = time.time()
    frames = get_frames(p)
    print(f"{len(frames)} frames read in {(time.time() - t):.2f} secs.")

    save_paths = [f'faces/{name}/image_{i}.jpg' for i in range(len(frames))]

    batch_processing_times = []
    for i in range(0, len(frames), batch_size):
        t = time.time()

        batch = frames[i:i + batch_size]
        batch_save_path = save_paths[i:i + batch_size]
        res = mtcnn.detect(batch)
        batch_processing_times.append(time.time() - t)

    print(
        f"Total batch processing time : {sum(batch_processing_times):.2f} seconds with "
        f"average {np.mean(batch_processing_times):.2f}, "
        f"minimum {min(batch_processing_times):.2f} and maximum {max(batch_processing_times):.2f} seconds")
