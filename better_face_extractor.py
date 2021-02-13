import glob
import json
import os
import time
import numpy as np
from facenet_pytorch import MTCNN

from utils import get_frames

mtcnn = MTCNN(margin=40, keep_all=True, post_process=False, device='cuda')

video_paths = glob.glob('data/dfdc_train_part_48/*.mp4')
metadata = json.load(open('data/dfdc_train_part_48/metadata.json', 'r'))
batch_size = 16

if not os.path.exists('faces'):
    os.mkdir('faces')

for p in video_paths[:1]:
    name = os.path.basename(p).split('.')[0]
    if not os.path.exists(name):
        os.mkdir(name)

    t = time.time()
    frames = get_frames(p)
    print(f"{len(frames)} frames read in {time.time() - t} secs.")

    save_paths = [f'data/faces/{name}/image_{i}.jpg' for i in range(len(frames))]

    batch_processing_times = []
    for i in range(len(frames)):
        t = time.time()

        batch = frames[i:i + batch_size]
        batch_save_path = save_paths[i:i + batch_size]
        mtcnn(batch, save_path=batch_save_path)

        batch_processing_times.append(time.time() - t)

    print(
        f"Total batch processing time : {sum(batch_processing_times):.2f} seconds with"
        f"average time: {np.mean(batch_processing_times):.2f} seconds"
        f"minimum {min(batch_processing_times):.2f} and maximum {max(batch_processing_times):.2f} seconds")
