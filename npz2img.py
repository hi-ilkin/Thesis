import glob
import os

import numpy as np
import pandas as pd

import config
from utils import run_in_parallel

output_path = config.root + "/train_images/"
if not os.path.exists(output_path):
    os.mkdir(output_path)


def get_data(p):
    npz = np.load(p, allow_pickle=True)
    return npz['data'], npz['labels']


def save_img(img, label, chunk, idx):
    img_name = f"{chunk}_{idx}.png"
    img.save(f'{output_path}/{img_name}')

    return img_name, label


if __name__ == '__main__':
    df_labels = pd.DataFrame(columns=['labels'])

    for p in glob.glob(config.CHUNK_PATH)[:1]:
        chunk_name = os.path.basename(p).split('.')[0]
        images, labels = get_data(p)

        merged_labels = run_in_parallel(save_img, [(img, label, chunk_name, i) for i, (img, label) in
                                                   enumerate(zip(images, labels))])

        for im_name, l in merged_labels:
            df_labels.loc[im_name] = l

    df_labels.to_csv('labels.csv')
