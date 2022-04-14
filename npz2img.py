import glob
import os
import time

import numpy as np
import pandas as pd

import config
from utils import run_in_parallel, timeit

train_output_path = config.root + "/train_images/"
valid_output_path = config.root + "/val_images/"
test_output_path = config.root + "/test_images/"

for _p in [train_output_path, valid_output_path, test_output_path]:
    if not os.path.exists(_p):
        os.mkdir(_p)


@timeit
def get_data(p):
    npz = np.load(p, allow_pickle=True)
    return npz['data'], npz['labels']


def save_img(img, label, chunk, idx, type='train'):
    img_name = f"{chunk}_{idx}.png"
    if type == 'train':
        img.save(f'{train_output_path}/{img_name}')
    elif type == 'valid':
        img.save(f'{valid_output_path}/{img_name}')
    else:
        img.save(f'{test_output_path}/{img_name}')
    return img_name, label


def process_chunks(chunk_paths, label_path, mode):
    df_labels = pd.DataFrame(columns=['names', 'labels'])
    for p in chunk_paths:
        print(f'Processing {p}: {mode}')
        chunk_name = os.path.basename(p).split('.')[0]
        images, labels = get_data(p)

        merged_labels = run_in_parallel(save_img, [(img, label, chunk_name, i, mode) for i, (img, label) in
                                                   enumerate(zip(images, labels))])

        label_dict = {'names': [], 'labels': []}
        for im_name, l in merged_labels:
            label_dict['names'].append(im_name)
            label_dict['labels'].append(l)
        df_labels = pd.concat([df_labels, pd.DataFrame(label_dict)], ignore_index=True)

    df_labels.to_csv(label_path, index=False)


if __name__ == '__main__':

    test_ids = list(range(41, 50))
    chunks = [os.path.join(os.path.dirname(config.CHUNK_PATH), f'chunk_{t}.npz') for t in test_ids]
    print(chunks)
    # val_chunk = chunks.pop()
    process_chunks(chunks, config.root + '/test_labels.csv', mode='test')
    #process_chunks([val_chunk], config.root + '/val_labels.csv', mode='val')
