import os
import pickle
import time
import traceback

from albumentations import Compose, LongestMaxSize, PadIfNeeded
import cv2
import pandas as pd
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN

import config

mtcnn = MTCNN(**config.FACE_DETECTOR_KWARGS)

# read names
# make a prediction
# if no prediction confidence = 0
# save confidences
batch_size = 4096

labels_df = pd.read_csv(config.TRAIN_LABELS)
img_names = labels_df.names.to_list()
labels = labels_df.labels.to_list()


def transform(size=224):
    return Compose([
        LongestMaxSize(size),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT)
    ])


def get_img(im_name):
    im = Image.open(os.path.join(config.TRAIN_IMAGES, im_name))
    return im.resize((160, 160))


def extract_face_confidences():
    start = 0
    results = {'names': [], 'labels': [], 'confidence': []}
    print(f"Number of iterations: {len(img_names) // batch_size}")

    for i in range(len(img_names) // batch_size + 1):
        batch = img_names[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        t = time.time()
        imgs = [get_img(p) for p in batch]

        results['names'].extend(batch)
        results['labels'].extend(batch_labels)

        print(f'Image read: {time.time() - t:.2f}')

        tmp_conf = []
        try:
            t = time.time()
            boxes, probs = mtcnn.detect(imgs)
            print(f"Detection in {time.time() - t:.2f}")
            for name, b, p in zip(img_names, boxes, probs):
                if b is None or p.size != 1:
                    tmp_conf.append(0)
                else:
                    tmp_conf.append(p[0])
            results['confidence'].extend(tmp_conf)

        except Exception as e:
            traceback.print_exc()
            results['confidence'].extend([0] * len(batch))

        start += batch_size

    pd.DataFrame(results).to_csv('labels_with_confidences.csv', index=False)


def make_dataset():
    """
    Helps to create face no-face dataset
    """
    labels_df = pd.read_csv('labels_with_confidences.csv')
    labels_df = labels_df.sort_values('confidence')

    face_dataset = {'names': [], 'labels': []}

    for idx, row in labels_df.iterrows():
        print(row)
        _, name, label, confidence = row
        img = cv2.imread(os.path.join(config.TRAIN_IMAGES, name))
        print(confidence)
        cv2.imshow('img', cv2.resize(img, (0, 0), fx=2, fy=2))
        key = cv2.waitKey()
        if key == 's':
            continue
        elif key == 'f':
            face_dataset['names'].append(name)
            face_dataset['labels'].append(1)
        elif key == 'n':
            face_dataset['names'].append(name)
            face_dataset['labels'].append(0)
        else:
            break

    pd.DataFrame(face_dataset).to_csv('face_no_face_labels.csv', index=False)


if __name__ == '__main__':
    make_dataset()
