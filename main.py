import glob
import shutil
import time
import zipfile
from datetime import datetime
from importlib import reload

import config


def update_config(part):
    with open('config.py', 'r') as f:
        data = f.readlines()

    with open('config.py', 'w') as f:
        data[3] = f'part = {part}\n'
        for line in data:
            f.write(line)


def process():
    for part in range(3, 5):
        update_config(part)
        time.sleep(1)  # wait until new config file was created
        reload(config)  # re-import module

        print(f'[{datetime.now()}] Processing data: {config.part} : zip path: {config.ZIP_FILE_SRC_DIRECTORY}')

        print(f"[{datetime.now()}] Extracting videos from zip file...")
        with zipfile.ZipFile(config.ZIP_FILE_SRC_DIRECTORY, 'r') as zip_ref:
            zip_ref.extractall(config.ZIP_FILE_DST_DIRECTORY)
        print(f'[{datetime.now()}] {len(glob.glob(config.VIDEO_PATH))} FILES EXTRACTED')

        import cleaning_with_coordinate_clustering
        import face_extractor_v2

        reload(cleaning_with_coordinate_clustering)
        reload(face_extractor_v2)

        print(f'[{datetime.now()}] STARTING MAIN PROCESSING')
        # main processing part
        face_extractor_v2.extract_face_coordinates_from_original_videos()
        cleaning_with_coordinate_clustering.run_face_extraction()

        print(f'[{datetime.now()}] Processing completed, removing video files')
        # delete videos
        shutil.rmtree(config.ZIP_FILE_DST_DIRECTORY)


if __name__ == '__main__':
    process()
